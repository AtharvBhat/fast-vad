use crate::vad::{VadError, constants, filterbank};
use wide::f32x8;

const PROBABILITY_EPSILON: f32 = 1e-6;
const MS_PER_SECOND: usize = 1000;

struct VADState {
    noise_floor: f32x8,
    prev_frame: f32x8,
    prev_delta: f32x8,
}

fn init_state() -> VADState {
    VADState {
        noise_floor: f32x8::splat(0.0),
        prev_frame: f32x8::splat(0.0),
        prev_delta: f32x8::splat(0.0),
    }
}

/// Internal configuration derived from user-facing [`VadConfig`] or [`VADModes`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct DetectionConfig {
    min_speech_frames: usize,
    min_silence_frames: usize,
    hangover_frames: usize,
    logit_threshold: f32,
}

/// VAD operating mode controlling sensitivity vs. precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VADModes {
    /// Low false-negative rate; more speech accepted.
    Permissive,
    /// Balanced mode for general use.
    Normal,
    /// Low false-positive rate; stricter speech detection.
    Aggressive,
}

impl VADModes {
    /// Converts an integer index (0=Permissive, 1=Normal, 2=Aggressive) to a `VADModes`.
    pub fn from_index(index: i32) -> Option<Self> {
        match index {
            0 => Some(Self::Permissive),
            1 => Some(Self::Normal),
            2 => Some(Self::Aggressive),
            _ => None,
        }
    }

    /// Returns the integer index for this mode.
    pub fn as_index(self) -> i32 {
        match self {
            Self::Permissive => 0,
            Self::Normal => 1,
            Self::Aggressive => 2,
        }
    }
}

/// Custom detection parameters for VAD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadConfig {
    /// Speech probability threshold in (0, 1).
    pub threshold_probability: f32,
    /// Minimum speech run to confirm onset (milliseconds).
    pub min_speech_ms: usize,
    /// Minimum silence run to confirm offset (milliseconds).
    pub min_silence_ms: usize,
    /// Extra speech extension after voiced region ends (milliseconds).
    pub hangover_ms: usize,
}

fn mode_to_detection_config(mode: VADModes) -> DetectionConfig {
    match mode {
        VADModes::Permissive => DetectionConfig {
            min_speech_frames: 2,
            min_silence_frames: 12,
            hangover_frames: 6,
            logit_threshold: 0.489548,
        },
        VADModes::Normal => DetectionConfig {
            min_speech_frames: 2,
            min_silence_frames: 8,
            hangover_frames: 2,
            logit_threshold: 0.944462,
        },
        VADModes::Aggressive => DetectionConfig {
            min_speech_frames: 4,
            min_silence_frames: 6,
            hangover_frames: 2,
            logit_threshold: 1.208311,
        },
    }
}

fn clamp_probability(probability: f32) -> f32 {
    probability.clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
}

fn probability_to_logit(probability: f32) -> f32 {
    let p = clamp_probability(probability);
    (p / (1.0 - p)).ln()
}

fn ms_to_frames(ms: usize, sample_rate: usize, frame_size: usize) -> usize {
    if ms == 0 {
        return 0;
    }
    let numerator = (ms as u128) * (sample_rate as u128);
    let denominator = (MS_PER_SECOND as u128) * (frame_size as u128);
    numerator.div_ceil(denominator) as usize
}

fn vad_config_to_detection_config(
    config: VadConfig,
    sample_rate: usize,
    frame_size: usize,
) -> DetectionConfig {
    DetectionConfig {
        min_speech_frames: ms_to_frames(config.min_speech_ms, sample_rate, frame_size),
        min_silence_frames: ms_to_frames(config.min_silence_ms, sample_rate, frame_size),
        hangover_frames: ms_to_frames(config.hangover_ms, sample_rate, frame_size),
        logit_threshold: probability_to_logit(config.threshold_probability),
    }
}

fn classify_frame(
    raw_frame_features: f32x8,
    state: &mut VADState,
    logit_thresh: f32,
    is_first_frame: bool,
) -> bool {
    if is_first_frame {
        let logit = (raw_frame_features * constants::RAW_WEIGHTS).reduce_add() + constants::BIAS;
        let res = logit >= logit_thresh;
        state.noise_floor = raw_frame_features;
        state.prev_frame = raw_frame_features;
        state.prev_delta = f32x8::splat(0.0);
        return res;
    }

    let norm = raw_frame_features - state.noise_floor;
    let delta = raw_frame_features - state.prev_frame;
    let delta2 = delta - state.prev_delta;

    let logit = (raw_frame_features * constants::RAW_WEIGHTS
        + norm * constants::NORM_WEIGHTS
        + delta * constants::DELTA_WEIGHTS
        + delta2 * constants::DELTA2_WEIGHTS)
        .reduce_add()
        + constants::BIAS;

    let res = logit >= logit_thresh;

    if !res {
        state.noise_floor = state.noise_floor
            + (raw_frame_features - state.noise_floor) * f32x8::splat(constants::NOISE_FLOOR_ALPHA);
    }
    state.prev_frame = raw_frame_features;
    state.prev_delta = delta;
    res
}

fn drop_short_regions(labels: &mut [bool], target_val: bool, min_frames: usize) {
    let mut in_region = false;
    let mut start = 0;

    for i in 0..labels.len() {
        if labels[i] == target_val && !in_region {
            start = i;
            in_region = true;
        } else if labels[i] != target_val && in_region {
            if (i - start) < min_frames {
                for item in labels.iter_mut().take(i).skip(start) {
                    *item = !target_val;
                }
            }
            in_region = false;
        }
    }

    if in_region && (labels.len() - start) < min_frames {
        for item in labels.iter_mut().skip(start) {
            *item = !target_val;
        }
    }
}

fn apply_hangover(raw_labels: &[bool], hangover_frames: usize) -> Vec<bool> {
    let mut labels = raw_labels.to_vec();
    let mut counter: usize = 0;
    for i in 0..raw_labels.len() {
        if raw_labels[i] {
            counter = hangover_frames;
        } else if counter > 0 {
            labels[i] = true;
            counter -= 1;
        }
    }
    labels
}

fn post_process_frame_labels(raw_labels: Vec<bool>, config: DetectionConfig) -> Vec<bool> {
    let mut labels = apply_hangover(&raw_labels, config.hangover_frames);
    drop_short_regions(&mut labels, true, config.min_speech_frames);
    drop_short_regions(&mut labels, false, config.min_silence_frames);
    labels
}

fn frame_labels_to_sample_labels(
    frame_labels: &[bool],
    frame_size: usize,
    total_samples: usize,
) -> Vec<bool> {
    if frame_labels.is_empty() {
        return vec![false; total_samples];
    }

    let mut sample_labels = Vec::with_capacity(total_samples);
    for &label in frame_labels {
        sample_labels.extend(std::iter::repeat_n(label, frame_size));
    }

    if sample_labels.len() < total_samples {
        let tail_label = *frame_labels.last().expect("non-empty frame_labels");
        sample_labels.resize(total_samples, tail_label);
    } else if sample_labels.len() > total_samples {
        sample_labels.truncate(total_samples);
    }

    sample_labels
}

fn frame_labels_to_segments(
    frame_labels: &[bool],
    frame_size: usize,
    total_samples: usize,
) -> Vec<[usize; 2]> {
    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut start_sample = 0usize;

    for (i, &label) in frame_labels.iter().enumerate() {
        let frame_start = i * frame_size;
        if label && !in_speech {
            in_speech = true;
            start_sample = frame_start;
        } else if !label && in_speech {
            segments.push([start_sample, frame_start]);
            in_speech = false;
        }
    }

    if in_speech {
        let full_frame_end = frame_labels.len() * frame_size;
        let segment_end = if total_samples > full_frame_end {
            total_samples
        } else {
            full_frame_end
        };
        segments.push([start_sample, segment_end]);
    }

    segments
}

#[derive(Debug, Clone)]
struct OnlineSmoother {
    config: DetectionConfig,
    in_speech: bool,
    speech_run: usize,
    silence_run: usize,
    hangover_left: usize,
}

impl OnlineSmoother {
    fn new(config: DetectionConfig) -> Self {
        Self {
            config,
            in_speech: false,
            speech_run: 0,
            silence_run: 0,
            hangover_left: 0,
        }
    }

    fn apply(&mut self, raw_is_speech: bool) -> bool {
        if raw_is_speech {
            self.speech_run += 1;
            self.silence_run = 0;
            self.hangover_left = self.config.hangover_frames;

            if !self.in_speech && self.speech_run >= self.config.min_speech_frames {
                self.in_speech = true;
            }

            return self.in_speech;
        }

        self.speech_run = 0;

        if !self.in_speech {
            self.silence_run += 1;
            self.hangover_left = 0;
            return false;
        }

        self.silence_run += 1;

        if self.silence_run < self.config.min_silence_frames {
            return true;
        }

        if self.hangover_left > 0 {
            self.hangover_left -= 1;
            return true;
        }

        self.in_speech = false;
        self.silence_run = 0;
        false
    }

    fn reset(&mut self) {
        self.in_speech = false;
        self.speech_run = 0;
        self.silence_run = 0;
        self.hangover_left = 0;
    }
}

#[cfg(test)]
fn apply_online_smoothing(raw_labels: &[bool], config: DetectionConfig) -> Vec<bool> {
    let mut smoother = OnlineSmoother::new(config);
    raw_labels
        .iter()
        .map(|&label| smoother.apply(label))
        .collect()
}

/// Batch voice activity detector.
///
/// Processes a complete audio buffer and returns per-sample or per-frame labels.
/// Config is fixed at construction; use [`with_mode`](Self::with_mode) or
/// [`with_config`](Self::with_config) to control detection behaviour.
pub struct VAD {
    filterbank: filterbank::FilterBank,
    frame_size: usize,
    config: DetectionConfig,
}

impl VAD {
    fn build(sample_rate: usize, config: DetectionConfig) -> Result<Self, VadError> {
        let filterbank = filterbank::FilterBank::new(sample_rate)?;
        let frame_size = filterbank.frame_size();
        Ok(Self {
            filterbank,
            frame_size,
            config,
        })
    }

    /// Creates a `VAD` with the default [`VADModes::Normal`] mode.
    pub fn new(sample_rate: usize) -> Result<Self, VadError> {
        Self::build(sample_rate, mode_to_detection_config(VADModes::Normal))
    }

    /// Creates a `VAD` with an explicit detection mode.
    pub fn with_mode(sample_rate: usize, mode: VADModes) -> Result<Self, VadError> {
        Self::build(sample_rate, mode_to_detection_config(mode))
    }

    /// Creates a `VAD` with custom detection parameters.
    pub fn with_config(sample_rate: usize, config: VadConfig) -> Result<Self, VadError> {
        let filterbank = filterbank::FilterBank::new(sample_rate)?;
        let frame_size = filterbank.frame_size();
        let det_config = vad_config_to_detection_config(config, sample_rate, frame_size);
        Ok(Self {
            filterbank,
            frame_size,
            config: det_config,
        })
    }

    /// Number of samples per analysis frame.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    fn run_frames(&self, audio: &[f32]) -> Vec<bool> {
        let features = self.filterbank.compute_filterbank(audio);
        if features.is_empty() {
            return vec![];
        }
        let mut state = init_state();
        let raw_labels: Vec<bool> = features
            .iter()
            .enumerate()
            .map(|(i, frame)| {
                classify_frame(*frame, &mut state, self.config.logit_threshold, i == 0)
            })
            .collect();
        post_process_frame_labels(raw_labels, self.config)
    }

    /// Returns one `bool` per frame indicating speech presence.
    pub fn detect_frames(&self, audio: &[f32]) -> Vec<bool> {
        self.run_frames(audio)
    }

    /// Returns one `bool` per sample indicating speech presence.
    pub fn detect(&self, audio: &[f32]) -> Vec<bool> {
        let frame_labels = self.run_frames(audio);
        frame_labels_to_sample_labels(&frame_labels, self.frame_size, audio.len())
    }

    /// Returns `[start, end]` sample index pairs for each speech segment.
    pub fn detect_segments(&self, audio: &[f32]) -> Vec<[usize; 2]> {
        let frame_labels = self.run_frames(audio);
        if frame_labels.is_empty() {
            return vec![];
        }
        frame_labels_to_segments(&frame_labels, self.frame_size, audio.len())
    }
}

/// Streaming voice activity detector that processes one frame at a time.
pub struct VadStateful {
    filterbank: filterbank::FilterBank,
    frame_size: usize,
    state: VADState,
    is_first_frame: bool,
    smoother: OnlineSmoother,
}

impl VadStateful {
    fn build(sample_rate: usize, config: DetectionConfig) -> Result<Self, VadError> {
        let filterbank = filterbank::FilterBank::new(sample_rate)?;
        let frame_size = filterbank.frame_size();
        Ok(Self {
            filterbank,
            frame_size,
            state: init_state(),
            is_first_frame: true,
            smoother: OnlineSmoother::new(config),
        })
    }

    /// Creates a `VadStateful` with the default [`VADModes::Normal`] mode.
    pub fn new(sample_rate: usize) -> Result<Self, VadError> {
        Self::build(sample_rate, mode_to_detection_config(VADModes::Normal))
    }

    /// Creates a `VadStateful` with an explicit detection mode.
    pub fn with_mode(sample_rate: usize, mode: VADModes) -> Result<Self, VadError> {
        Self::build(sample_rate, mode_to_detection_config(mode))
    }

    /// Creates a `VadStateful` with custom detection parameters.
    pub fn with_config(sample_rate: usize, config: VadConfig) -> Result<Self, VadError> {
        let filterbank = filterbank::FilterBank::new(sample_rate)?;
        let frame_size = filterbank.frame_size();
        let det_config = vad_config_to_detection_config(config, sample_rate, frame_size);
        Ok(Self {
            filterbank,
            frame_size,
            state: init_state(),
            is_first_frame: true,
            smoother: OnlineSmoother::new(det_config),
        })
    }

    /// Number of samples per frame expected by [`detect_frame`](Self::detect_frame).
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Processes a single frame and returns whether speech is active.
    ///
    /// `frame` must have exactly `frame_size()` samples.
    pub fn detect_frame(&mut self, frame: &[f32]) -> Result<bool, VadError> {
        if frame.len() != self.frame_size {
            return Err(VadError::InvalidFrameLength {
                expected: self.frame_size,
                got: frame.len(),
            });
        }

        let features = self.filterbank.compute_filterbank(frame);
        debug_assert_eq!(features.len(), 1);

        let raw_is_speech = classify_frame(
            features[0],
            &mut self.state,
            self.smoother.config.logit_threshold,
            self.is_first_frame,
        );
        self.is_first_frame = false;

        Ok(self.smoother.apply(raw_is_speech))
    }

    /// Resets internal state so the detector can be reused for a new stream.
    pub fn reset_state(&mut self) {
        self.state = init_state();
        self.is_first_frame = true;
        self.smoother.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DetectionConfig, PROBABILITY_EPSILON, VAD, VADModes, VadConfig, VadStateful,
        apply_online_smoothing, clamp_probability, frame_labels_to_segments,
        mode_to_detection_config, ms_to_frames, probability_to_logit,
    };
    use crate::vad::VadError;

    fn make_audio(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + 11) % 101) as f32 / 50.0) - 1.0)
            .collect()
    }

    fn mode_as_vad_config(mode: VADModes, sample_rate: usize, frame_size: usize) -> VadConfig {
        let cfg = mode_to_detection_config(mode);
        let probability = 1.0 / (1.0 + (-cfg.logit_threshold).exp());
        let frame_ms = (frame_size * 1000) / sample_rate;
        VadConfig {
            threshold_probability: probability,
            min_speech_ms: cfg.min_speech_frames * frame_ms,
            min_silence_ms: cfg.min_silence_frames * frame_ms,
            hangover_ms: cfg.hangover_frames * frame_ms,
        }
    }

    #[test]
    fn probability_clamp_and_logit_conversion_behave_as_expected() {
        assert!((clamp_probability(-1.0) - PROBABILITY_EPSILON).abs() < f32::EPSILON);
        assert!((clamp_probability(2.0) - (1.0 - PROBABILITY_EPSILON)).abs() < f32::EPSILON);

        let low = probability_to_logit(0.0);
        let high = probability_to_logit(1.0);
        assert!(low.is_finite());
        assert!(high.is_finite());
        assert!(low < 0.0);
        assert!(high > 0.0);
    }

    #[test]
    fn ms_to_frames_uses_ceil_for_supported_sample_rates() {
        assert_eq!(ms_to_frames(0, 16000, 512), 0);
        assert_eq!(ms_to_frames(1, 16000, 512), 1);
        assert_eq!(ms_to_frames(32, 16000, 512), 1);
        assert_eq!(ms_to_frames(33, 16000, 512), 2);

        assert_eq!(ms_to_frames(0, 8000, 256), 0);
        assert_eq!(ms_to_frames(1, 8000, 256), 1);
        assert_eq!(ms_to_frames(32, 8000, 256), 1);
        assert_eq!(ms_to_frames(33, 8000, 256), 2);
    }

    #[test]
    fn unsupported_sample_rate_returns_error_for_vad() {
        assert!(matches!(
            VAD::new(44100),
            Err(VadError::UnsupportedSampleRate(44100))
        ));
        assert!(matches!(
            VAD::with_mode(44100, VADModes::Normal),
            Err(VadError::UnsupportedSampleRate(44100))
        ));
        assert!(matches!(
            VAD::with_config(
                44100,
                VadConfig {
                    threshold_probability: 0.5,
                    min_speech_ms: 0,
                    min_silence_ms: 0,
                    hangover_ms: 0
                }
            ),
            Err(VadError::UnsupportedSampleRate(44100))
        ));
    }

    #[test]
    fn unsupported_sample_rate_returns_error_for_vad_stateful() {
        assert!(matches!(
            VadStateful::new(44100),
            Err(VadError::UnsupportedSampleRate(44100))
        ));
        assert!(matches!(
            VadStateful::with_mode(44100, VADModes::Normal),
            Err(VadError::UnsupportedSampleRate(44100))
        ));
        assert!(matches!(
            VadStateful::with_config(
                44100,
                VadConfig {
                    threshold_probability: 0.5,
                    min_speech_ms: 0,
                    min_silence_ms: 0,
                    hangover_ms: 0
                }
            ),
            Err(VadError::UnsupportedSampleRate(44100))
        ));
    }

    #[test]
    fn detect_frame_rejects_wrong_length() {
        let mut vad = VadStateful::new(16000).unwrap();
        let expected = vad.frame_size();
        let frame = vec![0.0f32; expected - 1];

        let err = vad
            .detect_frame(&frame)
            .expect_err("wrong frame length should return an error");

        assert_eq!(
            err,
            VadError::InvalidFrameLength {
                expected,
                got: expected - 1
            }
        );
    }

    #[test]
    fn detect_returns_sample_labels_with_tail_carry_forward() {
        let vad = VAD::with_mode(16000, VADModes::Normal).unwrap();
        let frame_size = vad.frame_size();
        let audio = make_audio(frame_size * 3 + 37);

        let frame_labels = vad.detect_frames(&audio);
        assert_eq!(frame_labels.len(), 3);

        let sample_labels = vad.detect(&audio);
        assert_eq!(sample_labels.len(), audio.len());

        for (i, &label) in frame_labels.iter().enumerate() {
            let start = i * frame_size;
            let end = start + frame_size;
            assert!(sample_labels[start..end].iter().all(|&v| v == label));
        }

        let tail_start = frame_labels.len() * frame_size;
        let tail_label = *frame_labels.last().expect("at least one frame label");
        assert!(sample_labels[tail_start..].iter().all(|&v| v == tail_label));
    }

    #[test]
    fn detect_returns_all_false_when_audio_has_no_full_frames() {
        let vad = VAD::new(16000).unwrap();
        let frame_size = vad.frame_size();
        let audio = make_audio(frame_size - 1);

        let frame_labels = vad.detect_frames(&audio);
        assert!(frame_labels.is_empty());

        let sample_labels = vad.detect(&audio);
        assert_eq!(sample_labels.len(), audio.len());
        assert!(sample_labels.iter().all(|&v| !v));

        let segments = vad.detect_segments(&audio);
        assert!(segments.is_empty());
    }

    #[test]
    fn detect_segments_reconstruct_detect_output() {
        let vad = VAD::with_mode(16000, VADModes::Aggressive).unwrap();
        let frame_size = vad.frame_size();
        let audio = make_audio(frame_size * 5 + 21);

        let sample_labels = vad.detect(&audio);
        let segments = vad.detect_segments(&audio);

        let mut reconstructed = vec![false; audio.len()];
        let mut prev_end = 0usize;
        for [start, end] in segments {
            assert!(start < end);
            assert!(start >= prev_end);
            assert!(end <= audio.len());
            for val in reconstructed.iter_mut().take(end).skip(start) {
                *val = true;
            }
            prev_end = end;
        }

        assert_eq!(reconstructed, sample_labels);
    }

    #[test]
    fn frame_segments_extend_last_speech_through_tail() {
        let frame_size = 512usize;
        let labels = [false, true, true];
        let total_samples = labels.len() * frame_size + 13;
        let segments = frame_labels_to_segments(&labels, frame_size, total_samples);
        assert_eq!(segments, vec![[frame_size, total_samples]]);
    }

    #[test]
    fn with_config_matches_with_mode_for_16k() {
        let frame_size = VAD::new(16000).unwrap().frame_size();
        let audio = make_audio(frame_size * 8 + 99);

        for mode in [VADModes::Permissive, VADModes::Normal, VADModes::Aggressive] {
            let mode_vad = VAD::with_mode(16000, mode).unwrap();
            let cfg_vad =
                VAD::with_config(16000, mode_as_vad_config(mode, 16000, frame_size)).unwrap();

            assert_eq!(
                mode_vad.detect_frames(&audio),
                cfg_vad.detect_frames(&audio)
            );
            assert_eq!(mode_vad.detect(&audio), cfg_vad.detect(&audio));
            assert_eq!(
                mode_vad.detect_segments(&audio),
                cfg_vad.detect_segments(&audio)
            );
        }
    }

    #[test]
    fn with_config_matches_with_mode_for_8k() {
        let frame_size = VAD::new(8000).unwrap().frame_size();
        let audio = make_audio(frame_size * 9 + 47);

        for mode in [VADModes::Permissive, VADModes::Normal, VADModes::Aggressive] {
            let mode_vad = VAD::with_mode(8000, mode).unwrap();
            let cfg_vad =
                VAD::with_config(8000, mode_as_vad_config(mode, 8000, frame_size)).unwrap();

            assert_eq!(
                mode_vad.detect_frames(&audio),
                cfg_vad.detect_frames(&audio)
            );
            assert_eq!(mode_vad.detect(&audio), cfg_vad.detect(&audio));
            assert_eq!(
                mode_vad.detect_segments(&audio),
                cfg_vad.detect_segments(&audio)
            );
        }
    }

    #[test]
    fn stateful_reset_state_reproduces_sequence_outputs() {
        let mut vad = VadStateful::new(16000).unwrap();
        let frame_size = vad.frame_size();
        let mut frames: Vec<Vec<f32>> = Vec::new();
        for idx in 0..6 {
            let mut frame = make_audio(frame_size);
            for sample in &mut frame {
                *sample += idx as f32 * 0.01;
            }
            frames.push(frame);
        }

        let first_pass: Vec<bool> = frames
            .iter()
            .map(|f| vad.detect_frame(f).expect("valid frame"))
            .collect();

        vad.reset_state();

        let second_pass: Vec<bool> = frames
            .iter()
            .map(|f| vad.detect_frame(f).expect("valid frame"))
            .collect();

        assert_eq!(first_pass, second_pass);
    }

    #[test]
    fn stateful_with_config_matches_with_mode() {
        let mut mode_vad = VadStateful::with_mode(16000, VADModes::Normal).unwrap();
        let frame_size = mode_vad.frame_size();
        let cfg = mode_as_vad_config(VADModes::Normal, 16000, frame_size);
        let mut cfg_vad = VadStateful::with_config(16000, cfg).unwrap();

        for idx in 0..10 {
            let mut frame = make_audio(frame_size);
            for sample in &mut frame {
                *sample += idx as f32 * 0.005;
            }

            let mode_out = mode_vad.detect_frame(&frame).expect("valid frame");
            let cfg_out = cfg_vad.detect_frame(&frame).expect("valid frame");
            assert_eq!(mode_out, cfg_out);
        }
    }

    #[test]
    fn online_smoothing_applies_min_speech_min_silence_and_hangover() {
        let config = DetectionConfig {
            min_speech_frames: 2,
            min_silence_frames: 3,
            hangover_frames: 2,
            logit_threshold: 0.0,
        };

        let raw_labels = vec![
            true, false, true, true, false, false, false, false, true, true, false, false, false,
            false, false,
        ];

        let smoothed = apply_online_smoothing(&raw_labels, config);

        assert_eq!(
            smoothed,
            vec![
                false, false, false, true, true, true, true, true, true, true, true, true, true,
                true, false,
            ]
        );
    }
}
