use crate::vad::simd;
use rayon::prelude::*;
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;
use std::sync::Arc;
use wide::f32x8;

/// Computes log-filterbank energies from raw audio frames.
pub struct FilterBank {
    fft_forward: Arc<dyn realfft::RealToComplex<f32> + Send + Sync>,
    hann_window: Vec<f32>,
    frame_size: usize,
}

impl Default for FilterBank {
    fn default() -> Self {
        Self::new(16000).expect("16000 Hz is a supported sample rate")
    }
}

impl FilterBank {
    /// Creates a `FilterBank` for the given sample rate.
    ///
    /// Supported sample rates: 8000, 16000 Hz.
    pub fn new(sample_rate: usize) -> Result<Self, super::VadError> {
        match sample_rate {
            16000 => {
                let frame_size = 512;
                let mut fft_planner = RealFftPlanner::new();
                Ok(Self {
                    fft_forward: fft_planner.plan_fft_forward(frame_size),
                    hann_window: compute_hann_window(frame_size),
                    frame_size,
                })
            }
            8000 => {
                let frame_size = 256;
                let mut fft_planner = RealFftPlanner::new();
                Ok(Self {
                    fft_forward: fft_planner.plan_fft_forward(frame_size),
                    hann_window: compute_hann_window(frame_size),
                    frame_size,
                })
            }
            _ => Err(super::VadError::UnsupportedSampleRate(sample_rate)),
        }
    }

    /// Number of samples per analysis frame.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// The Hann window applied to each frame before FFT.
    pub fn hann_window(&self) -> &[f32] {
        &self.hann_window
    }

    /// Allocates a fresh FFT output buffer sized for this filterbank.
    pub fn make_output_vec(&self) -> Vec<Complex32> {
        self.fft_forward.make_output_vec()
    }

    /// Allocates a fresh FFT scratch buffer sized for this filterbank.
    pub fn make_scratch_vec(&self) -> Vec<Complex32> {
        self.fft_forward.make_scratch_vec()
    }

    /// Processes a single frame using caller-supplied scratch buffers.
    ///
    /// Avoids any thread-pool overhead — suitable for streaming use.
    /// `window_buf` must have `frame_size` elements; `fft_output` and
    /// `fft_scratch` must come from [`make_output_vec`](Self::make_output_vec)
    /// and [`make_scratch_vec`](Self::make_scratch_vec).
    pub fn process_single_frame(
        &self,
        frame: &[f32],
        window_buf: &mut [f32],
        fft_output: &mut [Complex32],
        fft_scratch: &mut [Complex32],
    ) -> f32x8 {
        window_buf.copy_from_slice(frame);
        simd::apply_hanning_window_simd(window_buf, &self.hann_window);
        self.fft_forward
            .process_with_scratch(window_buf, fft_output, fft_scratch)
            .expect("FFT processing failed");
        simd::compute_band_energies_simd(fft_output)
    }

    /// Computes log-filterbank energies for each complete frame in `input`.
    ///
    /// Drops any trailing samples that do not fill a complete frame.
    pub fn compute_filterbank(&self, input: &[f32]) -> Vec<f32x8> {
        input
            .par_chunks_exact(self.frame_size)
            .map_init(
                || {
                    (
                        vec![0.0f32; self.frame_size],
                        self.fft_forward.make_output_vec(),
                        self.fft_forward.make_scratch_vec(),
                    )
                },
                |(window, output, scratch), frame| {
                    window.copy_from_slice(frame);
                    simd::apply_hanning_window_simd(window, &self.hann_window);
                    self.fft_forward
                        .process_with_scratch(window, output, scratch)
                        .expect("FFT processing failed");
                    simd::compute_band_energies_simd(output)
                },
            )
            .collect()
    }
}

fn compute_hann_window(size: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; size];
    for (n, sample) in window.iter_mut().enumerate() {
        *sample = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / size as f32).cos());
    }
    window
}

#[cfg(test)]
mod tests {
    use crate::vad::constants;
    use crate::vad::filterbank::FilterBank;
    use std::f32::consts::PI;
    use wide::f32x8;

    const SAMPLE_RATE: usize = 16000;
    const FRAME_SIZE: usize = 512;

    fn sine_frame(freq_hz: f32, amplitude: f32) -> Vec<f32> {
        (0..FRAME_SIZE)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f32 / SAMPLE_RATE as f32).sin())
            .collect()
    }

    fn silence_frame() -> Vec<f32> {
        vec![0.0f32; FRAME_SIZE]
    }

    fn multi_tone_frame(tones: &[(f32, f32)]) -> Vec<f32> {
        let mut frame = vec![0.0f32; FRAME_SIZE];
        for &(freq, amp) in tones {
            for (i, sample) in frame.iter_mut().enumerate() {
                *sample += amp * (2.0 * PI * freq * i as f32 / SAMPLE_RATE as f32).sin();
            }
        }
        frame
    }

    fn white_noise_frame(amplitude: f32, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..FRAME_SIZE)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((state >> 33) as f32) / (u32::MAX as f32);
                amplitude * (2.0 * u - 1.0)
            })
            .collect()
    }

    fn energies_to_array(v: f32x8) -> [f32; 8] {
        v.to_array()
    }

    fn peak_band(energies: &[f32; 8]) -> usize {
        energies
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    #[test]
    fn silence_all_bands_very_low() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let result = fb.compute_filterbank(&silence_frame());
        let energies = energies_to_array(result[0]);

        for (band, &energy) in energies.iter().enumerate() {
            assert!(
                energy < -20.0,
                "band {band} energy {:.2} too high for silence",
                energy
            );
        }
    }

    #[test]
    fn tone_dominates_other_bands() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let test_cases: &[(f32, usize)] = &[
            (150.0, 0),
            (300.0, 1),
            (500.0, 2),
            (800.0, 3),
            (1300.0, 4),
            (2000.0, 5),
            (2800.0, 6),
            (3600.0, 7),
        ];

        for &(freq, expected_band) in test_cases {
            let result = fb.compute_filterbank(&sine_frame(freq, 0.5));
            let energies = energies_to_array(result[0]);

            for band in 0..constants::NUM_BANDS {
                if band != expected_band {
                    let margin = energies[expected_band] - energies[band];
                    assert!(
                        margin > 2.0,
                        "{freq}Hz: band {expected_band} ({:.2}) should dominate \
                         band {band} ({:.2}) by >2.0 ln-units",
                        energies[expected_band],
                        energies[band]
                    );
                }
            }
        }
    }

    #[test]
    fn louder_signal_higher_energy() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let quiet = fb.compute_filterbank(&sine_frame(500.0, 0.1));
        let loud = fb.compute_filterbank(&sine_frame(500.0, 0.9));

        let e_quiet = energies_to_array(quiet[0]);
        let e_loud = energies_to_array(loud[0]);

        assert!(
            e_loud[2] > e_quiet[2] + 1.0,
            "loud ({:.2}) should exceed quiet ({:.2}) in band 2",
            e_loud[2],
            e_quiet[2]
        );
    }

    #[test]
    fn double_amplitude_approximately_ln4() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let e1 = fb.compute_filterbank(&sine_frame(500.0, 0.25));
        let e2 = fb.compute_filterbank(&sine_frame(500.0, 0.50));

        let a1 = energies_to_array(e1[0]);
        let a2 = energies_to_array(e2[0]);

        let diff = a2[2] - a1[2];
        assert!(
            (diff - 1.386).abs() < 0.5,
            "expected ~1.386 ln-scale increase, got {diff:.3}"
        );
    }

    #[test]
    fn energy_monotone_with_amplitude() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let amplitudes = [0.05f32, 0.1, 0.2, 0.4, 0.8];
        let mut prev = f32::NEG_INFINITY;
        for &amp in &amplitudes {
            let result = fb.compute_filterbank(&sine_frame(500.0, amp));
            let e = energies_to_array(result[0])[2];
            assert!(
                e > prev,
                "energy not monotone: amp={amp}, energy={e:.3}, prev={prev:.3}"
            );
            prev = e;
        }
    }

    #[test]
    fn white_noise_all_bands_active() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let result = fb.compute_filterbank(&white_noise_frame(0.3, 12345));
        let energies = energies_to_array(result[0]);

        for (band, &energy) in energies.iter().enumerate() {
            assert!(
                energy > -20.0,
                "band {band} energy {:.2} too low for white noise",
                energy
            );
        }
    }

    #[test]
    fn white_noise_no_extreme_dominance() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let result = fb.compute_filterbank(&white_noise_frame(0.3, 99999));
        let energies = energies_to_array(result[0]);

        let max = energies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = energies.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            (max - min) < 10.0,
            "white noise band spread too wide: min={min:.2}, max={max:.2}"
        );
    }

    #[test]
    fn white_noise_wider_bands_more_energy() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let result = fb.compute_filterbank(&white_noise_frame(0.3, 54321));
        let energies = energies_to_array(result[0]);

        assert!(
            energies[7] > energies[0],
            "band 7 ({:.2}) should exceed band 0 ({:.2}) for white noise",
            energies[7],
            energies[0]
        );
    }

    #[test]
    fn multi_tone_energy_in_expected_bands() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let frame = multi_tone_frame(&[(150.0, 0.3), (500.0, 0.2), (1300.0, 0.15)]);
        let result = fb.compute_filterbank(&frame);
        let energies = energies_to_array(result[0]);

        assert!(energies[0] > energies[1] + 1.5, "band 0 vs 1");
        assert!(energies[2] > energies[3] + 1.5, "band 2 vs 3");
        assert!(energies[4] > energies[5] + 1.5, "band 4 vs 5");
    }

    #[test]
    fn same_input_same_output() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let frame = sine_frame(440.0, 0.5);

        let r1 = fb.compute_filterbank(&frame);
        let r2 = fb.compute_filterbank(&frame);

        let e1 = energies_to_array(r1[0]);
        let e2 = energies_to_array(r2[0]);

        for band in 0..constants::NUM_BANDS {
            assert_eq!(e1[band], e2[band], "non-deterministic at band {band}");
        }
    }

    #[test]
    fn separate_instances_agree() {
        let fb1 = FilterBank::new(SAMPLE_RATE).unwrap();
        let fb2 = FilterBank::new(SAMPLE_RATE).unwrap();
        let frame = sine_frame(1000.0, 0.5);

        let e1 = energies_to_array(fb1.compute_filterbank(&frame)[0]);
        let e2 = energies_to_array(fb2.compute_filterbank(&frame)[0]);

        for band in 0..constants::NUM_BANDS {
            assert!(
                (e1[band] - e2[band]).abs() < 1e-5,
                "instances disagree at band {band}: {} vs {}",
                e1[band],
                e2[band]
            );
        }
    }

    #[test]
    fn rate_16khz_produces_correct_frame_count() {
        let fb = FilterBank::new(16000).unwrap();
        let audio = vec![0.0f32; 16000];
        let result = fb.compute_filterbank(&audio);
        assert_eq!(result.len(), 31);
    }

    #[test]
    fn rate_8khz_produces_correct_frame_count() {
        let fb = FilterBank::new(8000).unwrap();
        let audio = vec![0.0f32; 8000];
        let result = fb.compute_filterbank(&audio);
        assert_eq!(result.len(), 31);
    }

    #[test]
    fn rate_16khz_output_finite() {
        let fb = FilterBank::new(16000).unwrap();
        let audio: Vec<f32> = (0..512)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f32 / 16000.0).sin())
            .collect();
        let energies = energies_to_array(fb.compute_filterbank(&audio)[0]);
        assert!(energies.iter().all(|e| e.is_finite()));
    }

    #[test]
    fn rate_8khz_output_finite() {
        let fb = FilterBank::new(8000).unwrap();
        let audio: Vec<f32> = (0..256)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f32 / 8000.0).sin())
            .collect();
        let energies = energies_to_array(fb.compute_filterbank(&audio)[0]);
        assert!(energies.iter().all(|e| e.is_finite()));
    }

    #[test]
    fn unsupported_sample_rate_returns_error() {
        assert!(FilterBank::new(44100).is_err());
    }

    #[test]
    fn frame_count_one_second() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let audio = vec![0.0f32; SAMPLE_RATE];
        let result = fb.compute_filterbank(&audio);
        assert_eq!(result.len(), SAMPLE_RATE / FRAME_SIZE);
    }

    #[test]
    fn frame_count_empty() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let result = fb.compute_filterbank(&[]);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn frame_count_less_than_one_frame() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let audio = vec![0.0f32; FRAME_SIZE - 1];
        let result = fb.compute_filterbank(&audio);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn frame_count_exact_one_frame() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let audio = vec![0.0f32; FRAME_SIZE];
        let result = fb.compute_filterbank(&audio);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn frame_count_trailing_discarded() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let audio = vec![0.0f32; FRAME_SIZE * 3 + 100];
        let result = fb.compute_filterbank(&audio);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn batch_matches_individual() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();

        let mut audio = Vec::with_capacity(FRAME_SIZE * 5);
        audio.extend(sine_frame(150.0, 0.5));
        audio.extend(sine_frame(800.0, 0.3));
        audio.extend(silence_frame());
        audio.extend(sine_frame(2000.0, 0.4));
        audio.extend(white_noise_frame(0.2, 777));

        let batch = fb.compute_filterbank(&audio);

        let frames: Vec<Vec<f32>> = vec![
            sine_frame(150.0, 0.5),
            sine_frame(800.0, 0.3),
            silence_frame(),
            sine_frame(2000.0, 0.4),
            white_noise_frame(0.2, 777),
        ];

        for (i, frame) in frames.iter().enumerate() {
            let single = fb.compute_filterbank(frame);
            let e_batch = energies_to_array(batch[i]);
            let e_single = energies_to_array(single[0]);

            for band in 0..constants::NUM_BANDS {
                assert!(
                    (e_batch[band] - e_single[band]).abs() < 1e-4,
                    "frame {i} band {band}: batch={:.4} vs single={:.4}",
                    e_batch[band],
                    e_single[band]
                );
            }
        }
    }

    #[test]
    fn tone_at_band_boundary_lands_in_one_band() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let result = fb.compute_filterbank(&sine_frame(200.0, 0.5));
        let energies = energies_to_array(result[0]);

        let p = peak_band(&energies);
        assert!(
            p == 0 || p == 1,
            "200Hz boundary tone should peak in band 0 or 1, got {p}"
        );

        for (band, &energy) in energies.iter().enumerate() {
            assert!(energy.is_finite(), "band {band} is not finite: {}", energy);
        }
    }

    #[test]
    fn no_nan_or_inf_on_silence() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let result = fb.compute_filterbank(&silence_frame());
        let energies = energies_to_array(result[0]);
        for (band, &energy) in energies.iter().enumerate() {
            assert!(energy.is_finite(), "band {band} is not finite");
        }
    }

    #[test]
    fn no_nan_or_inf_on_max_amplitude() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();
        let result = fb.compute_filterbank(&sine_frame(1000.0, 1.0));
        let energies = energies_to_array(result[0]);
        for (band, &energy) in energies.iter().enumerate() {
            assert!(energy.is_finite(), "band {band} is not finite");
        }
    }

    #[test]
    fn energy_phase_invariant() {
        let fb = FilterBank::new(SAMPLE_RATE).unwrap();

        let frame_0: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| 0.5 * (2.0 * PI * 500.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let frame_90: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| 0.5 * (2.0 * PI * 500.0 * i as f32 / SAMPLE_RATE as f32 + PI / 2.0).sin())
            .collect();

        let e0 = energies_to_array(fb.compute_filterbank(&frame_0)[0]);
        let e90 = energies_to_array(fb.compute_filterbank(&frame_90)[0]);

        assert!(
            (e0[2] - e90[2]).abs() < 0.05,
            "band 2: energy should be phase-invariant (0°={:.3}, 90°={:.3})",
            e0[2],
            e90[2]
        );
    }
}
