#![warn(missing_docs)]
//! Fast voice activity detection for 8 kHz and 16 kHz mono audio.
//!
//! This crate provides three main entry points:
//! - [`VAD`] for batch detection over a full buffer.
//! - [`VadStateful`] for streaming detection one frame at a time.
//! - [`FilterBank`] for direct access to the underlying 8-band log-energy features.
//!
//! The detector operates on fixed 32 ms frames:
//! - 512 samples at 16 kHz
//! - 256 samples at 8 kHz
//!
//! # Example
//!
//! ```rust
//! use fast_vad::{VAD, VADModes};
//!
//! let audio = vec![0.0f32; 16_000];
//! let vad = VAD::with_mode(16_000, VADModes::Normal)?;
//! let sample_labels = vad.detect(&audio);
//! assert_eq!(sample_labels.len(), audio.len());
//! # Ok::<(), fast_vad::VadError>(())
//! ```

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};
use realfft::num_complex::Complex32;

/// Core Rust implementation modules for detection and feature extraction.
pub mod vad;

pub use vad::VadError;
pub use vad::detector::{VAD, VADModes, VadConfig, VadStateful};
pub use vad::filterbank::FilterBank;

fn map_vad_error(err: vad::VadError) -> PyErr {
    match err {
        vad::VadError::UnsupportedSampleRate(rate) => PyErr::new::<PyValueError, _>(format!(
            "Unsupported sample rate: {rate} Hz. Only 8000 and 16000 Hz are supported."
        )),
        vad::VadError::InvalidFrameLength { expected, got } => PyErr::new::<PyValueError, _>(
            format!("Invalid frame length: expected {expected} samples, got {got}"),
        ),
    }
}

fn parse_mode(mode: i32) -> PyResult<vad::detector::VADModes> {
    vad::detector::VADModes::from_index(mode).ok_or_else(|| {
        PyErr::new::<PyValueError, _>(format!(
            "Unsupported mode value: {mode}. Use fast_vad.mode.permissive, fast_vad.mode.normal, or fast_vad.mode.aggressive."
        ))
    })
}

fn parse_vad_config(
    threshold_probability: f32,
    min_speech_ms: usize,
    min_silence_ms: usize,
    hangover_ms: usize,
) -> vad::detector::VadConfig {
    vad::detector::VadConfig {
        threshold_probability,
        min_speech_ms,
        min_silence_ms,
        hangover_ms,
    }
}

fn add_mode_namespace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let mode_module = PyModule::new(m.py(), "mode")?;
    mode_module.add("permissive", vad::detector::VADModes::Permissive.as_index())?;
    mode_module.add("normal", vad::detector::VADModes::Normal.as_index())?;
    mode_module.add("aggressive", vad::detector::VADModes::Aggressive.as_index())?;
    m.add_submodule(&mode_module)?;
    Ok(())
}

fn segments_to_array<'py>(py: Python<'py>, segments: Vec<[usize; 2]>) -> Bound<'py, PyArray2<u64>> {
    let mut arr = Array2::<u64>::zeros((segments.len(), 2));
    for (i, [start, end]) in segments.iter().enumerate() {
        arr[[i, 0]] = *start as u64;
        arr[[i, 1]] = *end as u64;
    }
    PyArray2::from_owned_array(py, arr)
}

/// Computes log-filterbank features from raw audio.
#[pyclass]
struct FeatureExtractor {
    feature_extractor: vad::filterbank::FilterBank,
    window_buf: Vec<f32>,
    fft_output: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
}

#[pymethods]
impl FeatureExtractor {
    /// Creates a `FeatureExtractor` for the given sample rate (8000 or 16000 Hz).
    #[new]
    fn new(sample_rate: usize) -> PyResult<Self> {
        let fe = vad::filterbank::FilterBank::new(sample_rate).map_err(map_vad_error)?;
        let window_buf = vec![0.0f32; fe.frame_size()];
        let fft_output = fe.make_output_vec();
        let fft_scratch = fe.make_scratch_vec();
        Ok(Self {
            feature_extractor: fe,
            window_buf,
            fft_output,
            fft_scratch,
        })
    }

    /// Number of samples per analysis frame.
    #[getter]
    fn frame_size(&self) -> usize {
        self.feature_extractor.frame_size()
    }

    /// Hann window applied to each frame before FFT.
    #[getter]
    fn hann_window<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.feature_extractor.hann_window())
    }

    /// Extracts filterbank features from a single frame of exactly `frame_size` samples.
    ///
    /// Returns a float32 array of shape `(8,)`.
    ///
    /// Raises `ValueError` if `len(frame) != frame_size`.
    fn extract_features_frame<'py>(
        &mut self,
        py: Python<'py>,
        frame: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let frame = frame.as_slice()?;
        let energies = py
            .detach(|| {
                self.feature_extractor.process_single_frame(
                    frame,
                    &mut self.window_buf,
                    &mut self.fft_output,
                    &mut self.fft_scratch,
                )
            })
            .map_err(map_vad_error)?;
        Ok(PyArray1::from_slice(py, &energies.to_array()))
    }

    /// Extracts filterbank features from `audio`.
    ///
    /// Returns a `(num_frames, 8)` float32 array. Trailing samples that do not
    /// fill a complete frame are discarded.
    fn extract_features<'py>(
        &self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let audio = audio.as_slice()?;
        let features = py.detach(|| self.feature_extractor.compute_filterbank(audio));
        let num_frames = features.len();

        let mut arr = Array2::<f32>::zeros((num_frames, vad::constants::NUM_BANDS));
        for (i, frame) in features.iter().enumerate() {
            arr.row_mut(i).assign(&ndarray::ArrayView1::from(
                &frame.to_array() as &[f32; vad::constants::NUM_BANDS]
            ));
        }
        Ok(PyArray2::from_owned_array(py, arr))
    }

    fn __repr__(&self) -> String {
        format!("FeatureExtractor(frame_size={})", self.frame_size())
    }
}

/// Batch voice activity detector.
///
/// Config is fixed at construction time.
/// Use [`VAD.with_mode`] or [`VAD.with_config`] to control detection behaviour.
#[pyclass(name = "VAD")]
struct PyVAD {
    vad: vad::detector::VAD,
}

#[pymethods]
impl PyVAD {
    /// Creates a `VAD` with the default Normal mode.
    ///
    /// Args:
    ///     sample_rate: Audio sample rate in Hz. Supported values: 8000, 16000.
    #[new]
    fn new(sample_rate: usize) -> PyResult<Self> {
        Ok(Self {
            vad: vad::detector::VAD::new(sample_rate).map_err(map_vad_error)?,
        })
    }

    /// Creates a `VAD` with an explicit detection mode.
    #[classmethod]
    fn with_mode(_cls: &Bound<'_, PyType>, sample_rate: usize, mode: i32) -> PyResult<Self> {
        let mode = parse_mode(mode)?;
        Ok(Self {
            vad: vad::detector::VAD::with_mode(sample_rate, mode).map_err(map_vad_error)?,
        })
    }

    /// Creates a `VAD` with custom detection parameters.
    #[classmethod]
    fn with_config(
        _cls: &Bound<'_, PyType>,
        sample_rate: usize,
        threshold_probability: f32,
        min_speech_ms: usize,
        min_silence_ms: usize,
        hangover_ms: usize,
    ) -> PyResult<Self> {
        let config = parse_vad_config(
            threshold_probability,
            min_speech_ms,
            min_silence_ms,
            hangover_ms,
        );
        Ok(Self {
            vad: vad::detector::VAD::with_config(sample_rate, config).map_err(map_vad_error)?,
        })
    }

    /// Returns one `bool` per sample indicating speech presence.
    fn detect<'py>(
        &self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Vec<bool>> {
        let audio = audio.as_slice()?;
        Ok(py.detach(|| self.vad.detect(audio)))
    }

    /// Returns one `bool` per frame indicating speech presence.
    fn detect_frames<'py>(
        &self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Vec<bool>> {
        let audio = audio.as_slice()?;
        Ok(py.detach(|| self.vad.detect_frames(audio)))
    }

    /// Returns a `(N, 2)` uint64 array of `[start, end]` sample indices for each speech segment.
    fn detect_segments<'py>(
        &self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        let audio = audio.as_slice()?;
        let segments = py.detach(|| self.vad.detect_segments(audio));
        Ok(segments_to_array(py, segments))
    }

    fn __repr__(&self) -> String {
        let sample_rate = match self.vad.frame_size() {
            512 => 16000,
            256 => 8000,
            _ => 0, // Should never happen
        };
        format!("VAD(sample_rate={})", sample_rate)
    }
}

/// Streaming voice activity detector that processes one frame at a time.
///
/// Config is fixed at construction time.
/// Use [`VadStateful.with_mode`] or [`VadStateful.with_config`] to control detection behaviour.
#[pyclass(name = "VadStateful")]
struct PyVadStateful {
    vad: Box<vad::detector::VadStateful>,
}

#[pymethods]
impl PyVadStateful {
    /// Creates a `VadStateful` with the default Normal mode.
    ///
    /// Args:
    ///     sample_rate: Audio sample rate in Hz. Supported values: 8000, 16000.
    #[new]
    fn new(sample_rate: usize) -> PyResult<Self> {
        Ok(Self {
            vad: Box::new(vad::detector::VadStateful::new(sample_rate).map_err(map_vad_error)?),
        })
    }

    /// Creates a `VadStateful` with an explicit detection mode.
    #[classmethod]
    fn with_mode(_cls: &Bound<'_, PyType>, sample_rate: usize, mode: i32) -> PyResult<Self> {
        let mode = parse_mode(mode)?;
        Ok(Self {
            vad: Box::new(
                vad::detector::VadStateful::with_mode(sample_rate, mode).map_err(map_vad_error)?,
            ),
        })
    }

    /// Creates a `VadStateful` with custom detection parameters.
    #[classmethod]
    fn with_config(
        _cls: &Bound<'_, PyType>,
        sample_rate: usize,
        threshold_probability: f32,
        min_speech_ms: usize,
        min_silence_ms: usize,
        hangover_ms: usize,
    ) -> PyResult<Self> {
        let config = parse_vad_config(
            threshold_probability,
            min_speech_ms,
            min_silence_ms,
            hangover_ms,
        );
        Ok(Self {
            vad: Box::new(
                vad::detector::VadStateful::with_config(sample_rate, config)
                    .map_err(map_vad_error)?,
            ),
        })
    }

    /// Number of samples per frame expected by `detect_frame`.
    #[getter]
    fn frame_size(&self) -> usize {
        self.vad.frame_size()
    }

    /// Processes one frame and returns whether speech is active.
    ///
    /// `frame` must contain exactly `frame_size` samples.
    fn detect_frame<'py>(
        &mut self,
        py: Python<'py>,
        frame: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<bool> {
        let frame = frame.as_slice()?;
        py.detach(|| self.vad.detect_frame(frame))
            .map_err(map_vad_error)
    }

    /// Resets internal state so the detector can be reused for a new stream.
    fn reset_state(&mut self) {
        self.vad.reset_state();
    }

    fn __repr__(&self) -> String {
        let sample_rate = match self.frame_size() {
            512 => 16000,
            256 => 8000,
            _ => 0, // Should never happen
        };
        format!("VadStateful(sample_rate={})", sample_rate)
    }
}

#[pymodule]
fn fast_vad(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<FeatureExtractor>()?;
    m.add_class::<PyVAD>()?;
    m.add_class::<PyVadStateful>()?;
    add_mode_namespace(m)?;
    Ok(())
}
