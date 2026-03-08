//! Core Rust implementation for feature extraction and voice activity detection.

/// Internal constants for the filterbank and detector model.
#[doc(hidden)]
pub mod constants;
/// Stateless and stateful voice activity detector implementations.
pub mod detector;
/// Log-energy filterbank feature extraction.
pub mod filterbank;
/// Internal SIMD helpers used by the filterbank implementation.
#[doc(hidden)]
pub mod simd;

/// Errors returned by fast-vad.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VadError {
    /// Sample rate is not supported. Only 8000 and 16000 Hz are accepted.
    #[error("unsupported sample rate: {0} Hz. Only 8000 and 16000 Hz are supported.")]
    UnsupportedSampleRate(usize),
    /// Frame passed to `detect_frame` has the wrong length.
    #[error("invalid frame length: expected {expected} samples, got {got}")]
    InvalidFrameLength {
        /// Number of samples required for one frame.
        expected: usize,
        /// Number of samples actually provided by the caller.
        got: usize,
    },
}
