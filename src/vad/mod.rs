pub mod constants;
pub mod detector;
pub mod filterbank;
pub mod simd;

/// Errors returned by fast-vad.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VadError {
    /// Sample rate is not supported. Only 8000 and 16000 Hz are accepted.
    #[error("unsupported sample rate: {0} Hz. Only 8000 and 16000 Hz are supported.")]
    UnsupportedSampleRate(usize),
    /// Frame passed to `detect_frame` has the wrong length.
    #[error("invalid frame length: expected {expected} samples, got {got}")]
    InvalidFrameLength { expected: usize, got: usize },
}
