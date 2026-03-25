import numpy as np
from numpy.typing import NDArray

__version__: str

class FeatureExtractor:
    """Computes log-filterbank features from raw audio."""

    def __init__(self, sample_rate: int) -> None:
        """
        Args:
            sample_rate: Audio sample rate in Hz. Supported values: 8000, 16000.

        Raises:
            ValueError: If ``sample_rate`` is not supported.
        """
        ...

    @property
    def frame_size(self) -> int:
        """Number of samples per analysis frame."""
        ...

    @property
    def hann_window(self) -> NDArray[np.float32]:
        """Hann window applied to each frame before FFT. Shape: (frame_size,)."""
        ...

    def extract_features_frame(self, frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Extract filterbank features from a single frame.

        Args:
            frame: 1-D float32 array of exactly ``frame_size`` samples.

        Returns:
            Float32 array of shape (8,).

        Raises:
            ValueError: If ``len(frame) != frame_size``.
        """
        ...

    def extract_features(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Extract filterbank features from audio.

        Args:
            audio: 1-D float32 array of audio samples.

        Returns:
            Float32 array of shape (num_frames, 8). Trailing samples that do
            not fill a complete frame are discarded.
        """
        ...

    def feature_engineer(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute 24-dimensional features for each frame in audio.

        Each row contains 8 log-energy values, 8 first-order deltas, and
        8 second-order deltas.

        Args:
            audio: 1-D float32 array of audio samples.

        Returns:
            Float32 array of shape (num_frames, 24).
        """
        ...

class mode:
    """Integer constants for VAD operating modes."""

    permissive: int
    """Low false-negative rate; more speech accepted."""
    normal: int
    """Balanced mode for general use."""
    aggressive: int
    """Low false-positive rate; stricter speech detection."""

VADMode = int
"""Integer alias for a VAD mode. Use constants from :class:`mode`."""

class VAD:
    """Batch voice activity detector.

    Config is fixed at construction. Use :meth:`with_mode` or :meth:`with_config`
    to select detection behaviour; the plain constructor defaults to Normal mode.
    """

    def __init__(self, sample_rate: int) -> None:
        """
        Create a VAD with the default Normal mode.

        Args:
            sample_rate: Audio sample rate in Hz. Supported values: 8000, 16000.

        Raises:
            ValueError: If ``sample_rate`` is not supported.
        """
        ...

    @classmethod
    def with_mode(cls, sample_rate: int, mode: VADMode) -> "VAD":
        """
        Create a VAD with an explicit detection mode.

        Args:
            sample_rate: Audio sample rate in Hz. Supported values: 8000, 16000.
            mode: Detection mode from :class:`mode`.

        Raises:
            ValueError: If ``sample_rate`` or ``mode`` is not supported.
        """
        ...

    @classmethod
    def with_config(
        cls,
        sample_rate: int,
        threshold_probability: float,
        min_speech_ms: int,
        min_silence_ms: int,
        hangover_ms: int,
    ) -> "VAD":
        """
        Create a VAD with custom detection parameters.

        Args:
            sample_rate: Audio sample rate in Hz. Supported values: 8000, 16000.
            threshold_probability: Speech probability threshold in (0, 1).
            min_speech_ms: Minimum speech run to confirm onset (ms).
            min_silence_ms: Minimum silence run to confirm offset (ms).
            hangover_ms: Extra speech extension after voiced region ends (ms).

        Raises:
            ValueError: If ``sample_rate`` is not supported.
        """
        ...

    def detect(self, audio: NDArray[np.float32]) -> NDArray[np.bool_]:
        """
        Return one bool per sample indicating speech presence.

        Args:
            audio: 1-D float32 array of audio samples.

        Returns:
            Array of booleans, one per sample.
        """
        ...

    def detect_frames(self, audio: NDArray[np.float32]) -> NDArray[np.bool_]:
        """
        Return one bool per frame indicating speech presence.

        Args:
            audio: 1-D float32 array of audio samples.

        Returns:
            List of booleans, one per frame. Trailing samples are discarded.
        """
        ...

    def detect_segments(self, audio: NDArray[np.float32]) -> NDArray[np.uint64]:
        """
        Return speech segments as a (N, 2) uint64 array of [start, end] sample indices.

        Args:
            audio: 1-D float32 array of audio samples.

        Returns:
            Array of shape (N, 2). Each row is [start_sample, end_sample].
        """
        ...

class VadStateful:
    """Streaming VAD that processes one frame at a time.

    Config is fixed at construction. Use :meth:`with_mode` or :meth:`with_config`
    to select detection behaviour; the plain constructor defaults to Normal mode.
    """

    def __init__(self, sample_rate: int) -> None:
        """
        Create a VadStateful with the default Normal mode.

        Args:
            sample_rate: Audio sample rate in Hz. Supported values: 8000, 16000.

        Raises:
            ValueError: If ``sample_rate`` is not supported.
        """
        ...

    @classmethod
    def with_mode(cls, sample_rate: int, mode: VADMode) -> "VadStateful":
        """
        Create a VadStateful with an explicit detection mode.

        Args:
            sample_rate: Audio sample rate in Hz. Supported values: 8000, 16000.
            mode: Detection mode from :class:`mode`.

        Raises:
            ValueError: If ``sample_rate`` or ``mode`` is not supported.
        """
        ...

    @classmethod
    def with_config(
        cls,
        sample_rate: int,
        threshold_probability: float,
        min_speech_ms: int,
        min_silence_ms: int,
        hangover_ms: int,
    ) -> "VadStateful":
        """
        Create a VadStateful with custom detection parameters.

        Args:
            sample_rate: Audio sample rate in Hz. Supported values: 8000, 16000.
            threshold_probability: Speech probability threshold in (0, 1).
            min_speech_ms: Minimum speech run to confirm onset (ms).
            min_silence_ms: Minimum silence run to confirm offset (ms).
            hangover_ms: Extra speech extension after voiced region ends (ms).

        Raises:
            ValueError: If ``sample_rate`` is not supported.
        """
        ...

    @property
    def frame_size(self) -> int:
        """Number of samples expected by :meth:`detect_frame`."""
        ...

    def detect_frame(self, frame: NDArray[np.float32]) -> bool:
        """
        Process one frame and return whether speech is active.

        Args:
            frame: 1-D float32 array of exactly ``frame_size`` samples.

        Returns:
            True if speech is active after this frame.

        Raises:
            ValueError: If ``len(frame) != frame_size``.
        """
        ...

    def reset_state(self) -> None:
        """Reset internal state so the detector can be reused for a new stream."""
        ...
