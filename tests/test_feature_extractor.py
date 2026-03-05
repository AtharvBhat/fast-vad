"""
Tests for FeatureExtractor Python bindings.

Frame parameters (from Rust constants):
  FRAME_SIZE_16K = 512 samples  (16 kHz)
  FRAME_SIZE_8K  = 256 samples  (8 kHz)
  BIN_WIDTH      = 31.25 Hz  (same for both, since ratio sample_rate/frame_size is identical)

Band → bin ranges (half-open) and approximate Hz:
  0: [3,  6)  →   94 –  188 Hz
  1: [6, 12)  →  188 –  375 Hz
  2: [12, 19) →  375 –  594 Hz
  3: [19, 32) →  594 – 1000 Hz
  4: [32, 51) → 1000 – 1594 Hz
  5: [51, 77) → 1594 – 2406 Hz
  6: [77,108) → 2406 – 3375 Hz
  7: [108,128)→ 3375 – 4000 Hz
"""

import math
import pytest
import numpy as np
import fast_vad

SAMPLE_RATE_16K = 16000
FRAME_SIZE_16K  = 512

SAMPLE_RATE_8K  = 8000
FRAME_SIZE_8K   = 256

# Bin width is identical for both rates (31.25 Hz), so the same centre
# frequencies apply for tone-in-band tests.
BAND_TONE_HZ = [125.0, 281.25, 468.75, 781.25, 1281.25, 1968.75, 2875.0, 3687.5]


def make_sine(freq_hz: float, sample_rate: int, frame_size: int,
              num_frames: int = 1, amplitude: float = 0.5) -> np.ndarray:
    """Return a float32 array of `num_frames` complete frames of a sine tone."""
    n = frame_size * num_frames
    t = np.arange(n, dtype=np.float32)
    return (amplitude * np.sin(2.0 * math.pi * freq_hz * t / sample_rate)).astype(np.float32)


@pytest.fixture(scope="module")
def extractor_16k():
    return fast_vad.FeatureExtractor(SAMPLE_RATE_16K)


@pytest.fixture(scope="module")
def extractor_8k():
    return fast_vad.FeatureExtractor(SAMPLE_RATE_8K)


# ---------------------------------------------------------------------------
# Basic contract — 16 kHz
# ---------------------------------------------------------------------------

def test_output_shape_single_frame(extractor_16k):
    audio = make_sine(1000.0, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=1)
    out = extractor_16k.extract_features(audio)
    assert out.shape == (1, 8), f"Expected (1, 8), got {out.shape}"


def test_output_shape_multiple_frames(extractor_16k):
    audio = make_sine(1000.0, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=10)
    out = extractor_16k.extract_features(audio)
    assert out.shape == (10, 8)


def test_trailing_samples_discarded(extractor_16k):
    """Partial last frame must be silently dropped."""
    audio = make_sine(1000.0, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=5)
    audio_extra = np.concatenate([audio, np.zeros(100, dtype=np.float32)])
    out_exact = extractor_16k.extract_features(audio)
    out_extra = extractor_16k.extract_features(audio_extra)
    assert out_exact.shape == out_extra.shape
    np.testing.assert_array_equal(out_exact, out_extra)


def test_empty_input_returns_empty(extractor_16k):
    audio = np.array([], dtype=np.float32)
    out = extractor_16k.extract_features(audio)
    assert out.shape == (0, 8)


def test_wrong_sample_rate_raises():
    with pytest.raises(ValueError):
        fast_vad.FeatureExtractor(44100)


def test_output_dtype(extractor_16k):
    audio = make_sine(1000.0, SAMPLE_RATE_16K, FRAME_SIZE_16K)
    out = extractor_16k.extract_features(audio)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Basic contract — 8 kHz
# ---------------------------------------------------------------------------

def test_8k_output_shape_single_frame(extractor_8k):
    audio = make_sine(1000.0, SAMPLE_RATE_8K, FRAME_SIZE_8K, num_frames=1)
    out = extractor_8k.extract_features(audio)
    assert out.shape == (1, 8)


def test_8k_output_shape_multiple_frames(extractor_8k):
    audio = make_sine(1000.0, SAMPLE_RATE_8K, FRAME_SIZE_8K, num_frames=10)
    out = extractor_8k.extract_features(audio)
    assert out.shape == (10, 8)


def test_8k_trailing_samples_discarded(extractor_8k):
    audio = make_sine(1000.0, SAMPLE_RATE_8K, FRAME_SIZE_8K, num_frames=5)
    audio_extra = np.concatenate([audio, np.zeros(50, dtype=np.float32)])
    out_exact = extractor_8k.extract_features(audio)
    out_extra = extractor_8k.extract_features(audio_extra)
    assert out_exact.shape == out_extra.shape
    np.testing.assert_array_equal(out_exact, out_extra)


def test_8k_empty_input_returns_empty(extractor_8k):
    audio = np.array([], dtype=np.float32)
    out = extractor_8k.extract_features(audio)
    assert out.shape == (0, 8)


def test_8k_output_dtype(extractor_8k):
    audio = make_sine(1000.0, SAMPLE_RATE_8K, FRAME_SIZE_8K)
    out = extractor_8k.extract_features(audio)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Silence / floor
# ---------------------------------------------------------------------------

def test_silence_output_near_floor(extractor_16k):
    """Silence should produce the log-energy floor: ln(0 + 1e-10) ≈ -23.025."""
    audio = np.zeros(FRAME_SIZE_16K * 4, dtype=np.float32)
    out = extractor_16k.extract_features(audio)
    floor = math.log(1e-10)
    assert np.allclose(out, floor, atol=0.01), (
        f"Silence should give log-energy floor ≈ {floor:.3f}, got range "
        f"[{out.min():.3f}, {out.max():.3f}]"
    )


def test_8k_silence_output_near_floor(extractor_8k):
    audio = np.zeros(FRAME_SIZE_8K * 4, dtype=np.float32)
    out = extractor_8k.extract_features(audio)
    floor = math.log(1e-10)
    assert np.allclose(out, floor, atol=0.01)


# ---------------------------------------------------------------------------
# Tone-in-band correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band,freq_hz", enumerate(BAND_TONE_HZ))
def test_tone_dominates_correct_band(extractor_16k, band, freq_hz):
    """A pure sine at a frequency inside a band should produce the highest energy in that band."""
    audio = make_sine(freq_hz, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=4, amplitude=0.9)
    out = extractor_16k.extract_features(audio)
    mean_energies = out.mean(axis=0)
    peak = int(np.argmax(mean_energies))
    assert peak == band, (
        f"Tone at {freq_hz} Hz should peak in band {band}, "
        f"but peaked in band {peak}. Energies: {mean_energies}"
    )


@pytest.mark.parametrize("band,freq_hz", enumerate(BAND_TONE_HZ))
def test_8k_tone_dominates_correct_band(extractor_8k, band, freq_hz):
    """Same bin layout applies at 8 kHz (bin width is identical)."""
    audio = make_sine(freq_hz, SAMPLE_RATE_8K, FRAME_SIZE_8K, num_frames=4, amplitude=0.9)
    out = extractor_8k.extract_features(audio)
    mean_energies = out.mean(axis=0)
    peak = int(np.argmax(mean_energies))
    assert peak == band, (
        f"8kHz tone at {freq_hz} Hz should peak in band {band}, "
        f"but peaked in band {peak}. Energies: {mean_energies}"
    )


# ---------------------------------------------------------------------------
# Energy monotonicity
# ---------------------------------------------------------------------------

def test_louder_signal_yields_higher_energy(extractor_16k):
    quiet = make_sine(1000.0, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=4, amplitude=0.1)
    loud  = make_sine(1000.0, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=4, amplitude=0.8)
    out_quiet = extractor_16k.extract_features(quiet)
    out_loud  = extractor_16k.extract_features(loud)
    assert out_loud.mean() > out_quiet.mean()


def test_double_amplitude_increases_energy(extractor_16k):
    a1 = make_sine(1000.0, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=4, amplitude=0.2)
    a2 = make_sine(1000.0, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=4, amplitude=0.4)
    out1 = extractor_16k.extract_features(a1)
    out2 = extractor_16k.extract_features(a2)
    assert out2.mean() > out1.mean()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_output(extractor_16k):
    audio = make_sine(1500.0, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=8)
    out1 = extractor_16k.extract_features(audio)
    out2 = extractor_16k.extract_features(audio)
    np.testing.assert_array_equal(out1, out2)


def test_8k_deterministic_output(extractor_8k):
    audio = make_sine(1500.0, SAMPLE_RATE_8K, FRAME_SIZE_8K, num_frames=8)
    out1 = extractor_8k.extract_features(audio)
    out2 = extractor_8k.extract_features(audio)
    np.testing.assert_array_equal(out1, out2)


def test_independent_instances_agree():
    audio = make_sine(2000.0, SAMPLE_RATE_16K, FRAME_SIZE_16K, num_frames=6)
    e1 = fast_vad.FeatureExtractor(SAMPLE_RATE_16K)
    e2 = fast_vad.FeatureExtractor(SAMPLE_RATE_16K)
    np.testing.assert_array_equal(
        e1.extract_features(audio),
        e2.extract_features(audio),
    )


# ---------------------------------------------------------------------------
# Log-energies are bounded below by the noise floor
# ---------------------------------------------------------------------------

def test_all_energies_above_floor(extractor_16k):
    """Log-energies must be ≥ ln(1e-10), the noise floor applied in simd.rs."""
    floor = math.log(1e-10)
    rng = np.random.default_rng(42)
    audio = rng.uniform(-1.0, 1.0, FRAME_SIZE_16K * 16).astype(np.float32)
    out = extractor_16k.extract_features(audio)
    assert np.all(out >= floor - 0.01), f"Found energies below floor {floor:.3f}: {out[out < floor - 0.01]}"


def test_8k_all_energies_above_floor(extractor_8k):
    floor = math.log(1e-10)
    rng = np.random.default_rng(42)
    audio = rng.uniform(-1.0, 1.0, FRAME_SIZE_8K * 16).astype(np.float32)
    out = extractor_8k.extract_features(audio)
    assert np.all(out >= floor - 0.01), f"Found energies below floor {floor:.3f}: {out[out < floor - 0.01]}"
