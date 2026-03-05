"""
Tests for FeatureExtractor Python bindings.

Frame parameters (from Rust constants):
  FRAME_SIZE  = 512 samples
  SAMPLE_RATE = 16000 Hz
  BIN_WIDTH   = 16000 / 512 = 31.25 Hz

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

SAMPLE_RATE = 16000
FRAME_SIZE = 512

# Pick a frequency at the centre of each band's bin range (bin * 31.25 Hz)
# Chosen so the tone maps cleanly into a single band.
#   band 0: bin 4  →  125.0 Hz
#   band 1: bin 9  →  281.25 Hz
#   band 2: bin 15 →  468.75 Hz
#   band 3: bin 25 →  781.25 Hz
#   band 4: bin 41 → 1281.25 Hz
#   band 5: bin 63 → 1968.75 Hz
#   band 6: bin 92 → 2875.0 Hz
#   band 7: bin 118→ 3687.5 Hz
BAND_TONE_HZ = [125.0, 281.25, 468.75, 781.25, 1281.25, 1968.75, 2875.0, 3687.5]


def make_sine(freq_hz: float, num_frames: int = 1, amplitude: float = 0.5) -> np.ndarray:
    """Return a float32 array of `num_frames` complete 512-sample frames of a sine tone."""
    n = FRAME_SIZE * num_frames
    t = np.arange(n, dtype=np.float32)
    return (amplitude * np.sin(2.0 * math.pi * freq_hz * t / SAMPLE_RATE)).astype(np.float32)


@pytest.fixture(scope="module")
def extractor():
    return fast_vad.FeatureExtractor()


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_output_shape_single_frame(extractor):
    audio = make_sine(1000.0, num_frames=1)
    out = extractor.extract_features(audio, SAMPLE_RATE)
    assert out.shape == (1, 8), f"Expected (1, 8), got {out.shape}"


def test_output_shape_multiple_frames(extractor):
    audio = make_sine(1000.0, num_frames=10)
    out = extractor.extract_features(audio, SAMPLE_RATE)
    assert out.shape == (10, 8)


def test_trailing_samples_discarded(extractor):
    """Partial last frame must be silently dropped."""
    audio = make_sine(1000.0, num_frames=5)
    audio_extra = np.concatenate([audio, np.zeros(100, dtype=np.float32)])
    out_exact = extractor.extract_features(audio, SAMPLE_RATE)
    out_extra = extractor.extract_features(audio_extra, SAMPLE_RATE)
    assert out_exact.shape == out_extra.shape
    np.testing.assert_array_equal(out_exact, out_extra)


def test_empty_input_returns_empty(extractor):
    audio = np.array([], dtype=np.float32)
    out = extractor.extract_features(audio, SAMPLE_RATE)
    assert out.shape == (0, 8)


def test_wrong_sample_rate_raises(extractor):
    audio = make_sine(1000.0)
    with pytest.raises(ValueError, match="16 kHz"):
        extractor.extract_features(audio, 8000)


def test_output_dtype(extractor):
    audio = make_sine(1000.0)
    out = extractor.extract_features(audio, SAMPLE_RATE)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Silence / floor
# ---------------------------------------------------------------------------

def test_silence_output_near_zero(extractor):
    """Silence should produce the log-energy floor: ln(0 + 1e-10) ≈ -23.025."""
    audio = np.zeros(FRAME_SIZE * 4, dtype=np.float32)
    out = extractor.extract_features(audio, SAMPLE_RATE)
    floor = math.log(1e-10)
    assert np.allclose(out, floor, atol=0.01), (
        f"Silence should give log-energy floor ≈ {floor:.3f}, got range "
        f"[{out.min():.3f}, {out.max():.3f}]"
    )


# ---------------------------------------------------------------------------
# Tone-in-band correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band,freq_hz", enumerate(BAND_TONE_HZ))
def test_tone_dominates_correct_band(extractor, band, freq_hz):
    """A pure sine at a frequency inside a band should produce the highest energy in that band."""
    audio = make_sine(freq_hz, num_frames=4, amplitude=0.9)
    out = extractor.extract_features(audio, SAMPLE_RATE)
    # Average across frames for stability
    mean_energies = out.mean(axis=0)
    peak = int(np.argmax(mean_energies))
    assert peak == band, (
        f"Tone at {freq_hz} Hz should peak in band {band}, "
        f"but peaked in band {peak}. Energies: {mean_energies}"
    )


# ---------------------------------------------------------------------------
# Energy monotonicity
# ---------------------------------------------------------------------------

def test_louder_signal_yields_higher_energy(extractor):
    quiet = make_sine(1000.0, num_frames=4, amplitude=0.1)
    loud  = make_sine(1000.0, num_frames=4, amplitude=0.8)
    out_quiet = extractor.extract_features(quiet, SAMPLE_RATE)
    out_loud  = extractor.extract_features(loud,  SAMPLE_RATE)
    assert out_loud.mean() > out_quiet.mean()


def test_double_amplitude_increases_energy(extractor):
    a1 = make_sine(1000.0, num_frames=4, amplitude=0.2)
    a2 = make_sine(1000.0, num_frames=4, amplitude=0.4)
    out1 = extractor.extract_features(a1, SAMPLE_RATE)
    out2 = extractor.extract_features(a2, SAMPLE_RATE)
    assert out2.mean() > out1.mean()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_output(extractor):
    audio = make_sine(1500.0, num_frames=8)
    out1 = extractor.extract_features(audio, SAMPLE_RATE)
    out2 = extractor.extract_features(audio, SAMPLE_RATE)
    np.testing.assert_array_equal(out1, out2)


def test_independent_instances_agree():
    audio = make_sine(2000.0, num_frames=6)
    e1 = fast_vad.FeatureExtractor()
    e2 = fast_vad.FeatureExtractor()
    np.testing.assert_array_equal(
        e1.extract_features(audio, SAMPLE_RATE),
        e2.extract_features(audio, SAMPLE_RATE),
    )


# ---------------------------------------------------------------------------
# Energies are non-negative
# ---------------------------------------------------------------------------

def test_all_energies_above_floor(extractor):
    """Log-energies must be ≥ ln(1e-10), the noise floor applied in simd.rs."""
    floor = math.log(1e-10)
    rng = np.random.default_rng(42)
    audio = rng.uniform(-1.0, 1.0, FRAME_SIZE * 16).astype(np.float32)
    out = extractor.extract_features(audio, SAMPLE_RATE)
    assert np.all(out >= floor - 0.01), f"Found energies below floor {floor:.3f}: {out[out < floor - 0.01]}"
