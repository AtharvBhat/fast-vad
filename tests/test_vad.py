import numpy as np
import pytest
import tomllib
from pathlib import Path

import fast_vad

SAMPLE_RATE_16K = 16000
FRAME_SIZE_16K = 512
SAMPLE_RATE_8K = 8000
FRAME_SIZE_8K = 256

# Custom params that reproduce the built-in mode configs.
# (threshold_probability, min_speech_ms, min_silence_ms, hangover_ms)
MODE_CUSTOM_PARAMS = {
    fast_vad.mode.permissive: (0.6199999469, 64, 384, 192),
    fast_vad.mode.normal:     (0.7200000789, 64, 256, 64),
    fast_vad.mode.aggressive: (0.7699999635, 128, 192, 64),
}


def make_audio(length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=length).astype(np.float32)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vad_16k():
    return fast_vad.VAD.with_mode(SAMPLE_RATE_16K, fast_vad.mode.normal)


@pytest.fixture(scope="module")
def vad_8k():
    return fast_vad.VAD.with_mode(SAMPLE_RATE_8K, fast_vad.mode.normal)


# ── mode namespace ────────────────────────────────────────────────────────────

def test_mode_namespace_values():
    assert fast_vad.mode.permissive == 0
    assert fast_vad.mode.normal == 1
    assert fast_vad.mode.aggressive == 2


def test_module_version_matches_pyproject():
    project_version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    assert fast_vad.__version__ == project_version


# ── Error handling: bad sample rate ───────────────────────────────────────────

@pytest.mark.parametrize("cls_factory", [
    lambda sr: fast_vad.VAD(sr),
    lambda sr: fast_vad.VAD.with_mode(sr, fast_vad.mode.normal),
    lambda sr: fast_vad.VAD.with_config(sr, 0.7, 64, 256, 64),
    lambda sr: fast_vad.VadStateful(sr),
    lambda sr: fast_vad.VadStateful.with_mode(sr, fast_vad.mode.normal),
    lambda sr: fast_vad.VadStateful.with_config(sr, 0.7, 64, 256, 64),
    lambda sr: fast_vad.FeatureExtractor(sr),
])
def test_unsupported_sample_rate_raises(cls_factory):
    with pytest.raises(ValueError, match="Unsupported sample rate"):
        cls_factory(44100)


# ── Error handling: bad mode ───────────────────────────────────────────────────

def test_vad_with_mode_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unsupported mode"):
        fast_vad.VAD.with_mode(SAMPLE_RATE_16K, 99)


def test_stateful_with_mode_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unsupported mode"):
        fast_vad.VadStateful.with_mode(SAMPLE_RATE_16K, 99)


def test_mode_rejects_string_raises():
    with pytest.raises(TypeError):
        fast_vad.VAD.with_mode(SAMPLE_RATE_16K, "normal")  # type: ignore[arg-type]


# ── Error handling: wrong frame length ────────────────────────────────────────

def test_stateful_detect_frame_wrong_length_raises():
    vad = fast_vad.VadStateful(SAMPLE_RATE_16K)
    with pytest.raises(ValueError, match="Invalid frame length"):
        vad.detect_frame(make_audio(vad.frame_size - 1))


def test_stateful_detect_frame_empty_raises():
    vad = fast_vad.VadStateful(SAMPLE_RATE_16K)
    with pytest.raises(ValueError, match="Invalid frame length"):
        vad.detect_frame(np.array([], dtype=np.float32))


# ── detect output shape and type ──────────────────────────────────────────────

def test_detect_returns_list_of_bools(vad_16k):
    audio = make_audio(FRAME_SIZE_16K * 2)
    result = vad_16k.detect(audio)
    assert isinstance(result, list)
    assert len(result) == len(audio)
    assert all(isinstance(v, bool) for v in result)


def test_detect_frames_returns_list_of_bools(vad_16k):
    audio = make_audio(FRAME_SIZE_16K * 3)
    result = vad_16k.detect_frames(audio)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(v, bool) for v in result)


def test_detect_segments_returns_numpy_array(vad_16k):
    audio = make_audio(FRAME_SIZE_16K * 4, seed=5)
    result = vad_16k.detect_segments(audio)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint64
    assert result.ndim == 2
    assert result.shape[1] == 2


def test_detect_segments_empty_audio_returns_empty_array(vad_16k):
    audio = make_audio(FRAME_SIZE_16K - 1)
    result = vad_16k.detect_segments(audio)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)


# ── detect correctness ────────────────────────────────────────────────────────

def test_detect_sample_labels_carry_tail_from_last_frame(vad_16k):
    audio = make_audio(FRAME_SIZE_16K * 3 + 37, seed=1)
    frame_labels = vad_16k.detect_frames(audio)
    sample_labels = vad_16k.detect(audio)

    assert len(frame_labels) == 3
    assert len(sample_labels) == len(audio)

    for i, label in enumerate(frame_labels):
        start = i * FRAME_SIZE_16K
        end = start + FRAME_SIZE_16K
        assert all(v == label for v in sample_labels[start:end])

    tail_start = len(frame_labels) * FRAME_SIZE_16K
    assert all(v == frame_labels[-1] for v in sample_labels[tail_start:])


def test_detect_short_audio_returns_all_false(vad_16k):
    audio = make_audio(FRAME_SIZE_16K - 1, seed=2)
    assert vad_16k.detect_frames(audio) == []

    sample_labels = vad_16k.detect(audio)
    assert len(sample_labels) == len(audio)
    assert all(v is False for v in sample_labels)

    segments = vad_16k.detect_segments(audio)
    assert segments.shape == (0, 2)


def test_detect_segments_reconstruct_sample_labels(vad_16k):
    audio = make_audio(FRAME_SIZE_16K * 5 + 19, seed=3)
    sample_labels = vad_16k.detect(audio)
    segments = vad_16k.detect_segments(audio)

    reconstructed = [False] * len(audio)
    previous_end = 0
    for start, end in segments:
        start, end = int(start), int(end)
        assert 0 <= start < end <= len(audio)
        assert start >= previous_end
        for idx in range(start, end):
            reconstructed[idx] = True
        previous_end = end

    assert reconstructed == sample_labels


def test_detect_segments_values_are_valid_indices(vad_16k):
    audio = make_audio(FRAME_SIZE_16K * 6, seed=7)
    segments = vad_16k.detect_segments(audio)
    for start, end in segments:
        assert start < end
        assert end <= len(audio)


# ── with_config matches with_mode ─────────────────────────────────────────────

@pytest.mark.parametrize("mode", [
    fast_vad.mode.permissive,
    fast_vad.mode.normal,
    fast_vad.mode.aggressive,
])
def test_with_config_matches_with_mode_16k(mode):
    audio = make_audio(FRAME_SIZE_16K * 8 + 23, seed=40 + mode)
    p, min_speech_ms, min_silence_ms, hangover_ms = MODE_CUSTOM_PARAMS[mode]

    vad_mode = fast_vad.VAD.with_mode(SAMPLE_RATE_16K, mode)
    vad_cfg = fast_vad.VAD.with_config(SAMPLE_RATE_16K, p, min_speech_ms, min_silence_ms, hangover_ms)

    assert vad_mode.detect_frames(audio) == vad_cfg.detect_frames(audio)
    assert vad_mode.detect(audio) == vad_cfg.detect(audio)
    assert np.array_equal(vad_mode.detect_segments(audio), vad_cfg.detect_segments(audio))


@pytest.mark.parametrize("mode", [
    fast_vad.mode.permissive,
    fast_vad.mode.normal,
    fast_vad.mode.aggressive,
])
def test_with_config_matches_with_mode_8k(mode):
    audio = make_audio(FRAME_SIZE_8K * 10 + 17, seed=77 + mode)
    p, min_speech_ms, min_silence_ms, hangover_ms = MODE_CUSTOM_PARAMS[mode]

    vad_mode = fast_vad.VAD.with_mode(SAMPLE_RATE_8K, mode)
    vad_cfg = fast_vad.VAD.with_config(SAMPLE_RATE_8K, p, min_speech_ms, min_silence_ms, hangover_ms)

    assert vad_mode.detect_frames(audio) == vad_cfg.detect_frames(audio)
    assert vad_mode.detect(audio) == vad_cfg.detect(audio)
    assert np.array_equal(vad_mode.detect_segments(audio), vad_cfg.detect_segments(audio))


# ── VadStateful ───────────────────────────────────────────────────────────────

def test_stateful_frame_size_matches_expected():
    assert fast_vad.VadStateful(SAMPLE_RATE_16K).frame_size == FRAME_SIZE_16K
    assert fast_vad.VadStateful(SAMPLE_RATE_8K).frame_size == FRAME_SIZE_8K


def test_stateful_reset_restores_initial_sequence():
    vad = fast_vad.VadStateful.with_mode(SAMPLE_RATE_16K, fast_vad.mode.normal)
    frames = [make_audio(FRAME_SIZE_16K, seed=10 + i) for i in range(6)]

    first_pass = [vad.detect_frame(f) for f in frames]
    vad.reset_state()
    second_pass = [vad.detect_frame(f) for f in frames]

    assert first_pass == second_pass


def test_stateful_with_config_matches_with_mode():
    mode = fast_vad.mode.normal
    p, min_speech_ms, min_silence_ms, hangover_ms = MODE_CUSTOM_PARAMS[mode]

    vad_mode = fast_vad.VadStateful.with_mode(SAMPLE_RATE_16K, mode)
    vad_cfg = fast_vad.VadStateful.with_config(
        SAMPLE_RATE_16K, p, min_speech_ms, min_silence_ms, hangover_ms
    )

    frames = [make_audio(FRAME_SIZE_16K, seed=100 + i) for i in range(8)]
    assert [vad_mode.detect_frame(f) for f in frames] == [vad_cfg.detect_frame(f) for f in frames]


# ── FeatureExtractor ──────────────────────────────────────────────────────────

def test_feature_extractor_output_shape():
    fe = fast_vad.FeatureExtractor(SAMPLE_RATE_16K)
    audio = make_audio(FRAME_SIZE_16K * 4)
    features = fe.extract_features(audio)
    assert features.shape == (4, 8)
    assert features.dtype == np.float32


def test_feature_extractor_hann_window_shape():
    fe = fast_vad.FeatureExtractor(SAMPLE_RATE_16K)
    assert fe.hann_window.shape == (FRAME_SIZE_16K,)
    assert fe.frame_size == FRAME_SIZE_16K


def test_feature_extractor_short_audio_empty_output():
    fe = fast_vad.FeatureExtractor(SAMPLE_RATE_16K)
    features = fe.extract_features(make_audio(FRAME_SIZE_16K - 1))
    assert features.shape == (0, 8)
