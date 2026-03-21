import fast_vad


def test_feature_extractor_repr_16k():
    fe = fast_vad.FeatureExtractor(16000)
    r = repr(fe)
    assert "FeatureExtractor" in r
    assert "sample_rate=16000" in r


def test_feature_extractor_repr_8k():
    fe = fast_vad.FeatureExtractor(8000)
    r = repr(fe)
    assert "FeatureExtractor" in r
    assert "sample_rate=8000" in r


def test_vad_repr_16k():
    vad = fast_vad.VAD(16000)
    r = repr(vad)
    assert "VAD" in r
    assert "sample_rate=16000" in r
    assert "threshold_probability=" in r
    assert "min_speech_ms=" in r
    assert "min_silence_ms=" in r
    assert "hangover_ms=" in r


def test_vad_repr_8k():
    vad = fast_vad.VAD(8000)
    r = repr(vad)
    assert "VAD" in r
    assert "sample_rate=8000" in r
    assert "threshold_probability=" in r
    assert "min_speech_ms=" in r
    assert "min_silence_ms=" in r
    assert "hangover_ms=" in r


def test_vad_stateful_repr_16k():
    vad = fast_vad.VadStateful(16000)
    r = repr(vad)
    assert "VadStateful" in r
    assert "sample_rate=16000" in r
    assert "threshold_probability=" in r
    assert "min_speech_ms=" in r
    assert "min_silence_ms=" in r
    assert "hangover_ms=" in r


def test_vad_stateful_repr_8k():
    vad = fast_vad.VadStateful(8000)
    r = repr(vad)
    assert "VadStateful" in r
    assert "sample_rate=8000" in r
    assert "threshold_probability=" in r
    assert "min_speech_ms=" in r
    assert "min_silence_ms=" in r
    assert "hangover_ms=" in r


def test_feature_extractor_str():
    fe = fast_vad.FeatureExtractor(16000)
    s = str(fe)
    assert "FilterBank" in s
    assert "16000 Hz" in s
    assert "512 samples" in s


def test_vad_str():
    vad = fast_vad.VAD(16000)
    s = str(vad)
    assert "VAD" in s
    assert "16000 Hz" in s
    assert "threshold_probability" in s
    assert "min_speech_ms" in s
    assert "min_silence_ms" in s
    assert "hangover_ms" in s


def test_vad_stateful_str():
    vad = fast_vad.VadStateful(8000)
    s = str(vad)
    assert "VadStateful" in s
    assert "8000 Hz" in s
    assert "threshold_probability" in s
    assert "min_speech_ms" in s
    assert "min_silence_ms" in s
    assert "hangover_ms" in s