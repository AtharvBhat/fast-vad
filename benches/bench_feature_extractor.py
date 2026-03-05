"""
Criterion-style Python benchmark for FeatureExtractor using pytest-benchmark.

Mirrors benches/filterbank_bench.rs:
  - Same audio durations: 100ms → 1hr
  - Same LCG seed (reproducible input)
  - Throughput reported in Gelem/s, matching Criterion's Throughput::Elements
  - Warmup + statistical summary (mean, stddev, min, max, outliers)

Run with:
    uv run benches/bench_feature_extractor.py
"""

import numpy as np
import pytest
import fast_vad

SAMPLE_RATE = 16_000

DURATIONS = [
    ("100ms",  1_600),
    ("1s",     16_000),
    ("10s",    160_000),
    ("1min",   960_000),
    ("10min",  9_600_000),
    ("1hr",    57_600_000),
]


def lcg_noise(n: int, seed: int = 42) -> np.ndarray:
    """Identical LCG to generate_audio() in filterbank_bench.rs."""
    state = np.uint64(seed)
    out = np.empty(n, dtype=np.float32)
    mul = np.uint64(6_364_136_223_846_793_005)
    add = np.uint64(1_442_695_040_888_963_407)
    with np.errstate(over="ignore"):
        for i in range(n):
            state = state * mul + add
            u = float(state >> np.uint64(33)) / float(0xFFFF_FFFF)
            out[i] = np.float32(2.0 * u - 1.0)
    return out


def _make_audio(num_samples: int) -> np.ndarray:
    # LCG is slow for large arrays; fall back to numpy RNG for > 160k samples
    if num_samples <= 160_000:
        return lcg_noise(num_samples)
    rng = np.random.default_rng(42)
    return rng.uniform(-1.0, 1.0, num_samples).astype(np.float32)


def _throughput_str(num_samples: int, mean_s: float) -> str:
    elem_per_s = num_samples / mean_s
    if elem_per_s >= 1e9:
        return f"{elem_per_s / 1e9:.3f} Gelem/s"
    if elem_per_s >= 1e6:
        return f"{elem_per_s / 1e6:.3f} Melem/s"
    return f"{elem_per_s / 1e3:.3f} Kelem/s"


@pytest.fixture(scope="session")
def fe():
    return fast_vad.FeatureExtractor()


# ---------------------------------------------------------------------------
# Dynamically generate one benchmark function per duration, mirroring
# Criterion's  BenchmarkId::new("extract_features", label)  naming.
# ---------------------------------------------------------------------------

def _make_bench(label: str, num_samples: int):
    audio = _make_audio(num_samples)
    audio_duration_s = num_samples / SAMPLE_RATE
    rounds = 5 if num_samples >= 9_600_000 else 20

    def bench(benchmark, fe):
        benchmark.extra_info["num_samples"] = num_samples
        benchmark.extra_info["audio_duration_s"] = audio_duration_s

        benchmark.pedantic(
            fe.extract_features,
            args=(audio, SAMPLE_RATE),
            warmup_rounds=3,
            rounds=rounds,
            iterations=1,
        )

        # Annotate with throughput + realtime factor (shown in extra_info column)
        mean_s = benchmark.stats["mean"]
        benchmark.extra_info["throughput"] = _throughput_str(num_samples, mean_s)
        benchmark.extra_info["realtime"] = f"{audio_duration_s / mean_s:,.0f}x"

    bench.__name__ = f"test_extract_features_{label}"
    return bench


for _label, _n in DURATIONS:
    globals()[f"test_extract_features_{_label}"] = _make_bench(_label, _n)


if __name__ == "__main__":
    import sys

    sys.exit(
        pytest.main([
            __file__,
            "-v",
            "--benchmark-sort=name",
            "--benchmark-columns=mean,stddev,min,max,rounds",
            "--benchmark-warmup=on",
        ])
    )