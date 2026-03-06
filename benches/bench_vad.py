"""
Python benchmark for VAD using pytest-benchmark.

Run with:
    uv run pytest benches/bench_vad.py --benchmark-sort=mean --benchmark-group-by=group
"""

import numpy as np
import pytest
import fast_vad

SAMPLE_RATE_16K = 16_000
SAMPLE_RATE_8K = 8_000
FRAME_SIZE_16K = 512
FRAME_SIZE_8K = 256

DURATIONS_16K = [
    ("100ms", 1_600),
    ("1s", 16_000),
    ("10s", 160_000),
    ("1min", 960_000),
    ("10min", 9_600_000),
    ("1hr", 57_600_000),
]

DURATIONS_8K = [
    ("100ms", 800),
    ("1s", 8_000),
    ("10s", 80_000),
    ("1min", 480_000),
    ("10min", 4_800_000),
    ("1hr", 28_800_000),
]


def lcg_noise(n: int, seed: int = 42) -> np.ndarray:
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


def _run_stateful(vad_stateful, frames: np.ndarray) -> None:
    vad_stateful.reset_state()
    for frame in frames:
        vad_stateful.detect_frame(frame)


@pytest.fixture(scope="session")
def vad_16k():
    return fast_vad.VAD.with_mode(SAMPLE_RATE_16K, fast_vad.mode.normal)


@pytest.fixture(scope="session")
def vad_8k():
    return fast_vad.VAD.with_mode(SAMPLE_RATE_8K, fast_vad.mode.normal)


@pytest.fixture(scope="session")
def stateful_16k():
    return fast_vad.VadStateful.with_mode(SAMPLE_RATE_16K, fast_vad.mode.normal)


@pytest.fixture(scope="session")
def stateful_8k():
    return fast_vad.VadStateful.with_mode(SAMPLE_RATE_8K, fast_vad.mode.normal)


def _make_batch_bench_16k(label: str, num_samples: int):
    audio = _make_audio(num_samples)
    audio_duration_s = num_samples / SAMPLE_RATE_16K
    rounds = 5 if num_samples >= 9_600_000 else 20

    def bench(benchmark, vad_16k):
        benchmark.group = "16kHz/detect"
        benchmark.extra_info["audio"] = label
        benchmark.pedantic(
            vad_16k.detect,
            args=(audio,),
            warmup_rounds=3,
            rounds=rounds,
            iterations=1,
        )
        if benchmark.stats:
            mean_s = benchmark.stats["mean"]
            benchmark.extra_info["throughput"] = _throughput_str(num_samples, mean_s)
            benchmark.extra_info["realtime"] = f"{audio_duration_s / mean_s:,.0f}x"

    bench.__name__ = f"test_16000_detect_{label}"
    return bench


def _make_batch_bench_8k(label: str, num_samples: int):
    audio = _make_audio(num_samples)
    audio_duration_s = num_samples / SAMPLE_RATE_8K
    rounds = 5 if num_samples >= 4_800_000 else 20

    def bench(benchmark, vad_8k):
        benchmark.group = "8kHz/detect"
        benchmark.extra_info["audio"] = label
        benchmark.pedantic(
            vad_8k.detect,
            args=(audio,),
            warmup_rounds=3,
            rounds=rounds,
            iterations=1,
        )
        if benchmark.stats:
            mean_s = benchmark.stats["mean"]
            benchmark.extra_info["throughput"] = _throughput_str(num_samples, mean_s)
            benchmark.extra_info["realtime"] = f"{audio_duration_s / mean_s:,.0f}x"

    bench.__name__ = f"test_8000_detect_{label}"
    return bench


def _make_stateful_bench_16k(label: str, num_samples: int):
    audio = _make_audio(num_samples)
    n = (num_samples // FRAME_SIZE_16K) * FRAME_SIZE_16K
    frames = audio[:n].reshape(-1, FRAME_SIZE_16K)
    audio_duration_s = num_samples / SAMPLE_RATE_16K
    rounds = 5 if num_samples >= 9_600_000 else 20

    def bench(benchmark, stateful_16k):
        benchmark.group = "16kHz/stateful_detect_frame"
        benchmark.extra_info["audio"] = label
        benchmark.pedantic(
            _run_stateful,
            args=(stateful_16k, frames),
            warmup_rounds=3,
            rounds=rounds,
            iterations=1,
        )
        if benchmark.stats:
            mean_s = benchmark.stats["mean"]
            benchmark.extra_info["throughput"] = _throughput_str(num_samples, mean_s)
            benchmark.extra_info["realtime"] = f"{audio_duration_s / mean_s:,.0f}x"

    bench.__name__ = f"test_16000_stateful_detect_frame_{label}"
    return bench


def _make_stateful_bench_8k(label: str, num_samples: int):
    audio = _make_audio(num_samples)
    n = (num_samples // FRAME_SIZE_8K) * FRAME_SIZE_8K
    frames = audio[:n].reshape(-1, FRAME_SIZE_8K)
    audio_duration_s = num_samples / SAMPLE_RATE_8K
    rounds = 5 if num_samples >= 4_800_000 else 20

    def bench(benchmark, stateful_8k):
        benchmark.group = "8kHz/stateful_detect_frame"
        benchmark.extra_info["audio"] = label
        benchmark.pedantic(
            _run_stateful,
            args=(stateful_8k, frames),
            warmup_rounds=3,
            rounds=rounds,
            iterations=1,
        )
        if benchmark.stats:
            mean_s = benchmark.stats["mean"]
            benchmark.extra_info["throughput"] = _throughput_str(num_samples, mean_s)
            benchmark.extra_info["realtime"] = f"{audio_duration_s / mean_s:,.0f}x"

    bench.__name__ = f"test_8000_stateful_detect_frame_{label}"
    return bench


for _label, _n in DURATIONS_16K:
    globals()[f"test_16000_detect_{_label}"] = _make_batch_bench_16k(_label, _n)
    globals()[f"test_16000_stateful_detect_frame_{_label}"] = _make_stateful_bench_16k(_label, _n)

for _label, _n in DURATIONS_8K:
    globals()[f"test_8000_detect_{_label}"] = _make_batch_bench_8k(_label, _n)
    globals()[f"test_8000_stateful_detect_frame_{_label}"] = _make_stateful_bench_8k(_label, _n)


if __name__ == "__main__":
    import sys

    sys.exit(
        pytest.main(
            [
                __file__,
                "-v",
                "--benchmark-sort=mean",
                "--benchmark-group-by=group",
                "--benchmark-columns=mean,stddev,min,max,rounds",
                "--benchmark-warmup=on",
            ]
        )
    )
