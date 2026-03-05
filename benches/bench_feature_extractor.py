"""
Python benchmark for FeatureExtractor, mirroring benches/filterbank_bench.rs.

Measures throughput (samples/s and realtime factor) at the same audio durations
used in the Cargo/Criterion benchmark.
"""

import math
import timeit
import numpy as np
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

WARMUP_REPS = 3    # warm up JIT / rayon thread pool
MEASURE_REPS = 20  # repeats used to estimate mean latency


def lcg_noise(n: int, seed: int = 42) -> np.ndarray:
    """Same LCG as the Rust benchmark for reproducible input."""
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


def format_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def format_throughput(samples: int, seconds: float) -> str:
    elem_per_s = samples / seconds
    if elem_per_s >= 1e9:
        return f"{elem_per_s / 1e9:.3f} Gelem/s"
    elif elem_per_s >= 1e6:
        return f"{elem_per_s / 1e6:.3f} Melem/s"
    return f"{elem_per_s / 1e3:.3f} Kelem/s"


def main():
    fe = fast_vad.FeatureExtractor()

    header = f"{'Duration':<10} {'Samples':>12} {'Mean time':>12} {'Throughput':>14} {'Realtime':>10}"
    print(header)
    print("=" * len(header))

    for label, num_samples in DURATIONS:
        audio_duration_s = num_samples / SAMPLE_RATE

        # Build input (expensive for large sizes; do it once)
        if num_samples <= 160_000:
            audio = lcg_noise(num_samples)
        else:
            # For large inputs use numpy random (fast enough, not reproducible
            # across runs but fine for benchmarking)
            rng = np.random.default_rng(42)
            audio = rng.uniform(-1.0, 1.0, num_samples).astype(np.float32)

        def run():
            fe.extract_features(audio, SAMPLE_RATE)

        # Warm up
        for _ in range(WARMUP_REPS):
            run()

        # Choose measurement count to keep total wall time reasonable
        reps = MEASURE_REPS
        if num_samples >= 9_600_000:
            reps = 5

        elapsed = timeit.timeit(run, number=reps)
        mean_s = elapsed / reps
        realtime_factor = audio_duration_s / mean_s

        print(
            f"{label:<10} {num_samples:>12,} {format_time(mean_s):>12} "
            f"{format_throughput(num_samples, mean_s):>14} "
            f"{realtime_factor:>8.0f}x"
        )


if __name__ == "__main__":
    main()
