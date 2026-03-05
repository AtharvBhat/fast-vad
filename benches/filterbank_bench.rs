use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fast_vad::vad::filterbank::FilterBank;

fn generate_audio(num_samples: usize) -> Vec<f32> {
    let mut state: u64 = 42;
    (0..num_samples)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / (u32::MAX as f32);
            2.0 * u - 1.0
        })
        .collect()
}

fn bench_filterbank(c: &mut Criterion) {
    let fb = FilterBank::new(16000);

    let durations: &[(&str, usize)] = &[
        ("100ms", 1_600),
        ("1s", 16_000),
        ("10s", 160_000),
        ("1min", 960_000),
        ("10min", 9_600_000),
        ("1hr", 57_600_000),
    ];

    let mut group = c.benchmark_group("filterbank");

    for &(label, num_samples) in durations {
        let audio = generate_audio(num_samples);

        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("extract_features", label),
            &audio,
            |b, audio| {
                b.iter(|| fb.compute_filterbank(audio));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_filterbank);
criterion_main!(benches);
