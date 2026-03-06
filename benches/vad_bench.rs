use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fast_vad::vad::detector::{VAD, VADModes, VadStateful};

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

fn bench_vad_for_rate(
    c: &mut Criterion,
    sample_rate: usize,
    durations: &[(&str, usize)],
    label: &str,
) {
    let vad = VAD::with_mode(sample_rate, VADModes::Normal).unwrap();
    let frame_size = vad.frame_size();

    let mut group = c.benchmark_group(format!("vad_{label}"));

    for &(duration_label, num_samples) in durations {
        let audio = generate_audio(num_samples);

        group.throughput(Throughput::Elements(num_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("detect", duration_label),
            &audio,
            |b, audio| {
                b.iter(|| black_box(vad.detect(audio)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("stateful_detect_frame", duration_label),
            &audio,
            |b, audio| {
                let mut stateful = VadStateful::with_mode(sample_rate, VADModes::Normal).unwrap();
                b.iter(|| {
                    stateful.reset_state();
                    for frame in audio.chunks_exact(frame_size) {
                        black_box(stateful.detect_frame(frame).expect("valid frame"));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_vad(c: &mut Criterion) {
    let durations_16k: &[(&str, usize)] = &[
        ("100ms", 1_600),
        ("1s", 16_000),
        ("10s", 160_000),
        ("1min", 960_000),
        ("10min", 9_600_000),
        ("1hr", 57_600_000),
    ];

    let durations_8k: &[(&str, usize)] = &[
        ("100ms", 800),
        ("1s", 8_000),
        ("10s", 80_000),
        ("1min", 480_000),
        ("10min", 4_800_000),
        ("1hr", 28_800_000),
    ];

    bench_vad_for_rate(c, 16000, durations_16k, "16k");
    bench_vad_for_rate(c, 8000, durations_8k, "8k");
}

criterion_group!(benches, bench_vad);
criterion_main!(benches);
