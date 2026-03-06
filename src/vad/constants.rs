// Constants used in the VAD implementation.

use wide::f32x8;

// Number of FFT bins used for analysis. BAND_BINS covers [0, ANALYSIS_BINS),
// which spans 0–4 kHz at 16 kHz sample rate. The Nyquist bin and everything
// above it are intentionally excluded.
pub const ANALYSIS_BINS: usize = 128;

// 8 total frequency bands
pub const NUM_BANDS: usize = 8;

// The VAD operates on 512-sample frames, which corresponds to 32 ms at a 16 kHz sample rate.
pub static BAND_BINS: [(usize, usize); NUM_BANDS] = [
    (3, 6),
    (6, 12),
    (12, 19),
    (19, 32),
    (32, 51),
    (51, 77),
    (77, 108),
    (108, 128),
];

// Logistic regression weights and bias for 8 raw features, 8 noise normalized features
// 8 first order delta features, and 8 second order delta features.
pub static RAW_WEIGHTS: f32x8 = f32x8::new([
    0.01562034,  // raw_80-200
    0.20278303,  // raw_200-380
    0.09269018,  // raw_380-600
    -0.15091796, // raw_600-1k
    -0.01901991, // raw_1k-1.6k
    0.07493567,  // raw_1.6k-2.4k
    -0.08492067, // raw_2.4k-3.2k
    0.28920355,  // raw_3.2k-4k
]);

pub static NORM_WEIGHTS: f32x8 = f32x8::new([
    0.17463323,  // norm_80-200
    0.543_774_2,  // norm_200-380
    0.21194649,  // norm_380-600
    -0.00053826, // norm_600-1k
    0.10791918,  // norm_1k-1.6k
    -0.14064065, // norm_1.6k-2.4k
    0.33679298,  // norm_2.4k-3.2k
    0.152_284_6,  // norm_3.2k-4k
]);

pub static DELTA_WEIGHTS: f32x8 = f32x8::new([
    -0.18790886, // delta_80-200
    -0.73869383, // delta_200-380
    -0.30269337, // delta_380-600
    0.15577473,  // delta_600-1k
    -0.07288475, // delta_1k-1.6k
    0.05904672,  // delta_1.6k-2.4k
    -0.21098153, // delta_2.4k-3.2k
    -0.363_977_5, // delta_3.2k-4k
]);

pub static DELTA2_WEIGHTS: f32x8 = f32x8::new([
    0.08596912,  // delta2_80-200
    0.321_316_9,  // delta2_200-380
    0.139_114,  // delta2_380-600
    -0.02363876, // delta2_600-1k
    0.04661002,  // delta2_1k-1.6k
    -0.00053802, // delta2_1.6k-2.4k
    0.10279381,  // delta2_2.4k-3.2k
    0.12189715,  // delta2_3.2k-4k
]);

pub static BIAS: f32 = -0.50482284;

pub const NOISE_FLOOR_ALPHA: f32 = 0.01;
