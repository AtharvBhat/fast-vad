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
    -0.32454994,  // raw_80-200
    0.98038036,  // raw_200-380
    0.22799249,  // raw_380-600
    -1.03271973,  // raw_600-1k
    -0.10103340,  // raw_1k-1.6k
    0.23794277,  // raw_1.6k-2.4k
    0.32295096,  // raw_2.4k-3.2k
    -0.04611600,  // raw_3.2k-4k
]);

pub static NORM_WEIGHTS: f32x8 = f32x8::new([
    0.52500838,  // norm_80-200
    -0.48988292,  // norm_200-380
    -0.04958648,  // norm_380-600
    0.86406046,  // norm_600-1k
    0.19488138,  // norm_1k-1.6k
    -0.27344853,  // norm_1.6k-2.4k
    -0.13241436,  // norm_2.4k-3.2k
    0.43088898,  // norm_3.2k-4k
]);

pub static DELTA_WEIGHTS: f32x8 = f32x8::new([
    -0.21237020,  // delta_80-200
    -0.49818158,  // delta_200-380
    -0.18461762,  // delta_380-600
    0.16449890,  // delta_600-1k
    -0.08347286,  // delta_1k-1.6k
    0.04367284,  // delta_1.6k-2.4k
    -0.16818947,  // delta_2.4k-3.2k
    -0.32984850,  // delta_3.2k-4k
]);

pub static DELTA2_WEIGHTS: f32x8 = f32x8::new([
    0.09583185,  // delta2_80-200
    0.22922470,  // delta2_200-380
    0.08839030,  // delta2_380-600
    -0.03120437,  // delta2_600-1k
    0.04511663,  // delta2_1k-1.6k
    -0.00238994,  // delta2_1.6k-2.4k
    0.08310395,  // delta2_2.4k-3.2k
    0.11490002,  // delta2_3.2k-4k
]);

pub static BIAS: f32 = 0.25309636;
