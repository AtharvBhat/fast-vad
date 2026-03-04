// SIMD-optimized functions for VAD processing
use crate::vad::constants;
use realfft::num_complex::Complex32;
use wide::{f32x8, f32x16};

// Apply Hanning window using SIMD in place
pub fn apply_hanning_window_simd(frame: &mut [f32]) -> &[f32] {
    let hanning = constants::HANN_512;
    let simd_width = 16;

    // Process in chunks of simd_width
    (0..constants::FRAME_SIZE / simd_width).for_each(|i| {
        let frame_chunk = f32x16::from(&frame[i * simd_width..(i + 1) * simd_width]);
        let hanning_chunk = f32x16::from(&hanning[i * simd_width..(i + 1) * simd_width]);
        let windowed_chunk = frame_chunk * hanning_chunk;
        frame[i * simd_width..(i + 1) * simd_width].copy_from_slice(&windowed_chunk.to_array());
    });
    frame
}

// Compute Band energies using SIMD for the power calculation
pub fn compute_band_energies_simd(spectrum: &[Complex32]) -> f32x8 {
    let mut reals = [0.0f32; constants::FFT_BINS];
    let mut imags = [0.0f32; constants::FFT_BINS];

    spectrum.iter().enumerate().for_each(|(i, c)| {
        reals[i] = c.re;
        imags[i] = c.im;
    });

    // Handle bulk of the data with SIMD
    let simd_width = 16; // Using f32x16 for SIMD processing
    let mut power = [0.0f32; constants::FFT_BINS];
    (0..constants::FFT_BINS / simd_width).for_each(|i| {
        let re = f32x16::from(&reals[i * simd_width..(i + 1) * simd_width]);
        let im = f32x16::from(&imags[i * simd_width..(i + 1) * simd_width]);
        let p = re * re + im * im;
        power[i * simd_width..(i + 1) * simd_width].copy_from_slice(&p.to_array());
    });

    // Handle tail elements that don't fit into a full SIMD register
    (constants::FFT_BINS / simd_width * simd_width..constants::FFT_BINS).for_each(|i| {
        power[i] = reals[i] * reals[i] + imags[i] * imags[i];
    });

    let mut band_energies = [0.0f32; constants::NUM_BANDS];

    // Sum energies for each band
    constants::BAND_BINS
        .iter()
        .enumerate()
        .for_each(|(band_idx, &bin_range)| {
            let sum = power[bin_range.0..bin_range.1].iter().sum::<f32>();
            band_energies[band_idx] = (sum + 1e-10).ln();
        });
    f32x8::from(band_energies)
}
