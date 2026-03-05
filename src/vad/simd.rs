// SIMD-optimized functions for VAD processing
use crate::vad::constants;
use realfft::num_complex::Complex32;
use wide::{f32x8, f32x16};

// Apply Hanning window using SIMD in place
pub fn apply_hanning_window_simd(frame: &mut [f32]) {
    debug_assert_eq!(frame.len(), constants::FRAME_SIZE);
    let hanning = constants::HANN_512;
    let simd_width = 16;

    // Process in chunks of simd_width
    (0..constants::FRAME_SIZE / simd_width).for_each(|i| {
        let frame_chunk = f32x16::from(&frame[i * simd_width..(i + 1) * simd_width]);
        let hanning_chunk = f32x16::from(&hanning[i * simd_width..(i + 1) * simd_width]);
        let windowed_chunk = frame_chunk * hanning_chunk;
        frame[i * simd_width..(i + 1) * simd_width].copy_from_slice(&windowed_chunk.to_array());
    });
}

// Compute Band energies using SIMD for the power calculation
pub fn compute_band_energies_simd(spectrum: &[Complex32]) -> f32x8 {
    debug_assert!(spectrum.len() >= constants::ANALYSIS_BINS);

    let mut reals = [0.0f32; constants::ANALYSIS_BINS];
    let mut imags = [0.0f32; constants::ANALYSIS_BINS];

    spectrum
        .iter()
        .take(constants::ANALYSIS_BINS)
        .enumerate()
        .for_each(|(i, c)| {
            reals[i] = c.re;
            imags[i] = c.im;
        });

    // Handle bulk of the data with SIMD
    let simd_width = 16; // Using f32x16 for SIMD processing
    let mut power = [0.0f32; constants::ANALYSIS_BINS];
    (0..constants::ANALYSIS_BINS / simd_width).for_each(|i| {
        let re = f32x16::from(&reals[i * simd_width..(i + 1) * simd_width]);
        let im = f32x16::from(&imags[i * simd_width..(i + 1) * simd_width]);
        let p = re * re + im * im;
        power[i * simd_width..(i + 1) * simd_width].copy_from_slice(&p.to_array());
    });

    // Handle tail elements that don't fit into a full SIMD register
    (constants::ANALYSIS_BINS / simd_width * simd_width..constants::ANALYSIS_BINS).for_each(|i| {
        power[i] = reals[i] * reals[i] + imags[i] * imags[i];
    });

    let mut band_energies = [0.0f32; constants::NUM_BANDS];

    // Sum energies for each band
    constants::BAND_BINS
        .iter()
        .enumerate()
        .for_each(|(band_idx, &(lo, hi))| {
            let sum = power[lo..hi].iter().sum::<f32>();
            band_energies[band_idx] = (sum + 1e-10).ln();
        });
    f32x8::from(band_energies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::constants;
    use realfft::num_complex::Complex32;
    use std::f32::consts::PI;

    fn zero_spectrum() -> Vec<Complex32> {
        vec![Complex32::new(0.0, 0.0); constants::FFT_BINS]
    }

    // ─── apply_hanning_window_simd ──────────────────────────────────────────

    #[test]
    fn windowing_zeros_stay_zero() {
        let mut frame = vec![0.0f32; constants::FRAME_SIZE];
        apply_hanning_window_simd(&mut frame);
        assert!(frame.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn windowing_tapers_edges() {
        let mut frame = vec![1.0f32; constants::FRAME_SIZE];
        apply_hanning_window_simd(&mut frame);

        assert!(
            frame[0].abs() < 1e-6,
            "first sample not tapered: {}",
            frame[0]
        );
        assert!(
            frame[constants::FRAME_SIZE - 1].abs() < 1e-6,
            "last sample not tapered: {}",
            frame[constants::FRAME_SIZE - 1]
        );

        let mid = frame[constants::FRAME_SIZE / 2 - 1];
        assert!((mid - 1.0).abs() < 0.01, "center should be ~1.0, got {mid}");
    }

    #[test]
    fn windowing_matches_manual() {
        let mut frame: Vec<f32> = (0..constants::FRAME_SIZE).map(|i| i as f32).collect();
        let expected: Vec<f32> = (0..constants::FRAME_SIZE)
            .map(|i| i as f32 * constants::HANN_512[i])
            .collect();

        apply_hanning_window_simd(&mut frame);

        for i in 0..constants::FRAME_SIZE {
            assert!(
                (frame[i] - expected[i]).abs() < 1e-4,
                "mismatch at {i}: got {}, expected {}",
                frame[i],
                expected[i]
            );
        }
    }

    #[test]
    fn windowing_sine_reduces_edge_amplitude() {
        // A sine wave should have its edge samples significantly reduced after windowing.
        let freq = 500.0f32;
        let mut frame: Vec<f32> = (0..constants::FRAME_SIZE)
            .map(|i| (2.0 * PI * freq * i as f32 / 16000.0).sin())
            .collect();
        let original_edge = frame[1].abs();
        apply_hanning_window_simd(&mut frame);
        assert!(
            frame[1].abs() < original_edge * 0.1,
            "edge not sufficiently tapered: before={original_edge:.4}, after={:.4}",
            frame[1].abs()
        );
    }

    // ─── compute_band_energies_simd ─────────────────────────────────────────

    #[test]
    fn zero_spectrum_at_energy_floor() {
        // All-zero spectrum → each band sums to 0 → ln(0 + 1e-10) ≈ -23.025
        let spectrum = zero_spectrum();
        let result = compute_band_energies_simd(&spectrum).to_array();
        let floor = 1e-10f32.ln();
        for (band, &energy) in result.iter().enumerate() {
            assert!(
                (energy - floor).abs() < 0.01,
                "band {band}: expected floor {floor:.3}, got {:.3}",
                energy
            );
        }
    }

    #[test]
    fn impulse_in_band_dominates_others() {
        // A large impulse at bin 10 (inside band 1: bins 6–12) should make
        // band 1 dominate all other bands by a wide margin.
        let mut spectrum = zero_spectrum();
        spectrum[10] = Complex32::new(100.0, 0.0);
        let result = compute_band_energies_simd(&spectrum).to_array();

        for band in 0..constants::NUM_BANDS {
            if band != 1 {
                assert!(
                    result[1] > result[band] + 5.0,
                    "band 1 ({:.2}) should dominate band {band} ({:.2})",
                    result[1],
                    result[band]
                );
            }
        }
    }

    #[test]
    fn energy_matches_scalar() {
        // SIMD output must agree with a straightforward scalar computation.
        let mut spectrum = zero_spectrum();
        for &(lo, hi) in &constants::BAND_BINS {
            let mid = (lo + hi) / 2;
            spectrum[mid] = Complex32::new(1.0, 0.5);
        }

        let simd_result = compute_band_energies_simd(&spectrum).to_array();

        for (band, &(lo, hi)) in constants::BAND_BINS.iter().enumerate() {
            let power_sum: f32 = spectrum[lo..hi]
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .sum();
            let expected = (power_sum + 1e-10f32).ln();
            assert!(
                (simd_result[band] - expected).abs() < 1e-4,
                "band {band}: SIMD={:.4} vs scalar={:.4}",
                simd_result[band],
                expected
            );
        }
    }

    #[test]
    fn real_and_imaginary_contribute_equally() {
        // |re + 0i|² == |0 + im·i|² for equal magnitudes, so both components
        // should give identical band energy.  Combined (re + im·i) should give
        // ln(2) ≈ 0.693 more energy than either component alone.
        let mut spec_real = zero_spectrum();
        let mut spec_imag = zero_spectrum();
        let mut spec_both = zero_spectrum();

        // bin 5 is inside band 0 (bins 3–6)
        spec_real[5] = Complex32::new(1.0, 0.0);
        spec_imag[5] = Complex32::new(0.0, 1.0);
        spec_both[5] = Complex32::new(1.0, 1.0); // power = 2.0

        let e_real = compute_band_energies_simd(&spec_real).to_array()[0];
        let e_imag = compute_band_energies_simd(&spec_imag).to_array()[0];
        let e_both = compute_band_energies_simd(&spec_both).to_array()[0];

        assert!(
            (e_real - e_imag).abs() < 1e-4,
            "real ({e_real:.3}) and imaginary ({e_imag:.3}) should give equal energy"
        );
        let diff = e_both - e_real;
        assert!(
            (diff - 0.693).abs() < 0.05,
            "expected ln(2)≈0.693 increase for both components, got {diff:.3}"
        );
    }

    #[test]
    fn larger_amplitude_higher_energy() {
        // Doubling the spectral amplitude → 4× power → ln(4) ≈ 1.386 increase.
        let mut spec_small = zero_spectrum();
        let mut spec_large = zero_spectrum();

        // bin 15 is inside band 2 (bins 12–19)
        spec_small[15] = Complex32::new(1.0, 0.0);
        spec_large[15] = Complex32::new(2.0, 0.0);

        let e_small = compute_band_energies_simd(&spec_small).to_array()[2];
        let e_large = compute_band_energies_simd(&spec_large).to_array()[2];
        let diff = e_large - e_small;

        assert!(
            (diff - 1.386).abs() < 0.05,
            "expected ln(4)≈1.386 increase, got {diff:.3}"
        );
    }
}
