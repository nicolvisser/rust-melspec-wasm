use num::complex::Complex;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn mel_spectrogram_db(
    sample_rate: f32,
    waveform: Vec<f32>,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    top_db: f32,
) -> JsValue {
    let mel_spec: Vec<Vec<f32>> = mel_spectrogram(
        sample_rate,
        waveform,
        n_fft,
        win_length,
        hop_length,
        f_min,
        f_max,
        n_mels,
    );
    let result: Vec<Vec<f32>> = amplitude_to_db(mel_spec, top_db);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

fn mel_spectrogram(
    sample_rate: f32,
    waveform: Vec<f32>,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
) -> Vec<Vec<f32>> {
    let spectrogram = spectrogram(waveform, n_fft, win_length, hop_length, true);
    let n_freqs = spectrogram[0].len();
    let fbanks = mel_filter_bank(n_freqs, f_min, f_max, n_mels, sample_rate);

    let mel_spec = spectrogram
        .iter()
        .map(|spec_row| {
            (0..n_mels)
                .map(|j| {
                    spec_row
                        .iter()
                        .zip(fbanks.iter())
                        .map(|(&s, fbank)| s * fbank[j])
                        .sum()
                })
                .collect()
        })
        .collect();

    mel_spec
}

fn spectrogram(
    waveform: Vec<f32>,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    onesided: bool,
) -> Vec<Vec<f32>> {
    assert!(n_fft >= win_length);
    assert!(win_length >= hop_length);

    let pad_size = (win_length - hop_length) / 2;

    let padded_waveform: Vec<f32> = vec![0.0; pad_size]
        .into_iter()
        .chain(waveform)
        .chain(vec![0.0; pad_size])
        .collect();

    let mut spectrogram = Vec::new();
    for i in (0..padded_waveform.len() - win_length - pad_size).step_by(hop_length) {
        let window = &padded_waveform[i..i + win_length];
        let windowed_input: Vec<f32> = window
            .iter()
            .zip(hann_window(win_length).iter())
            .map(|(a, b)| a * b)
            .collect();
        let fft_output: Vec<Complex<f32>> = fft(windowed_input, n_fft).into_iter().collect();
        let magnitude_spectrum: Vec<f32> = fft_output.iter().map(|x| x.norm()).collect();
        spectrogram.push(magnitude_spectrum);
    }

    if onesided {
        spectrogram
            .iter()
            .map(|row| row.iter().take(n_fft / 2 + 1).cloned().collect())
            .collect()
    } else {
        spectrogram
    }
}

fn fft(input: Vec<f32>, n_fft: usize) -> Vec<Complex<f32>> {
    let num_samples = input.len();
    assert!(n_fft.is_power_of_two(), "n_fft must be a power of 2");
    assert!(
        num_samples <= n_fft,
        "n must be less than or equal to n_fft"
    );

    if num_samples <= 1 {
        return input.into_iter().map(|x| Complex::new(x, 0.0)).collect();
    }

    let padded_input = if num_samples < n_fft {
        let padding = vec![0.0; n_fft - num_samples];
        input.clone().into_iter().chain(padding).collect()
    } else {
        input.clone()
    };

    // Split into even and odd parts
    let even: Vec<f32> = padded_input.iter().step_by(2).cloned().collect();
    let odd: Vec<f32> = padded_input.iter().skip(1).step_by(2).cloned().collect();

    // Recursive FFT on even and odd parts
    let even_fft = fft(even, n_fft / 2);
    let odd_fft = fft(odd, n_fft / 2);

    // Combine results
    let mut output = vec![Complex::new(0.0, 0.0); n_fft];
    for k in 0..(n_fft / 2) {
        let t = odd_fft[k]
            * Complex::from_polar(1.0, -2.0 * std::f32::consts::PI * k as f32 / n_fft as f32);
        output[k] = even_fft[k] + t;
        output[k + n_fft / 2] = even_fft[k] - t; // Exploit symmetry
    }
    output
}

fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (length - 1) as f32).cos()))
        .collect()
}

fn mel_filter_bank(
    n_freqs: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    sample_rate: f32,
) -> Vec<Vec<f32>> {
    let f_nyquist = sample_rate / 2.0;

    let all_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| f_nyquist * i as f32 / (n_freqs - 1) as f32)
        .collect(); // (n_freqs,)

    let m_min = hz_to_mel(f_min);
    let m_max = hz_to_mel(f_max);

    let m_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32)
        .collect(); // (n_mels + 2,)

    let f_points: Vec<f32> = m_points.iter().map(|&mel| mel_to_hz(mel)).collect();

    let f_diff: Vec<f32> = f_points
        .iter()
        .skip(1)
        .zip(f_points.iter().take(f_points.len() - 1))
        .map(|(f2, f1)| f2 - f1)
        .collect(); // (n_mels + 1,)

    let slopes: Vec<Vec<f32>> = all_freqs
        .iter()
        .map(|&f| f_points.iter().map(|&fp| fp - f).collect())
        .collect(); // (n_freqs, n_mels + 2)

    let down_slopes: Vec<Vec<f32>> = slopes
        .iter()
        .map(|slope_slice| {
            slope_slice
                .iter()
                .take(n_mels)
                .zip(f_diff.iter().take(n_mels))
                .map(|(slope, &diff)| -1.0 * slope / diff)
                .collect()
        })
        .collect(); // (n_freqs, n_mels)

    let up_slopes: Vec<Vec<f32>> = slopes
        .iter()
        .map(|slope_slice| {
            slope_slice
                .iter()
                .skip(2)
                .take(n_mels)
                .zip(f_diff.iter().skip(1).take(n_mels))
                .map(|(slope, &diff)| slope / diff)
                .collect()
        })
        .collect();

    let mut fbanks: Vec<Vec<f32>> = up_slopes
        .iter()
        .zip(down_slopes.iter())
        .map(|(up, down)| {
            let row = down
                .iter()
                .zip(up.iter())
                .map(|(&d, &u)| d.min(u).max(0.0)) // Use both up and down slopes
                .collect::<Vec<f32>>();
            row
        })
        .collect();

    // Apply Slaney normalization
    for i in 0..n_mels {
        let enorm = 2.0 / (f_points[i + 2] - f_points[i]);
        for fbank in fbanks.iter_mut() {
            fbank[i] *= enorm;
        }
    }

    fbanks
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn amplitude_to_db(amplitudes: Vec<Vec<f32>>, top_db: f32) -> Vec<Vec<f32>> {
    let dbs: Vec<Vec<f32>> = amplitudes
        .iter()
        .map(|row| {
            row.iter()
                .map(|&amp| 20.0 * amp.max(1e-10).log10())
                .collect()
        })
        .collect();

    let max_db = dbs
        .iter()
        .map(|row| row.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
        .fold(f32::NEG_INFINITY, f32::max);

    dbs.iter()
        .map(|row| row.iter().map(|&amp| amp.max(max_db - top_db)).collect())
        .collect()
}

fn _assert_complex_eq(left: Complex<f32>, right: Complex<f32>) {
    // tolerance for floating-point comparison
    const EPSILON: f32 = 1e-5;

    assert!(
        (left.re - right.re).abs() < EPSILON,
        "left: {:?}, right: {:?}",
        left,
        right
    );
    assert!(
        (left.im - right.im).abs() < EPSILON,
        "left: {:?}, right: {:?}",
        left,
        right
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::complex::Complex;

    #[test]
    fn test_fft_constant() {
        let input = vec![1.0, 0.0, 0.0, 0.0]; // Changed to real input
        let output = fft(input, 4);
        _assert_complex_eq(output[0], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[2], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[3], Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_fft_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // Changed to real input
        let output = fft(input, 4);
        _assert_complex_eq(output[0], Complex::new(10.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(-2.0, 2.0));
        _assert_complex_eq(output[2], Complex::new(-2.0, 0.0));
        _assert_complex_eq(output[3], Complex::new(-2.0, -2.0));
    }

    #[test]
    fn test_fft_with_length_eight() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]; // Changed to real input
        let output = fft(input, 8);
        _assert_complex_eq(output[0], Complex::new(10.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(-0.41421356, -7.24264069));
        _assert_complex_eq(output[2], Complex::new(-2.0, 2.0));
        _assert_complex_eq(output[3], Complex::new(2.41421356, -1.24264069));
        _assert_complex_eq(output[4], Complex::new(-2.0, 0.0));
        _assert_complex_eq(output[5], Complex::new(2.41421356, 1.24264069));
        _assert_complex_eq(output[6], Complex::new(-2.0, -2.0));
        _assert_complex_eq(output[7], Complex::new(-0.41421356, 7.24264069));
    }
}
