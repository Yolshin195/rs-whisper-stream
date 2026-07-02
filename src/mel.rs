//! Self-contained computation of the log-mel filterbank matrix that Whisper's
//! `pcm_to_mel` needs.
//!
//! The official `candle-examples` Whisper demos ship precomputed
//! `melfilters.bytes` / `melfilters128.bytes` binary blobs. To keep this crate
//! dependency-free (no extra asset files to fetch/ship), we compute the same
//! triangular mel filterbank ourselves, using the standard "Slaney style" mel
//! scale that librosa (and, in turn, OpenAI Whisper's own filter-precompute
//! step) uses. The numerical result is equivalent to the shipped assets.
//!
//! If you already have a `melfilters.bytes` file lying around (e.g. copied
//! from the candle repo) and prefer to use it verbatim, you can still do so:
//! just read the bytes yourself and pass them via
//! [`crate::WhisperStreamBuilder::mel_filters`].

fn hz_to_mel(hz: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let mut mel = (hz - f_min) / f_sp;

    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if hz >= min_log_hz {
        mel = min_log_mel + (hz / min_log_hz).ln() / logstep;
    }
    mel
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let mut hz = f_min + f_sp * mel;

    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= min_log_mel {
        hz = min_log_hz * ((mel - min_log_mel) * logstep).exp();
    }
    hz
}

/// Computes an `n_mels x (n_fft/2 + 1)` triangular mel filterbank, flattened
/// row-major (i.e. filter 0's weights, then filter 1's weights, ...), which
/// is exactly the layout `candle_transformers::models::whisper::audio::pcm_to_mel`
/// expects.
pub fn compute_mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let fmin = 0.0;
    let fmax = sample_rate as f64 / 2.0;
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);

    let mel_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let hz_pts: Vec<f64> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();

    let mut weights = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let f_left = hz_pts[m];
        let f_center = hz_pts[m + 1];
        let f_right = hz_pts[m + 2];

        let enorm = 2.0 / (f_right - f_left);
        for (k, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - f_left) / (f_center - f_left);
            let upper = (f_right - f) / (f_right - f_center);
            let w = lower.min(upper).max(0.0);
            weights[m * n_freqs + k] = (w * enorm) as f32;
        }
    }
    weights
}
