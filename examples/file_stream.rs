//! Reads a .wav file from disk and feeds it into `whisper-stream` in small
//! blocks -- as if it were arriving live from a microphone/network socket --
//! printing recognized text to the terminal as soon as each chunk is decoded.
//!
//! Usage:
//!   cargo run --release --example file_stream -- path/to/audio.wav
//!
//! If no path is given, it looks for `audio.wav` in the current directory.
//! Any sample rate / channel count is fine -- it's downmixed to mono and
//! resampled to 16 kHz automatically.

use std::env;
use whisper_stream::{ModelSize, Task, WhisperStreamBuilder};

fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| "audio.wav".to_string());

    println!("Reading {path} ...");
    let mut reader = hound::WavReader::open(&path)
        .map_err(|e| anyhow::anyhow!("failed to open '{path}': {e}"))?;
    let spec = reader.spec();
    println!(
        "  format: {} Hz, {} channel(s), {} bits ({:?})",
        spec.sample_rate, spec.channels, spec.bits_per_sample, spec.sample_format
    );

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_amplitude = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_amplitude))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };

    // Downmix to mono if needed.
    let mono: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    } else {
        samples
    };

    // Whisper expects 16 kHz mono; resample (simple linear interpolation --
    // good enough for a demo, swap in `rubato` for production-quality resampling).
    let pcm = resample_linear(&mono, spec.sample_rate as usize, 16_000);

    println!("\nLoading model (this downloads weights from the Hugging Face Hub on first run) ...");
    let mut stream = WhisperStreamBuilder::new()
        .model_size(ModelSize::Tiny) // must be a multilingual (non `.en`) checkpoint for non-English audio
        .language("th") // set explicitly instead of relying on auto-detect per chunk
        .task(Task::Transcribe)
        .timestamps(false) // internal <|t|> tokens don't yet drive sub-segmentation in `run()`, just adds overhead
        .chunk_seconds(15.0) // seconds of audio accumulated per decode pass
        .cpu(true)
        .build()?;

    println!("Model ready. Streaming transcription:\n");

    // Simulate audio arriving live: push it in small blocks instead of all at once.
    const FEED_BLOCK: usize = 1600; // 100ms @ 16kHz
    for block in pcm.chunks(FEED_BLOCK) {
        for chunk in stream.feed_pcm(block)? {
            print_chunk(&chunk);
        }
    }
    // Flush whatever's left buffered at the end of the file.
    for chunk in stream.flush()? {
        print_chunk(&chunk);
    }

    Ok(())
}

fn print_chunk(chunk: &whisper_stream::TranscriptChunk) {
    let text = chunk.text.trim();
    if text.is_empty() {
        return;
    }
    println!(
        "[{:>6.1}s - {:>6.1}s] {}",
        chunk.start,
        chunk.start + chunk.duration,
        text
    );
}

fn resample_linear(input: &[f32], from_rate: usize, to_rate: usize) -> Vec<f32> {
    if input.is_empty() || from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = (input.len() as f64 * ratio) as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let idx = src_pos.floor() as usize;
            let frac = src_pos - idx as f64;
            let a = input.get(idx).copied().unwrap_or(0.0);
            let b = input.get(idx + 1).copied().unwrap_or(a);
            (a as f64 + (b as f64 - a as f64) * frac) as f32
        })
        .collect()
}
