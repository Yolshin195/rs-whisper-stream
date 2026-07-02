//! `whisper-stream`: push raw PCM audio in, get recognized text back out,
//! chunk by chunk, as it becomes available.
//!
//! Built on top of [candle](https://github.com/huggingface/candle)'s Whisper
//! implementation (the same one used by `candle-examples/whisper` and
//! `whisper-microphone`), repackaged as a small, reusable, streaming-friendly
//! library instead of a one-shot CLI.
//!
//! # Quick start
//!
//! ```no_run
//! use whisper_stream::{WhisperStreamBuilder, ModelSize};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut stream = WhisperStreamBuilder::new()
//!     .model_size(ModelSize::TinyEn)
//!     .chunk_seconds(10.0) // how much audio to accumulate before each decode pass
//!     .build()?;
//!
//! // pcm_16k is mono f32 PCM sampled at 16 kHz, e.g. read from a mic or a file
//! # let pcm_16k: Vec<f32> = vec![];
//! for block in pcm_16k.chunks(1600) {
//!     for chunk in stream.feed_pcm(block)? {
//!         println!("[{:.1}s] {}", chunk.start, chunk.text);
//!     }
//! }
//! for chunk in stream.flush()? {
//!     println!("[{:.1}s] {}", chunk.start, chunk.text);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Note: Whisper itself isn't a truly incremental streaming model, it decodes
//! a window of audio at a time (like the official streaming demos do). This
//! crate handles the buffering/windowing/re-assembly for you: call
//! [`WhisperStream::feed_pcm`] as new audio arrives, and it will hand back
//! newly recognized [`TranscriptChunk`]s whenever enough audio has been
//! buffered to run another decode pass. Call [`WhisperStream::flush`] once at
//! the end of the stream to get the trailing partial chunk.

pub mod mel;
mod multilingual;

use anyhow::{Error as E, Result};
use candle_core::{IndexOp, Tensor, D};
use candle_nn::{
    ops::{log_softmax, softmax},
    VarBuilder,
};
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::distr::{weighted::WeightedIndex, Distribution};
use rand::SeedableRng;
use tokenizers::Tokenizer;

pub use candle_core::Device;

// ---------------------------------------------------------------------------
// Model wrapper (normal / quantized dispatch)
// ---------------------------------------------------------------------------

enum WModel {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl WModel {
    fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    fn decoder_forward(&mut self, x: &Tensor, xa: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }

    fn reset_kv_cache(&mut self) {
        match self {
            Self::Normal(m) => m.reset_kv_cache(),
            Self::Quantized(m) => m.reset_kv_cache(),
        }
    }
}

pub(crate) fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

// ---------------------------------------------------------------------------
// Public configuration types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Transcribe,
    Translate,
}

/// Convenience presets mapping to the official `openai/whisper-*` (and
/// `distil-whisper/*`) checkpoints on the Hugging Face Hub. Use
/// [`WhisperStreamBuilder::model_id`] instead if you want to point at a
/// custom/fine-tuned repo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSize {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    DistilMediumEn,
    DistilLargeV2,
    DistilLargeV3,
}

impl ModelSize {
    fn is_multilingual(&self) -> bool {
        !matches!(
            self,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn
        )
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::LargeV3Turbo => ("openai/whisper-large-v3-turbo", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
            Self::DistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
        }
    }

    /// Only tiny / tiny.en ship quantized GGUF weights in `lmz/candle-whisper`.
    fn quantized_ext(&self) -> Option<&'static str> {
        match self {
            Self::Tiny => Some("tiny"),
            Self::TinyEn => Some("tiny-en"),
            _ => None,
        }
    }
}

/// A piece of recognized speech, handed back as soon as it's decoded.
#[derive(Debug, Clone)]
pub struct TranscriptChunk {
    /// Offset from the very start of the stream, in seconds.
    pub start: f64,
    /// Duration of the underlying audio window, in seconds.
    pub duration: f64,
    /// Recognized text.
    pub text: String,
    /// Average log-probability of the decoded tokens (higher = more confident).
    pub avg_logprob: f64,
    /// Probability that this window contained no speech at all.
    pub no_speech_prob: f64,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct WhisperStreamBuilder {
    model_id: Option<String>,
    revision: Option<String>,
    size: ModelSize,
    quantized: bool,
    language: Option<String>,
    task: Task,
    timestamps: bool,
    max_initial_timestamp_index: Option<u32>,
    seed: u64,
    chunk_seconds: f64,
    cpu: bool,
    verbose: bool,
    mel_filters: Option<Vec<f32>>,
}

impl Default for WhisperStreamBuilder {
    fn default() -> Self {
        Self {
            model_id: None,
            revision: None,
            size: ModelSize::TinyEn,
            quantized: false,
            language: None,
            task: Task::Transcribe,
            timestamps: false,
            max_initial_timestamp_index: None,
            seed: 299_792_458,
            chunk_seconds: 10.0,
            cpu: true,
            verbose: false,
            mel_filters: None,
        }
    }
}

impl WhisperStreamBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Which official checkpoint to use. Defaults to [`ModelSize::TinyEn`].
    pub fn model_size(mut self, size: ModelSize) -> Self {
        self.size = size;
        self
    }

    /// Override the Hugging Face repo id directly (for custom/fine-tuned models).
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Use quantized GGUF weights (`lmz/candle-whisper`). Only tiny/tiny.en
    /// are currently supported quantized.
    pub fn quantized(mut self, quantized: bool) -> Self {
        self.quantized = quantized;
        self
    }

    /// Force a specific spoken language (e.g. `"en"`, `"ru"`). If unset and
    /// the model is multilingual, the language is auto-detected from the
    /// first audio chunk.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    pub fn task(mut self, task: Task) -> Self {
        self.task = task;
        self
    }

    /// Enable Whisper's internal `<|t|>` timestamp tokens and constraints.
    /// Not required to get [`TranscriptChunk::start`]/`duration` (those are
    /// always computed from the audio window), but improves segmentation
    /// within a window for longer `chunk_seconds`.
    pub fn timestamps(mut self, timestamps: bool) -> Self {
        self.timestamps = timestamps;
        self
    }

    pub fn max_initial_timestamp_index(mut self, idx: u32) -> Self {
        self.max_initial_timestamp_index = Some(idx);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// How many seconds of audio to accumulate before running a decode pass.
    /// Smaller = lower latency but more overhead and less context for the
    /// model; larger = higher latency but better accuracy. Must be <= 30s
    /// (Whisper's fixed encoder window). Defaults to 10s.
    pub fn chunk_seconds(mut self, seconds: f64) -> Self {
        self.chunk_seconds = seconds.min(30.0).max(1.0);
        self
    }

    pub fn cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Supply your own precomputed mel filterbank bytes (e.g. copied from
    /// candle's `melfilters.bytes` / `melfilters128.bytes`) instead of the
    /// filterbank this crate computes on the fly. Bytes must be little-endian
    /// f32, flattened `n_mels x (n_fft/2 + 1)`.
    pub fn mel_filters_from_bytes(mut self, bytes: &[u8]) -> Self {
        let mut filters = vec![0f32; bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(bytes, &mut filters);
        self.mel_filters = Some(filters);
        self
    }

    pub fn build(self) -> Result<WhisperStream> {
        let device = candle_core::Device::Cpu;
        let device = if self.cpu {
            device
        } else {
            candle_core::Device::new_cuda(0).unwrap_or(device)
        };

        let (default_model, default_revision) = if self.quantized {
            ("lmz/candle-whisper", "main")
        } else {
            self.size.model_and_revision()
        };
        let model_id = self.model_id.unwrap_or_else(|| default_model.to_string());
        let revision = self.revision.unwrap_or_else(|| default_revision.to_string());

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let (config_filename, tokenizer_filename, weights_filename) = if self.quantized {
            let ext = self
                .size
                .quantized_ext()
                .ok_or_else(|| E::msg("quantized weights are only available for Tiny/TinyEn"))?;
            (
                repo.get(&format!("config-{ext}.json"))?,
                repo.get(&format!("tokenizer-{ext}.json"))?,
                repo.get(&format!("model-{ext}-q80.gguf"))?,
            )
        } else {
            (
                repo.get("config.json")?,
                repo.get("tokenizer.json")?,
                repo.get("model.safetensors")?,
            )
        };

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let model = if self.quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &weights_filename,
                &device,
            )?;
            WModel::Quantized(m::quantized_model::Whisper::load(&vb, config.clone())?)
        } else {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)?
            };
            WModel::Normal(m::model::Whisper::load(&vb, config.clone())?)
        };

        let mel_filters = self.mel_filters.unwrap_or_else(|| {
            mel::compute_mel_filters(m::SAMPLE_RATE, m::N_FFT, config.num_mel_bins)
        });

        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens: Vec<f32> = (0..config.vocab_size as u32)
            .map(|i| {
                if config.suppress_tokens.contains(&i) || self.timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &device)?;

        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or_else(|| E::msg("unable to find any non-speech token"))?;

        let language_token = match (self.size.is_multilingual(), &self.language) {
            (true, Some(language)) => Some(
                token_id(&tokenizer, &format!("<|{language}|>"))
                    .map_err(|_| E::msg(format!("language {language} is not supported")))?,
            ),
            _ => None,
        };

        let chunk_samples = (self.chunk_seconds * m::SAMPLE_RATE as f64) as usize;

        Ok(WhisperStream {
            model,
            tokenizer,
            device,
            config,
            mel_filters,
            rng: rand::rngs::StdRng::seed_from_u64(self.seed),
            task: self.task,
            timestamps: self.timestamps,
            max_initial_timestamp_index: self.max_initial_timestamp_index,
            verbose: self.verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
            language_token,
            is_multilingual: self.size.is_multilingual(),
            buffered_pcm: Vec::new(),
            chunk_samples,
            total_time_offset: 0.0,
        })
    }
}

// ---------------------------------------------------------------------------
// The streaming transcriber itself
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

/// Gzip-based proxy for text "repetitiveness": highly repetitive text (the
/// classic Whisper degenerate-loop failure mode) compresses very well, so a
/// high ratio here is a strong signal to retry decoding at a higher
/// temperature. Mirrors what the reference OpenAI/candle implementations use.
fn compression_ratio(text: &str) -> f64 {
    use std::io::Write;
    if text.is_empty() {
        return 1.0;
    }
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    if encoder.write_all(text.as_bytes()).is_err() {
        return 1.0;
    }
    match encoder.finish() {
        Ok(compressed) if !compressed.is_empty() => {
            text.len() as f64 / compressed.len() as f64
        }
        _ => 1.0,
    }
}

struct InternalSegment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

pub struct WhisperStream {
    model: WModel,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
    mel_filters: Vec<f32>,
    rng: rand::rngs::StdRng,
    task: Task,
    timestamps: bool,
    max_initial_timestamp_index: Option<u32>,
    verbose: bool,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
    is_multilingual: bool,
    buffered_pcm: Vec<f32>,
    chunk_samples: usize,
    total_time_offset: f64,
}

impl WhisperStream {
    /// Push newly-arrived mono f32 PCM audio, sampled at 16 kHz
    /// (`candle_transformers::models::whisper::m::SAMPLE_RATE`), into the
    /// stream. Whenever enough audio has been buffered, this runs a decode
    /// pass and returns the newly recognized chunks (may be empty if not
    /// enough audio has accumulated yet).
    pub fn feed_pcm(&mut self, pcm: &[f32]) -> Result<Vec<TranscriptChunk>> {
        self.buffered_pcm.extend_from_slice(pcm);
        let mut out = Vec::new();
        while self.buffered_pcm.len() >= self.chunk_samples {
            let chunk: Vec<f32> = self.buffered_pcm.drain(..self.chunk_samples).collect();
            out.extend(self.process_chunk(&chunk)?);
        }
        Ok(out)
    }

    /// Process whatever's left in the buffer (less than a full chunk), e.g.
    /// once the audio source has ended. Call this once after your last
    /// `feed_pcm`.
    pub fn flush(&mut self) -> Result<Vec<TranscriptChunk>> {
        if self.buffered_pcm.is_empty() {
            return Ok(Vec::new());
        }
        let chunk = std::mem::take(&mut self.buffered_pcm);
        self.process_chunk(&chunk)
    }

    /// The 16 kHz sample rate this stream expects `feed_pcm` audio in.
    pub fn expected_sample_rate(&self) -> usize {
        m::SAMPLE_RATE
    }

    fn process_chunk(&mut self, pcm: &[f32]) -> Result<Vec<TranscriptChunk>> {
        let mel = audio::pcm_to_mel(&self.config, pcm, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, self.config.num_mel_bins, mel_len / self.config.num_mel_bins),
            &self.device,
        )?;

        if self.is_multilingual && self.language_token.is_none() {
            let lt = multilingual::detect_language(&mut self.model, &self.tokenizer, &mel)?;
            self.language_token = Some(lt);
        }

        let segments = self.run(&mel)?;
        self.model.reset_kv_cache();

        let base_offset = self.total_time_offset;
        self.total_time_offset += pcm.len() as f64 / m::SAMPLE_RATE as f64;

        Ok(segments
            .into_iter()
            .map(|s| TranscriptChunk {
                start: base_offset + s.start,
                duration: s.duration,
                text: s.dr.text,
                avg_logprob: s.dr.avg_logprob,
                no_speech_prob: s.dr.no_speech_prob,
            })
            .collect())
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let audio_features = self.model.encoder_forward(mel, true)?;
        let sample_len = self.model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            Task::Transcribe => tokens.push(self.transcribe_token),
            Task::Translate => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?.unsqueeze(0)?;
            let ys = self.model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            if i == 0 {
                let logits = self.model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let logits = if self.timestamps {
                self.apply_timestamp_rules(&logits, &tokens)?
            } else {
                logits
            };
            let logits = logits.broadcast_add(&self.suppress_tokens)?;

            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > self.model.config().max_target_positions
            {
                break;
            }
            sum_logprob += prob.ln();
        }
        // Timestamp tokens (<|0.00|>, <|0.04|>, ...) aren't marked as
        // "special" in Whisper's tokenizer, so skip_special_tokens alone
        // doesn't hide them -- filter them out explicitly before turning
        // tokens into text.
        let timestamp_begin = self.no_timestamps_token + 1;
        let text_tokens: Vec<u32> = tokens.iter().copied().filter(|&t| t < timestamp_begin).collect();
        let text = self.tokenizer.decode(&text_tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;
        let compression_ratio = compression_ratio(&text);

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    if self.verbose {
                        eprintln!("decode error at temperature {t}: {err}");
                    }
                }
            }
        }
        unreachable!()
    }

    fn apply_timestamp_rules(&self, input_logits: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let device = input_logits.device().clone();
        let timestamp_begin = self.no_timestamps_token + 1;
        let vocab_size = self.model.config().vocab_size as u32;

        let sample_begin = if self.language_token.is_some() { 3 } else { 2 };
        let sampled_tokens = if tokens.len() > sample_begin {
            &tokens[sample_begin..]
        } else {
            &[]
        };

        let mut masks = Vec::new();
        let mut mask_buffer = vec![0.0f32; vocab_size as usize];

        if !sampled_tokens.is_empty() {
            let last_was_timestamp = sampled_tokens
                .last()
                .map(|&t| t >= timestamp_begin)
                .unwrap_or(false);
            let penultimate_was_timestamp = if sampled_tokens.len() >= 2 {
                sampled_tokens[sampled_tokens.len() - 2] >= timestamp_begin
            } else {
                false
            };

            if last_was_timestamp {
                if penultimate_was_timestamp {
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i >= timestamp_begin { f32::NEG_INFINITY } else { 0.0 };
                    }
                    masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);
                } else {
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i < self.eot_token { f32::NEG_INFINITY } else { 0.0 };
                    }
                    masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);
                }
            }

            let timestamp_tokens: Vec<u32> = sampled_tokens
                .iter()
                .filter(|&&t| t >= timestamp_begin)
                .cloned()
                .collect();
            if !timestamp_tokens.is_empty() {
                let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                    *timestamp_tokens.last().unwrap()
                } else {
                    timestamp_tokens.last().unwrap() + 1
                };
                for i in 0..vocab_size {
                    mask_buffer[i as usize] = if i >= timestamp_begin && i < timestamp_last {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    };
                }
                masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);
            }
        }

        if tokens.len() == sample_begin {
            for i in 0..vocab_size {
                mask_buffer[i as usize] = if i < timestamp_begin { f32::NEG_INFINITY } else { 0.0 };
            }
            masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);

            if let Some(max_initial_timestamp_index) = self.max_initial_timestamp_index {
                let last_allowed = timestamp_begin + max_initial_timestamp_index;
                if last_allowed < vocab_size {
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i > last_allowed { f32::NEG_INFINITY } else { 0.0 };
                    }
                    masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);
                }
            }
        }

        let mut logits = input_logits.clone();
        for mask in masks {
            logits = logits.broadcast_add(&mask)?;
        }

        let log_probs = log_softmax(&logits, 0)?;
        let timestamp_log_probs = log_probs.narrow(
            0,
            timestamp_begin as usize,
            vocab_size as usize - timestamp_begin as usize,
        )?;
        let text_log_probs = log_probs.narrow(0, 0, timestamp_begin as usize)?;

        let timestamp_logprob = {
            let max_val = timestamp_log_probs.max(0)?;
            let shifted = timestamp_log_probs.broadcast_sub(&max_val)?;
            let exp_shifted = shifted.exp()?;
            let sum_exp = exp_shifted.sum(0)?;
            let log_sum = sum_exp.log()?;
            max_val.broadcast_add(&log_sum)?.to_scalar::<f32>()?
        };
        let max_text_token_logprob: f32 = text_log_probs.max(0)?.to_scalar::<f32>()?;

        if timestamp_logprob > max_text_token_logprob {
            for i in 0..vocab_size {
                mask_buffer[i as usize] = if i < timestamp_begin { f32::NEG_INFINITY } else { 0.0 };
            }
            let mask_tensor = Tensor::new(mask_buffer.as_slice(), &device)?;
            logits = logits.broadcast_add(&mask_tensor)?;
        }

        Ok(logits)
    }

    fn run(&mut self, mel: &Tensor) -> Result<Vec<InternalSegment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                continue;
            }
            segments.push(InternalSegment {
                start: time_offset,
                duration: segment_duration,
                dr,
            });
        }
        Ok(segments)
    }
}
