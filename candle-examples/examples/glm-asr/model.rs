use std::io::Cursor;
use std::path::PathBuf;

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::glmasr::{
    GlmAsrCache, GlmAsrConfig, GlmAsrForConditionalGeneration, GlmAsrGenerationConfig,
};
use candle_transformers::models::voxtral;
use serde_json::Value;
use tokenizers::Tokenizer;

use super::download;

const SAMPLE_RATE: u32 = 16_000;
const CHUNK_SECONDS: usize = 30;
const CHUNK_SAMPLES: usize = SAMPLE_RATE as usize * CHUNK_SECONDS;

#[derive(Debug, serde::Serialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub tokens: Vec<u32>,
}

pub struct GlmAsrModel {
    model: GlmAsrForConditionalGeneration,
    tokenizer: Tokenizer,
    device: Device,
    cache: GlmAsrCache,
    merge_factor: usize,
}

impl GlmAsrModel {
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub fn new(model_id: &str, use_cpu: bool) -> Result<Self> {
        let device = candle_examples::device(use_cpu)?;
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let (model_files, tokenizer_file) = download::model_files(model_id)?;
        let config = load_model_config(&model_files.0)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files.1, dtype, &device)? };
        let model = GlmAsrForConditionalGeneration::new(&config, vb)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(anyhow::Error::msg)?;

        let cache = GlmAsrCache::new(true, dtype, &config.text_config, &device)?;
        let merge_factor = config.merge_factor();

        Ok(Self {
            model,
            tokenizer,
            device,
            cache,
            merge_factor,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// # Errors
    ///
    /// Returns an error if transcription fails.
    pub fn transcribe_audio(
        &mut self,
        audio_data: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
    ) -> Result<TranscriptionResult> {
        if audio_data.is_empty() {
            anyhow::bail!("Audio input is empty");
        }

        let audio = if sample_rate == SAMPLE_RATE {
            audio_data.to_vec()
        } else {
            candle_examples::audio::resample(audio_data, sample_rate, SAMPLE_RATE)
                .context("Failed to resample audio")?
        };

        let audio_len = audio.len();
        let padded_audio = if audio_len % CHUNK_SAMPLES != 0 {
            let target_samples = ((audio_len / CHUNK_SAMPLES) + 1) * CHUNK_SAMPLES;
            let mut padded = audio.clone();
            padded.resize(target_samples, 0.0);
            padded
        } else {
            audio
        };

        let mel_bytes = include_bytes!("../voxtral/melfilters128.bytes");
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        let mut cursor = Cursor::new(mel_bytes);
        cursor.read_f32_into::<LittleEndian>(&mut mel_filters)?;

        let audio_features = voxtral::extract_features(&padded_audio, &mel_filters, &self.device)
            .context("Failed to extract audio features")?;

        let (tokens, audio_offsets, audio_lengths) =
            build_prompt(&self.tokenizer, audio_len, SAMPLE_RATE, self.merge_factor)?;

        let prompt_len = tokens.len();
        let input_ids = Tensor::new(tokens.clone(), &self.device)?.unsqueeze(0)?;
        let batch_offsets = vec![audio_offsets];
        let batch_lengths = vec![audio_lengths];

        let generation_config = GlmAsrGenerationConfig {
            max_new_tokens,
            temperature: 0.0,
            top_p: None,
            device: self.device.clone(),
            cache: Some(self.cache.clone()),
        };

        let generated_tokens = self.model.generate(
            &input_ids,
            Some(&audio_features),
            Some(&batch_offsets),
            Some(&batch_lengths),
            generation_config,
        )?;

        let new_tokens = if generated_tokens.len() > prompt_len {
            &generated_tokens[prompt_len..]
        } else {
            &generated_tokens[..]
        };

        let text = self
            .tokenizer
            .decode(new_tokens, true)
            .map_err(anyhow::Error::msg)?;

        Ok(TranscriptionResult {
            text,
            tokens: new_tokens.to_vec(),
        })
    }
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
    Ok(encoding.get_ids().to_vec())
}

fn build_prompt(
    tokenizer: &Tokenizer,
    audio_len: usize,
    sample_rate: u32,
    merge_factor: usize,
) -> Result<(Vec<u32>, Vec<usize>, Vec<usize>)> {
    let num_chunks = (audio_len + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES;
    if num_chunks == 0 {
        anyhow::bail!("No audio chunks generated");
    }

    let mut tokens = Vec::new();
    let mut audio_offsets = Vec::new();
    let mut audio_lengths = Vec::new();

    tokens.extend(encode(tokenizer, "<|user|>")?);
    tokens.extend(encode(tokenizer, "\n")?);

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * CHUNK_SAMPLES;
        let end = ((chunk_idx + 1) * CHUNK_SAMPLES).min(audio_len);
        let seconds = (end - start) as f32 / sample_rate as f32;
        let audio_tokens = get_audio_token_length(seconds, merge_factor);

        tokens.extend(encode(tokenizer, "<|begin_of_audio|>")?);
        audio_offsets.push(tokens.len());
        tokens.extend(vec![0u32; audio_tokens]);
        tokens.extend(encode(tokenizer, "<|end_of_audio|>")?);
        audio_lengths.push(audio_tokens);
    }

    tokens.extend(encode(tokenizer, "<|user|>")?);
    tokens.extend(encode(
        tokenizer,
        "\nPlease transcribe this audio into text",
    )?);
    tokens.extend(encode(tokenizer, "<|assistant|>")?);
    tokens.extend(encode(tokenizer, "\n")?);

    Ok((tokens, audio_offsets, audio_lengths))
}

fn get_audio_token_length(seconds: f32, merge_factor: usize) -> usize {
    if seconds <= 0.0 {
        return 0;
    }

    let mel_len = (seconds * 100.0) as usize;
    if mel_len == 0 {
        return 0;
    }
    let audio_len_after_cnn = get_t_after_cnn(mel_len);
    if audio_len_after_cnn < merge_factor {
        return 0;
    }

    let mut audio_tokens = (audio_len_after_cnn - merge_factor) / merge_factor + 1;
    let max_tokens = 1500 / merge_factor;
    if audio_tokens > max_tokens {
        audio_tokens = max_tokens;
    }
    audio_tokens
}

fn get_t_after_cnn(mut length: usize) -> usize {
    length = (length + 2 - 2 - 1) + 1;
    length = (length + 2 - 2 - 1) / 2 + 1;
    length
}

fn load_model_config(config_file: &PathBuf) -> Result<GlmAsrConfig> {
    let config_str = std::fs::read_to_string(config_file)?;
    let json: Value = serde_json::from_str(&config_str).context("Failed to parse config.json")?;

    let audio_json = json
        .get("audio_config")
        .or_else(|| json.get("whisper_config"))
        .ok_or_else(|| anyhow::anyhow!("Missing audio_config in configuration"))?;

    let text_json = json
        .get("text_config")
        .or_else(|| json.get("lm_config"))
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in configuration"))?;

    let use_rope = json
        .get("use_rope")
        .and_then(|v| v.as_bool())
        .unwrap_or_else(|| {
            audio_json.get("use_rope").and_then(|v| v.as_bool()).unwrap_or_else(|| {
                audio_json.get("rope_parameters").is_some()
                    || audio_json.get("partial_rotary_factor").is_some()
            })
        });
    let audio_config = parse_audio_config(audio_json, use_rope)?;
    let text_config = parse_text_config(text_json)?;
    let projector_hidden_act = json
        .get("projector_hidden_act")
        .and_then(|v| v.as_str())
        .unwrap_or("gelu")
        .to_string();
    let projector_hidden_size = json
        .get("projector_hidden_size")
        .or_else(|| json.get("projector_hidden_dim"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or_else(|| {
            if audio_config.use_glm_encoder_names {
                Some(text_config.hidden_size * 2)
            } else {
                None
            }
        });
    let merge_factor = json
        .get("merge_factor")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    Ok(GlmAsrConfig {
        audio_config,
        text_config,
        projector_hidden_act,
        projector_hidden_size,
        merge_factor,
    })
}

fn parse_audio_config(
    audio_json: &Value,
    use_rope: bool,
) -> Result<voxtral::VoxtralEncoderConfig> {
    let hidden_size = audio_json
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1280) as usize;
    let num_attention_heads = audio_json
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;
    let head_dim = audio_json
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| hidden_size / num_attention_heads);
    let intermediate_size = audio_json
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size * 4);
    let model_type = audio_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let use_glm_encoder_names = model_type == "glmasr_encoder";
    let rope_theta = audio_json
        .get("rope_theta")
        .or_else(|| {
            audio_json
                .get("rope_parameters")
                .and_then(|v| v.get("rope_theta"))
        })
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000.0) as f32;
    let partial_rotary_factor = audio_json
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(if use_rope { 0.5 } else { 1.0 }) as f32;

    Ok(voxtral::VoxtralEncoderConfig {
        vocab_size: audio_json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(51866) as usize,
        hidden_size,
        num_hidden_layers: audio_json
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize,
        num_attention_heads,
        num_key_value_heads: audio_json
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(num_attention_heads as u64) as usize,
        head_dim,
        intermediate_size,
        dropout: audio_json
            .get("dropout")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        attention_dropout: audio_json
            .get("attention_dropout")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        activation_dropout: audio_json
            .get("activation_dropout")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        activation_function: audio_json
            .get("activation_function")
            .or_else(|| audio_json.get("hidden_act"))
            .and_then(|v| v.as_str())
            .unwrap_or("gelu")
            .to_string(),
        max_source_positions: audio_json
            .get("max_source_positions")
            .or_else(|| audio_json.get("max_position_embeddings"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1500) as usize,
        layerdrop: audio_json
            .get("layerdrop")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        initializer_range: audio_json
            .get("initializer_range")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.02),
        scale_embedding: audio_json
            .get("scale_embedding")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        num_mel_bins: audio_json
            .get("num_mel_bins")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize,
        use_rope,
        rope_theta,
        partial_rotary_factor,
        use_glm_encoder_names,
    })
}

fn parse_text_config(text_json: &Value) -> Result<voxtral::VoxtralLlamaConfig> {
    Ok(voxtral::VoxtralLlamaConfig {
        hidden_size: text_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize,
        intermediate_size: text_json
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(11008) as usize,
        vocab_size: text_json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize,
        num_hidden_layers: text_json
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize,
        num_attention_heads: text_json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize,
        num_key_value_heads: text_json
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize,
        head_dim: text_json
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        use_flash_attn: text_json
            .get("use_flash_attn")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        rms_norm_eps: text_json
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5),
        rope_theta: text_json
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32,
        max_position_embeddings: text_json
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize,
        tie_word_embeddings: text_json
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    })
}
