use crate::models::voxtral::{
    VoxtralConfig, VoxtralEncoder, VoxtralEncoderConfig, VoxtralLlama, VoxtralLlamaCache,
    VoxtralLlamaConfig, VoxtralMultiModalProjector,
};
use candle::{DType, Device, IndexOp, Result, Tensor};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct GlmAsrConfig {
    pub audio_config: VoxtralEncoderConfig,
    pub text_config: VoxtralLlamaConfig,
    pub projector_hidden_act: String,
    pub projector_hidden_size: Option<usize>,
    pub merge_factor: Option<usize>,
}

impl GlmAsrConfig {
    pub fn merge_factor(&self) -> usize {
        if let Some(merge) = self.merge_factor {
            return merge.max(1);
        }
        let hidden = self.audio_config.hidden_size;
        if hidden == 0 {
            return 1;
        }
        let merge = self.audio_config.intermediate_size / hidden;
        merge.max(1)
    }
}

#[derive(Debug, Clone)]
pub struct GlmAsrCache {
    cache: VoxtralLlamaCache,
    audio_processed: bool,
    cached_audio_embeds: Option<Tensor>,
    cached_audio_positions: Option<Vec<(usize, usize)>>,
}

impl GlmAsrCache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &VoxtralLlamaConfig,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            cache: VoxtralLlamaCache::new(use_kv_cache, dtype, config, device)?,
            audio_processed: false,
            cached_audio_embeds: None,
            cached_audio_positions: None,
        })
    }

    pub fn reset(&mut self) {
        self.audio_processed = false;
        self.cached_audio_embeds = None;
        self.cached_audio_positions = None;
    }
}

#[derive(Debug, Clone)]
pub struct GlmAsrGenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub device: Device,
    pub cache: Option<GlmAsrCache>,
}

impl GlmAsrGenerationConfig {
    pub fn new(device: Device) -> Self {
        Self {
            max_new_tokens: 500,
            temperature: 0.0,
            top_p: None,
            device,
            cache: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlmAsrForConditionalGeneration {
    audio_tower: VoxtralEncoder,
    language_model: VoxtralLlama,
    multi_modal_projector: VoxtralMultiModalProjector,
    audio_config: VoxtralEncoderConfig,
    text_config: VoxtralLlamaConfig,
}

impl GlmAsrForConditionalGeneration {
    pub fn new(cfg: &GlmAsrConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        let audio_tower = VoxtralEncoder::new(&cfg.audio_config, vb.pp("audio_tower"))?;
        let language_model = VoxtralLlama::load(vb.pp("language_model"), &cfg.text_config)?;

        let projector_cfg = VoxtralConfig {
            audio_config: cfg.audio_config.clone(),
            text_config: cfg.text_config.clone(),
            audio_token_id: 0,
            projector_hidden_act: cfg.projector_hidden_act.clone(),
            projector_hidden_size: cfg.projector_hidden_size,
        };
        let multi_modal_projector =
            VoxtralMultiModalProjector::new(&projector_cfg, vb.pp("multi_modal_projector"))?;

        Ok(Self {
            audio_tower,
            language_model,
            multi_modal_projector,
            audio_config: cfg.audio_config.clone(),
            text_config: cfg.text_config.clone(),
        })
    }

    pub fn text_config(&self) -> &VoxtralLlamaConfig {
        &self.text_config
    }

    pub fn audio_config(&self) -> &VoxtralEncoderConfig {
        &self.audio_config
    }

    pub fn get_audio_embeds(&self, input_features: &Tensor) -> Result<Tensor> {
        let audio_outputs = self.audio_tower.forward(input_features)?;

        let (batch_size, seq_len, hidden_size) = audio_outputs.dims3()?;
        let total_elements = batch_size * seq_len * hidden_size;
        let new_batch_size = total_elements / self.audio_config.intermediate_size;

        if total_elements % self.audio_config.intermediate_size != 0 {
            return Err(candle::Error::DimOutOfRange {
                shape: candle::Shape::from_dims(&[batch_size, seq_len, hidden_size]),
                dim: 0,
                op: "reshape",
            });
        }

        let audio_hidden =
            audio_outputs.reshape((new_batch_size, self.audio_config.intermediate_size))?;
        self.multi_modal_projector.forward(&audio_hidden)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_features: Option<&Tensor>,
        audio_offsets: Option<&[Vec<usize>]>,
        audio_lengths: Option<&[Vec<usize>]>,
        cache: &mut GlmAsrCache,
        index_pos: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.language_model.embed(input_ids)?;

        if let Some(features) = input_features {
            if !cache.audio_processed {
                let audio_embeds = self.get_audio_embeds(features)?;
                let (batch_size, seq_len) = input_ids.dims2()?;
                let audio_positions =
                    if let (Some(offsets), Some(lengths)) = (audio_offsets, audio_lengths) {
                        build_audio_positions(offsets, lengths, batch_size, seq_len)?
                    } else if let Some(cached) = &cache.cached_audio_positions {
                        cached.clone()
                    } else {
                        candle::bail!("audio_offsets/audio_lengths are required for the first pass")
                    };

                cache.cached_audio_embeds = Some(audio_embeds.clone());
                cache.cached_audio_positions = Some(audio_positions.clone());
                cache.audio_processed = true;

                inputs_embeds = replace_audio_tokens(
                    &inputs_embeds,
                    &audio_embeds,
                    &audio_positions,
                    input_ids.device(),
                )?;
            }
        }

        self.language_model
            .forward_input_embed(&inputs_embeds, index_pos, &mut cache.cache)
    }

    pub fn generate(
        &self,
        input_ids: &Tensor,
        input_features: Option<&Tensor>,
        audio_offsets: Option<&[Vec<usize>]>,
        audio_lengths: Option<&[Vec<usize>]>,
        config: GlmAsrGenerationConfig,
    ) -> Result<Vec<u32>> {
        if config.max_new_tokens == 0 {
            return input_ids.i(0)?.to_vec1::<u32>();
        }

        if config.temperature < 0.0 {
            candle::bail!(
                "Temperature must be non-negative, got {}",
                config.temperature
            );
        }

        if let Some(p) = config.top_p {
            if !(0.0..=1.0).contains(&p) {
                candle::bail!("top_p must be between 0 and 1, got {}", p);
            }
        }

        let mut final_cache = if let Some(cache) = config.cache {
            cache
        } else {
            let dummy_token = Tensor::new(&[1u32], &config.device)?;
            let dummy_embed = self.language_model.embed(&dummy_token)?;
            let model_dtype = dummy_embed.dtype();
            GlmAsrCache::new(true, model_dtype, &self.text_config, &config.device)?
        };

        let mut tokens = input_ids.i(0)?.to_vec1::<u32>()?;
        let initial_len = tokens.len();

        for idx in 0..config.max_new_tokens {
            let (input, index_pos) = if idx == 0 {
                (input_ids.clone(), 0)
            } else {
                let last_token = tokens[tokens.len() - 1];
                let calculated_pos = initial_len + idx - 1;
                (
                    Tensor::new(&[last_token], &config.device)?.unsqueeze(0)?,
                    calculated_pos,
                )
            };

            let logits = if idx == 0 {
                self.forward(
                    &input,
                    input_features,
                    audio_offsets,
                    audio_lengths,
                    &mut final_cache,
                    index_pos,
                )?
            } else {
                self.forward(&input, None, None, None, &mut final_cache, index_pos)?
            };

            let logits = if logits.dims().len() == 3 {
                logits.i((.., logits.dim(1)? - 1, ..))?
            } else {
                logits
            };

            let next_token = if config.temperature > 0.0 {
                let prs = (logits / config.temperature)?;
                let prs = candle_nn::ops::softmax_last_dim(&prs)?;
                if let Some(top_p_val) = config.top_p {
                    sample_top_p(&prs.squeeze(0)?, top_p_val)?
                } else {
                    sample_from_probs(&prs.squeeze(0)?)?
                }
            } else {
                let logits = logits.squeeze(0)?;
                logits.argmax(0)?.to_scalar::<u32>()?
            };

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

fn build_audio_positions(
    audio_offsets: &[Vec<usize>],
    audio_lengths: &[Vec<usize>],
    batch_size: usize,
    seq_len: usize,
) -> Result<Vec<(usize, usize)>> {
    if audio_offsets.len() != audio_lengths.len() {
        candle::bail!("audio_offsets and audio_lengths must have the same batch size");
    }
    if audio_offsets.len() != batch_size {
        candle::bail!(
            "audio_offsets batch size {} does not match input batch size {}",
            audio_offsets.len(),
            batch_size
        );
    }

    let mut positions = Vec::new();
    for (batch_idx, (offsets, lengths)) in audio_offsets.iter().zip(audio_lengths).enumerate() {
        if offsets.len() != lengths.len() {
            candle::bail!("audio_offsets/audio_lengths length mismatch for batch {batch_idx}");
        }
        for (offset, length) in offsets.iter().zip(lengths) {
            let end = offset.saturating_add(*length);
            if end > seq_len {
                candle::bail!(
                    "audio span {}..{} exceeds sequence length {}",
                    offset,
                    end,
                    seq_len
                );
            }
            for idx in 0..*length {
                positions.push((batch_idx, offset + idx));
            }
        }
    }
    Ok(positions)
}

fn replace_audio_tokens(
    inputs_embeds: &Tensor,
    audio_embeds: &Tensor,
    audio_positions: &[(usize, usize)],
    device: &Device,
) -> Result<Tensor> {
    if audio_positions.is_empty() {
        return Ok(inputs_embeds.clone());
    }

    let (batch_size, seq_len, hidden_size) = inputs_embeds.dims3()?;
    let num_audio_tokens = audio_positions.len();
    let (total_audio_embeds, _) = audio_embeds.dims2()?;

    let audio_embeds = if total_audio_embeds >= num_audio_tokens {
        if num_audio_tokens == total_audio_embeds {
            audio_embeds.clone()
        } else {
            audio_embeds.i(0..num_audio_tokens)?
        }
    } else {
        candle::bail!(
            "Not enough audio embeddings: need {}, got {}",
            num_audio_tokens,
            total_audio_embeds
        );
    };

    let mut result = inputs_embeds.clone();
    for (idx, &(batch_idx, seq_idx)) in audio_positions.iter().enumerate() {
        if batch_idx >= batch_size || seq_idx >= seq_len {
            candle::bail!(
                "Invalid audio position: ({}, {}) for tensor shape ({}, {}, {})",
                batch_idx,
                seq_idx,
                batch_size,
                seq_len,
                hidden_size
            );
        }

        let audio_embed = audio_embeds.i(idx)?;
        let mut position_mask = vec![0f32; batch_size * seq_len];
        position_mask[batch_idx * seq_len + seq_idx] = 1.0;
        let position_mask = Tensor::new(position_mask.as_slice(), device)?
            .reshape((batch_size, seq_len, 1))?
            .to_dtype(inputs_embeds.dtype())?;

        let audio_embed_broadcast = audio_embed.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((
            batch_size,
            seq_len,
            hidden_size,
        ))?;

        let inverse_mask = (1.0 - &position_mask)?;
        result = (result.broadcast_mul(&inverse_mask)?
            + audio_embed_broadcast.broadcast_mul(&position_mask)?)?;
    }

    Ok(result)
}

fn sample_top_p(probs: &Tensor, top_p: f64) -> Result<u32> {
    let probs_vec = probs.to_vec1::<f32>()?;
    let mut indexed_probs: Vec<(usize, f32)> = probs_vec.iter().cloned().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumulative = 0.0;
    let mut filtered = Vec::new();

    for (idx, prob) in indexed_probs {
        cumulative += prob as f64;
        filtered.push((idx, prob));
        if cumulative >= top_p {
            break;
        }
    }

    let filtered_sum: f32 = filtered.iter().map(|(_, p)| p).sum();
    let mut rng = rand::rng();
    let mut sample: f32 = rng.random();
    sample *= filtered_sum;

    let mut acc = 0.0;
    for (idx, prob) in filtered.iter().cloned() {
        acc += prob;
        if sample <= acc {
            return Ok(idx as u32);
        }
    }

    Ok(filtered.last().map(|(idx, _)| *idx as u32).unwrap_or(0))
}

fn sample_from_probs(probs: &Tensor) -> Result<u32> {
    let probs_vec = probs.to_vec1::<f32>()?;
    let mut rng = rand::rng();
    let mut sample: f32 = rng.random();
    for (idx, prob) in probs_vec.iter().enumerate() {
        if sample <= *prob {
            return Ok(idx as u32);
        }
        sample -= *prob;
    }
    Ok(0)
}
