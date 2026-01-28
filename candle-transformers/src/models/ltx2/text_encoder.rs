use candle::{DType, Result, Tensor, D};

use super::embeddings_connector::Embeddings1DConnector;
use super::feature_extractor::GemmaFeaturesExtractorProjLinear;
use super::tokenizer::LtxvGemmaTokenizer;
use crate::models::gemma3::Model as Gemma3Model;

pub struct AVGemmaEncoderOutput {
    pub video_encoding: Tensor,
    pub audio_encoding: Tensor,
    pub attention_mask: Tensor,
}

pub struct AVGemmaTextEncoderModel {
    pub feature_extractor: GemmaFeaturesExtractorProjLinear,
    pub embeddings_connector: Embeddings1DConnector,
    pub audio_embeddings_connector: Embeddings1DConnector,
    pub tokenizer: Option<LtxvGemmaTokenizer>,
    pub model: Option<Gemma3Model>,
}

impl AVGemmaTextEncoderModel {
    pub fn new(
        feature_extractor: GemmaFeaturesExtractorProjLinear,
        embeddings_connector: Embeddings1DConnector,
        audio_embeddings_connector: Embeddings1DConnector,
    ) -> Self {
        Self {
            feature_extractor,
            embeddings_connector,
            audio_embeddings_connector,
            tokenizer: None,
            model: None,
        }
    }

    fn convert_to_additive_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
        let mask = attention_mask.to_dtype(dtype)?;
        let max = match dtype {
            DType::F16 => 65504.0,
            _ => 1e9,
        };
        let max = Tensor::full(max as f32, (), mask.device())?.to_dtype(dtype)?;
        let mask = mask.broadcast_sub(&Tensor::ones_like(&mask)?)?;
        let mask = mask.broadcast_mul(&max)?;
        let (b, t) = mask.dims2()?;
        mask.reshape((b, 1, 1, t))
    }

    fn norm_and_concat_padded_batch(
        encoded: &Tensor,
        attention_mask: &Tensor,
        padding_side: &str,
    ) -> Result<Tensor> {
        let (b, t, d, l) = encoded.dims4()?;
        let mask = attention_mask
            .to_dtype(encoded.dtype())?
            .reshape((b, t, 1, 1))?;

        let masked = encoded.broadcast_mul(&mask)?;
        let seq = attention_mask
            .sum_keepdim(D::Minus1)?
            .to_dtype(encoded.dtype())?; // [B,1]
        let denom = seq.reshape((b, 1, 1, 1))?.broadcast_mul(&Tensor::full(
            d as f32,
            (),
            encoded.device(),
        )?)?;
        let mean = masked.sum_keepdim(vec![1, 2])?.broadcast_div(&denom)?;

        let one = Tensor::ones_like(&mask)?;
        let inv = mask.broadcast_sub(&one)?;
        let large = Tensor::full(1e4f32, (), encoded.device())?.to_dtype(encoded.dtype())?;
        let min_adjust = encoded.broadcast_sub(&inv.broadcast_mul(&large)?)?; // pad -> +large
        let max_adjust = encoded.broadcast_add(&inv.broadcast_mul(&large)?)?; // pad -> -large
        let x_min = min_adjust.min_keepdim(1)?.min_keepdim(2)?;
        let x_max = max_adjust.max_keepdim(1)?.max_keepdim(2)?;
        let range = x_max.broadcast_sub(&x_min)?;
        let eps = Tensor::full(1e-6f32, (), encoded.device())?.to_dtype(encoded.dtype())?;
        let normed = encoded.broadcast_sub(&mean)?;
        let normed = normed.broadcast_mul(&Tensor::full(8f32, (), encoded.device())?)?;
        let denom = range.broadcast_add(&eps)?;
        let normed = normed.broadcast_div(&denom)?;

        let normed = normed.reshape((b, t, d * l))?;
        let mask_flat = mask.reshape((b, t, 1))?.broadcast_mul(&Tensor::ones(
            (1, 1, d * l),
            mask.dtype(),
            mask.device(),
        )?)?;
        let mask_flat = mask_flat.to_dtype(normed.dtype())?;
        let normed = normed.broadcast_mul(&mask_flat)?;
        if padding_side == "right" || padding_side == "left" {
            Ok(normed)
        } else {
            Ok(normed)
        }
    }

    pub fn forward_tokens(
        &mut self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        padding_side: &str,
    ) -> Result<AVGemmaEncoderOutput> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| candle::Error::msg("gemma3 model not set"))?;
        let hidden_states = model.forward_hidden_states(input_ids, Some(attention_mask), 0)?;
        let hidden_states = Tensor::stack(&hidden_states, D::Minus1)?; // [B,T,D,L]
        let projected =
            Self::norm_and_concat_padded_batch(&hidden_states, attention_mask, padding_side)?;
        let projected = self.feature_extractor.forward(&projected)?;
        let additive_mask = Self::convert_to_additive_mask(attention_mask, projected.dtype())?;
        let (video_encoded, _) = self
            .embeddings_connector
            .forward(&projected, Some(&additive_mask))?;
        let (audio_encoded, _) = self
            .audio_embeddings_connector
            .forward(&projected, Some(&additive_mask))?;
        let mask = attention_mask.to_dtype(projected.dtype())?.reshape((
            projected.dim(0)?,
            projected.dim(1)?,
            1,
        ))?;
        let video_encoded = video_encoded.broadcast_mul(&mask)?;
        let audio_encoded = audio_encoded.broadcast_mul(&mask)?;
        Ok(AVGemmaEncoderOutput {
            video_encoding: video_encoded,
            audio_encoding: audio_encoded,
            attention_mask: attention_mask.clone(),
        })
    }

    pub fn forward(&mut self, text: &str) -> Result<AVGemmaEncoderOutput> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| candle::Error::msg("tokenizer not set"))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| candle::Error::msg("gemma3 model not set"))?;
        let pairs = tokenizer.tokenize_with_weights(text)?;
        let ids: Vec<u32> = pairs.iter().map(|(id, _)| *id).collect();
        let mask: Vec<u32> = pairs.iter().map(|(_, m)| *m).collect();
        let input_ids = Tensor::from_vec(ids, (1, pairs.len()), model.device())?;
        let attention_mask = Tensor::from_vec(mask, (1, pairs.len()), model.device())?;
        self.forward_tokens(&input_ids, &attention_mask, "left")
    }
}

pub fn encode_text(
    text_encoder: &mut AVGemmaTextEncoderModel,
    prompts: &[String],
) -> Result<Vec<(Tensor, Tensor)>> {
    let mut outputs = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let encoded = text_encoder.forward(prompt)?;
        outputs.push((encoded.video_encoding, encoded.audio_encoding));
    }
    Ok(outputs)
}
