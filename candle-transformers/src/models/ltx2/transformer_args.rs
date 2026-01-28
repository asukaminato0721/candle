use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::Linear;

use super::adaln::AdaLayerNormSingle;
use super::rope::{precompute_freqs_cis, LtxRopeType};
use super::text_projection::PixArtAlphaTextProjection;

#[derive(Debug, Clone)]
pub struct Modality {
    pub latent: Tensor,
    pub timesteps: Tensor,
    pub positions: Tensor,
    pub context: Tensor,
    pub enabled: bool,
    pub context_mask: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct TransformerArgs {
    pub x: Tensor,
    pub context: Tensor,
    pub context_mask: Option<Tensor>,
    pub timesteps: Tensor,
    pub embedded_timestep: Tensor,
    pub positional_embeddings: (Tensor, Tensor),
    pub cross_positional_embeddings: Option<(Tensor, Tensor)>,
    pub cross_scale_shift_timestep: Option<Tensor>,
    pub cross_gate_timestep: Option<Tensor>,
    pub enabled: bool,
}

pub struct TransformerArgsPreprocessor {
    patchify_proj: Linear,
    adaln: AdaLayerNormSingle,
    caption_projection: PixArtAlphaTextProjection,
    inner_dim: usize,
    max_pos: Vec<usize>,
    num_attention_heads: usize,
    use_middle_indices_grid: bool,
    timestep_scale_multiplier: usize,
    double_precision_rope: bool,
    positional_embedding_theta: f64,
    rope_type: LtxRopeType,
}

impl TransformerArgsPreprocessor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        patchify_proj: Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        inner_dim: usize,
        max_pos: Vec<usize>,
        num_attention_heads: usize,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: usize,
        double_precision_rope: bool,
        positional_embedding_theta: f64,
        rope_type: LtxRopeType,
    ) -> Self {
        Self {
            patchify_proj,
            adaln,
            caption_projection,
            inner_dim,
            max_pos,
            num_attention_heads,
            use_middle_indices_grid,
            timestep_scale_multiplier,
            double_precision_rope,
            positional_embedding_theta,
            rope_type,
        }
    }

    fn prepare_timestep(
        &self,
        timestep: &Tensor,
        batch: usize,
        hidden_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let timestep = (timestep * self.timestep_scale_multiplier as f64)?;
        let (timestep, embedded) = self.adaln.forward(&timestep.flatten_all()?, hidden_dtype)?;
        let tokens = timestep.dim(0)? / batch.max(1);
        let tdim = timestep.dim(D::Minus1)?;
        let edim = embedded.dim(D::Minus1)?;
        let timestep = timestep.reshape((batch, tokens, tdim))?;
        let embedded = embedded.reshape((batch, tokens, edim))?;
        Ok((timestep, embedded))
    }

    fn prepare_context(
        &self,
        context: &Tensor,
        x: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let batch = x.dim(0)?;
        let context = self.caption_projection.forward(context)?;
        let context = context.reshape((batch, (), x.dim(D::Minus1)?))?;
        Ok((context, mask.map(|m| m.clone())))
    }

    fn prepare_attention_mask(
        &self,
        mask: Option<&Tensor>,
        x_dtype: DType,
    ) -> Result<Option<Tensor>> {
        let Some(mask) = mask else {
            return Ok(None);
        };
        let is_float = matches!(
            mask.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 | DType::F8E4M3
        );
        if is_float {
            return Ok(Some(mask.clone()));
        }
        let mask = mask.to_dtype(DType::F32)?;
        let mask = (&mask - 1.0)?;
        let mask = mask.to_dtype(x_dtype)?;
        let mask = mask.reshape((mask.dim(0)?, 1, (), mask.dim(D::Minus1)?))?;
        Ok(Some((mask * f32::MAX as f64)?))
    }

    fn prepare_positional_embeddings(
        &self,
        positions: &Tensor,
        x_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        precompute_freqs_cis(
            positions,
            self.inner_dim,
            x_dtype,
            self.positional_embedding_theta,
            &self.max_pos,
            self.use_middle_indices_grid,
            self.num_attention_heads,
            self.rope_type,
            self.double_precision_rope,
        )
    }

    pub fn prepare(&self, modality: &Modality) -> Result<TransformerArgs> {
        let x = self.patchify_proj.forward(&modality.latent)?;
        let (timestep, embedded) =
            self.prepare_timestep(&modality.timesteps, x.dim(0)?, modality.latent.dtype())?;
        let (context, mask) =
            self.prepare_context(&modality.context, &x, modality.context_mask.as_ref())?;
        let mask = self.prepare_attention_mask(mask.as_ref(), modality.latent.dtype())?;
        let pe =
            self.prepare_positional_embeddings(&modality.positions, modality.latent.dtype())?;
        Ok(TransformerArgs {
            x,
            context,
            context_mask: mask,
            timesteps: timestep,
            embedded_timestep: embedded,
            positional_embeddings: pe,
            cross_positional_embeddings: None,
            cross_scale_shift_timestep: None,
            cross_gate_timestep: None,
            enabled: modality.enabled,
        })
    }
}

pub struct MultiModalTransformerArgsPreprocessor {
    simple_preprocessor: TransformerArgsPreprocessor,
    cross_scale_shift_adaln: AdaLayerNormSingle,
    cross_gate_adaln: AdaLayerNormSingle,
    cross_pe_max_pos: usize,
    audio_cross_attention_dim: usize,
    av_ca_timestep_scale_multiplier: usize,
}

impl MultiModalTransformerArgsPreprocessor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        patchify_proj: Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: usize,
        max_pos: Vec<usize>,
        num_attention_heads: usize,
        cross_pe_max_pos: usize,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: usize,
        timestep_scale_multiplier: usize,
        double_precision_rope: bool,
        positional_embedding_theta: f64,
        rope_type: LtxRopeType,
        av_ca_timestep_scale_multiplier: usize,
    ) -> Self {
        let simple_preprocessor = TransformerArgsPreprocessor::new(
            patchify_proj,
            adaln,
            caption_projection,
            inner_dim,
            max_pos,
            num_attention_heads,
            use_middle_indices_grid,
            timestep_scale_multiplier,
            double_precision_rope,
            positional_embedding_theta,
            rope_type,
        );
        Self {
            simple_preprocessor,
            cross_scale_shift_adaln,
            cross_gate_adaln,
            cross_pe_max_pos,
            audio_cross_attention_dim,
            av_ca_timestep_scale_multiplier,
        }
    }

    pub fn prepare(&self, modality: &Modality) -> Result<TransformerArgs> {
        let mut args = self.simple_preprocessor.prepare(modality)?;
        let positions = modality.positions.i((.., .., .., 0))?;
        let cross_pe = precompute_freqs_cis(
            &positions,
            self.audio_cross_attention_dim,
            modality.latent.dtype(),
            self.simple_preprocessor.positional_embedding_theta,
            &[self.cross_pe_max_pos],
            true,
            self.simple_preprocessor.num_attention_heads,
            self.simple_preprocessor.rope_type,
            self.simple_preprocessor.double_precision_rope,
        )?;
        let (scale_shift_timestep, gate_timestep) = self.prepare_cross_attention_timestep(
            &modality.timesteps,
            self.simple_preprocessor.timestep_scale_multiplier,
            args.x.dim(0)?,
            modality.latent.dtype(),
        )?;
        args.cross_positional_embeddings = Some(cross_pe);
        args.cross_scale_shift_timestep = Some(scale_shift_timestep);
        args.cross_gate_timestep = Some(gate_timestep);
        Ok(args)
    }

    fn prepare_cross_attention_timestep(
        &self,
        timestep: &Tensor,
        timestep_scale_multiplier: usize,
        batch_size: usize,
        hidden_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let timestep = (timestep * timestep_scale_multiplier as f64)?;
        let av_ca_factor =
            self.av_ca_timestep_scale_multiplier as f64 / timestep_scale_multiplier as f64;
        let (scale_shift, _) = self
            .cross_scale_shift_adaln
            .forward(&timestep.flatten_all()?, hidden_dtype)?;
        let tokens = scale_shift.dim(0)? / batch_size.max(1);
        let scale_shift = scale_shift.reshape((batch_size, tokens, scale_shift.dim(D::Minus1)?))?;
        let (gate, _) = self
            .cross_gate_adaln
            .forward(&(timestep.flatten_all()? * av_ca_factor)?, hidden_dtype)?;
        let tokens = gate.dim(0)? / batch_size.max(1);
        let gate = gate.reshape((batch_size, tokens, gate.dim(D::Minus1)?))?;
        Ok((scale_shift, gate))
    }
}
