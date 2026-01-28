use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use serde_json::Value;

use super::adaln::AdaLayerNormSingle;
use super::diffusion::BatchedPerturbationConfig;
use super::rope::LtxRopeType;
use super::text_projection::PixArtAlphaTextProjection;
use super::transformer::{BasicAVTransformerBlock, TransformerConfig};
use super::transformer_args::{
    Modality, MultiModalTransformerArgsPreprocessor, TransformerArgsPreprocessor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtxModelType {
    AudioVideo,
    VideoOnly,
    AudioOnly,
}

impl LtxModelType {
    pub fn is_video_enabled(self) -> bool {
        matches!(self, Self::AudioVideo | Self::VideoOnly)
    }

    pub fn is_audio_enabled(self) -> bool {
        matches!(self, Self::AudioVideo | Self::AudioOnly)
    }
}

#[derive(Debug, Clone)]
pub struct LtxTransformerConfig {
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub cross_attention_dim: usize,
    pub norm_eps: f64,
    pub caption_channels: usize,
    pub positional_embedding_theta: f64,
    pub positional_embedding_max_pos: Vec<usize>,
    pub timestep_scale_multiplier: usize,
    pub use_middle_indices_grid: bool,
    pub audio_num_attention_heads: usize,
    pub audio_attention_head_dim: usize,
    pub audio_in_channels: usize,
    pub audio_out_channels: usize,
    pub audio_cross_attention_dim: usize,
    pub audio_positional_embedding_max_pos: Vec<usize>,
    pub av_ca_timestep_scale_multiplier: usize,
    pub rope_type: LtxRopeType,
    pub double_precision_rope: bool,
}

impl Default for LtxTransformerConfig {
    fn default() -> Self {
        Self {
            num_attention_heads: 32,
            attention_head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            num_layers: 48,
            cross_attention_dim: 4096,
            norm_eps: 1e-6,
            caption_channels: 3840,
            positional_embedding_theta: 10000.0,
            positional_embedding_max_pos: vec![20, 2048, 2048],
            timestep_scale_multiplier: 1000,
            use_middle_indices_grid: true,
            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_in_channels: 128,
            audio_out_channels: 128,
            audio_cross_attention_dim: 2048,
            audio_positional_embedding_max_pos: vec![20],
            av_ca_timestep_scale_multiplier: 1,
            rope_type: LtxRopeType::Interleaved,
            double_precision_rope: false,
        }
    }
}

impl LtxTransformerConfig {
    pub fn from_config_value(config: &Value) -> Self {
        let mut cfg = LtxTransformerConfig::default();
        let tr = config.get("transformer").unwrap_or(config);
        cfg.num_attention_heads = tr
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.num_attention_heads as u64) as usize;
        cfg.attention_head_dim = tr
            .get("attention_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.attention_head_dim as u64) as usize;
        cfg.in_channels = tr
            .get("in_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.in_channels as u64) as usize;
        cfg.out_channels = tr
            .get("out_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.out_channels as u64) as usize;
        cfg.num_layers = tr
            .get("num_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.num_layers as u64) as usize;
        cfg.cross_attention_dim = tr
            .get("cross_attention_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.cross_attention_dim as u64) as usize;
        cfg.norm_eps = tr
            .get("norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.norm_eps);
        cfg.caption_channels = tr
            .get("caption_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.caption_channels as u64) as usize;
        cfg.positional_embedding_theta = tr
            .get("positional_embedding_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.positional_embedding_theta);
        if let Some(v) = tr
            .get("positional_embedding_max_pos")
            .and_then(|v| v.as_array())
        {
            cfg.positional_embedding_max_pos = v
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as usize)
                .collect();
        }
        cfg.timestep_scale_multiplier =
            tr.get("timestep_scale_multiplier")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.timestep_scale_multiplier as u64) as usize;
        cfg.use_middle_indices_grid = tr
            .get("use_middle_indices_grid")
            .and_then(|v| v.as_bool())
            .unwrap_or(cfg.use_middle_indices_grid);
        cfg.audio_num_attention_heads =
            tr.get("audio_num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.audio_num_attention_heads as u64) as usize;
        cfg.audio_attention_head_dim =
            tr.get("audio_attention_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.audio_attention_head_dim as u64) as usize;
        cfg.audio_in_channels = tr
            .get("audio_in_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.audio_in_channels as u64) as usize;
        cfg.audio_out_channels = tr
            .get("audio_out_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.audio_out_channels as u64) as usize;
        cfg.audio_cross_attention_dim =
            tr.get("audio_cross_attention_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.audio_cross_attention_dim as u64) as usize;
        if let Some(v) = tr
            .get("audio_positional_embedding_max_pos")
            .and_then(|v| v.as_array())
        {
            cfg.audio_positional_embedding_max_pos = v
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as usize)
                .collect();
        }
        cfg.av_ca_timestep_scale_multiplier =
            tr.get("av_ca_timestep_scale_multiplier")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.av_ca_timestep_scale_multiplier as u64) as usize;
        cfg.rope_type = tr
            .get("rope_type")
            .and_then(|v| v.as_str())
            .map(LtxRopeType::from_str)
            .unwrap_or(cfg.rope_type);
        cfg.double_precision_rope = tr
            .get("frequencies_precision")
            .and_then(|v| v.as_str())
            .map(|v| v == "float64")
            .unwrap_or(cfg.double_precision_rope);
        cfg
    }
}

#[derive(Debug, Clone)]
struct LayerNormNoAffine {
    eps: f64,
}

impl LayerNormNoAffine {
    fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for LayerNormNoAffine {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = (x - &mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        (x - &mean)?.broadcast_div(&(var + self.eps)?.sqrt()?)
    }
}

pub struct LtxModel {
    model_type: LtxModelType,
    video_pp: Option<TransformerArgsPreprocessor>,
    audio_pp: Option<TransformerArgsPreprocessor>,
    video_pp_mm: Option<MultiModalTransformerArgsPreprocessor>,
    audio_pp_mm: Option<MultiModalTransformerArgsPreprocessor>,
    transformer_blocks: Vec<BasicAVTransformerBlock>,
    scale_shift_table: Option<Tensor>,
    norm_out: Option<LayerNormNoAffine>,
    proj_out: Option<Linear>,
    audio_scale_shift_table: Option<Tensor>,
    audio_norm_out: Option<LayerNormNoAffine>,
    audio_proj_out: Option<Linear>,
    norm_eps: f64,
}

impl LtxModel {
    pub fn from_config_value(
        cfg: &Value,
        vb: VarBuilder,
        model_type: LtxModelType,
    ) -> Result<Self> {
        let cfg = LtxTransformerConfig::from_config_value(cfg);
        Self::new(cfg, vb, model_type)
    }

    pub fn new(
        cfg: LtxTransformerConfig,
        vb: VarBuilder,
        model_type: LtxModelType,
    ) -> Result<Self> {
        let inner_dim = cfg.num_attention_heads * cfg.attention_head_dim;
        let audio_inner_dim = cfg.audio_num_attention_heads * cfg.audio_attention_head_dim;

        let mut video_pp = None;
        let mut audio_pp = None;
        let mut video_pp_mm = None;
        let mut audio_pp_mm = None;

        let mut scale_shift_table = None;
        let mut norm_out = None;
        let mut proj_out = None;

        let mut audio_scale_shift_table = None;
        let mut audio_norm_out = None;
        let mut audio_proj_out = None;

        if model_type.is_video_enabled() {
            let patchify_proj = linear(cfg.in_channels, inner_dim, vb.pp("patchify_proj"))?;
            let adaln_single = AdaLayerNormSingle::new(inner_dim, 6, vb.pp("adaln_single"))?;
            let caption_projection = PixArtAlphaTextProjection::new(
                cfg.caption_channels,
                inner_dim,
                None,
                "gelu_tanh",
                vb.pp("caption_projection"),
            )?;
            scale_shift_table = Some(vb.get((2, inner_dim), "scale_shift_table")?);
            norm_out = Some(LayerNormNoAffine::new(cfg.norm_eps));
            proj_out = Some(linear(inner_dim, cfg.out_channels, vb.pp("proj_out"))?);

            if model_type.is_audio_enabled() {
                let av_ca_video_scale_shift = AdaLayerNormSingle::new(
                    inner_dim,
                    4,
                    vb.pp("av_ca_video_scale_shift_adaln_single"),
                )?;
                let av_ca_a2v_gate =
                    AdaLayerNormSingle::new(inner_dim, 1, vb.pp("av_ca_a2v_gate_adaln_single"))?;
                let cross_pe_max_pos = cfg
                    .positional_embedding_max_pos
                    .get(0)
                    .copied()
                    .unwrap_or(20)
                    .max(
                        cfg.audio_positional_embedding_max_pos
                            .get(0)
                            .copied()
                            .unwrap_or(20),
                    );
                video_pp_mm = Some(MultiModalTransformerArgsPreprocessor::new(
                    patchify_proj,
                    adaln_single,
                    caption_projection,
                    av_ca_video_scale_shift,
                    av_ca_a2v_gate,
                    inner_dim,
                    cfg.positional_embedding_max_pos.clone(),
                    cfg.num_attention_heads,
                    cross_pe_max_pos,
                    cfg.use_middle_indices_grid,
                    cfg.audio_cross_attention_dim,
                    cfg.timestep_scale_multiplier,
                    cfg.double_precision_rope,
                    cfg.positional_embedding_theta,
                    cfg.rope_type,
                    cfg.av_ca_timestep_scale_multiplier,
                ));
            } else {
                video_pp = Some(TransformerArgsPreprocessor::new(
                    patchify_proj,
                    adaln_single,
                    caption_projection,
                    inner_dim,
                    cfg.positional_embedding_max_pos.clone(),
                    cfg.num_attention_heads,
                    cfg.use_middle_indices_grid,
                    cfg.timestep_scale_multiplier,
                    cfg.double_precision_rope,
                    cfg.positional_embedding_theta,
                    cfg.rope_type,
                ));
            }
        }

        if model_type.is_audio_enabled() {
            let audio_patchify_proj = linear(
                cfg.audio_in_channels,
                audio_inner_dim,
                vb.pp("audio_patchify_proj"),
            )?;
            let audio_adaln_single =
                AdaLayerNormSingle::new(audio_inner_dim, 6, vb.pp("audio_adaln_single"))?;
            let audio_caption_projection = PixArtAlphaTextProjection::new(
                cfg.caption_channels,
                audio_inner_dim,
                None,
                "gelu_tanh",
                vb.pp("audio_caption_projection"),
            )?;
            audio_scale_shift_table =
                Some(vb.get((2, audio_inner_dim), "audio_scale_shift_table")?);
            audio_norm_out = Some(LayerNormNoAffine::new(cfg.norm_eps));
            audio_proj_out = Some(linear(
                audio_inner_dim,
                cfg.audio_out_channels,
                vb.pp("audio_proj_out"),
            )?);

            if model_type.is_video_enabled() {
                let av_ca_audio_scale_shift = AdaLayerNormSingle::new(
                    audio_inner_dim,
                    4,
                    vb.pp("av_ca_audio_scale_shift_adaln_single"),
                )?;
                let av_ca_v2a_gate = AdaLayerNormSingle::new(
                    audio_inner_dim,
                    1,
                    vb.pp("av_ca_v2a_gate_adaln_single"),
                )?;
                let cross_pe_max_pos = cfg
                    .positional_embedding_max_pos
                    .get(0)
                    .copied()
                    .unwrap_or(20)
                    .max(
                        cfg.audio_positional_embedding_max_pos
                            .get(0)
                            .copied()
                            .unwrap_or(20),
                    );
                audio_pp_mm = Some(MultiModalTransformerArgsPreprocessor::new(
                    audio_patchify_proj,
                    audio_adaln_single,
                    audio_caption_projection,
                    av_ca_audio_scale_shift,
                    av_ca_v2a_gate,
                    audio_inner_dim,
                    cfg.audio_positional_embedding_max_pos.clone(),
                    cfg.audio_num_attention_heads,
                    cross_pe_max_pos,
                    cfg.use_middle_indices_grid,
                    cfg.audio_cross_attention_dim,
                    cfg.timestep_scale_multiplier,
                    cfg.double_precision_rope,
                    cfg.positional_embedding_theta,
                    cfg.rope_type,
                    cfg.av_ca_timestep_scale_multiplier,
                ));
            } else {
                audio_pp = Some(TransformerArgsPreprocessor::new(
                    audio_patchify_proj,
                    audio_adaln_single,
                    audio_caption_projection,
                    audio_inner_dim,
                    cfg.audio_positional_embedding_max_pos.clone(),
                    cfg.audio_num_attention_heads,
                    cfg.use_middle_indices_grid,
                    cfg.timestep_scale_multiplier,
                    cfg.double_precision_rope,
                    cfg.positional_embedding_theta,
                    cfg.rope_type,
                ));
            }
        }

        let video_block_cfg = if model_type.is_video_enabled() {
            Some(TransformerConfig {
                dim: inner_dim,
                heads: cfg.num_attention_heads,
                d_head: cfg.attention_head_dim,
                context_dim: cfg.cross_attention_dim,
            })
        } else {
            None
        };
        let audio_block_cfg = if model_type.is_audio_enabled() {
            Some(TransformerConfig {
                dim: audio_inner_dim,
                heads: cfg.audio_num_attention_heads,
                d_head: cfg.audio_attention_head_dim,
                context_dim: cfg.audio_cross_attention_dim,
            })
        } else {
            None
        };
        let mut transformer_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for idx in 0..cfg.num_layers {
            transformer_blocks.push(BasicAVTransformerBlock::new(
                idx,
                video_block_cfg.clone(),
                audio_block_cfg.clone(),
                cfg.rope_type,
                cfg.norm_eps,
                vb_blocks.pp(idx),
            )?);
        }

        Ok(Self {
            model_type,
            video_pp,
            audio_pp,
            video_pp_mm,
            audio_pp_mm,
            transformer_blocks,
            scale_shift_table,
            norm_out,
            proj_out,
            audio_scale_shift_table,
            audio_norm_out,
            audio_proj_out,
            norm_eps: cfg.norm_eps,
        })
    }

    fn process_output(
        &self,
        scale_shift_table: &Tensor,
        norm_out: &LayerNormNoAffine,
        proj_out: &Linear,
        x: &Tensor,
        embedded_timestep: &Tensor,
    ) -> Result<Tensor> {
        let table = scale_shift_table.unsqueeze(0)?.unsqueeze(0)?;
        let embedded = embedded_timestep.unsqueeze(2)?;
        let scale_shift = table.broadcast_add(&embedded)?;
        let shift = scale_shift.i((.., .., 0, ..))?;
        let scale = scale_shift.i((.., .., 1, ..))?;
        let x = norm_out.forward(x)?;
        let scale = (scale + 1.0)?;
        let x = (x.broadcast_mul(&scale)? + &shift)?;
        proj_out.forward(&x)
    }

    pub fn forward(
        &self,
        video: Option<Modality>,
        audio: Option<Modality>,
        perturbations: BatchedPerturbationConfig,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if !self.model_type.is_video_enabled() && video.is_some() {
            candle::bail!("video is not enabled for this model")
        }
        if !self.model_type.is_audio_enabled() && audio.is_some() {
            candle::bail!("audio is not enabled for this model")
        }

        let video_args = if let Some(v) = video.as_ref() {
            if let Some(pp) = self.video_pp_mm.as_ref() {
                Some(pp.prepare(v)?)
            } else if let Some(pp) = self.video_pp.as_ref() {
                Some(pp.prepare(v)?)
            } else {
                None
            }
        } else {
            None
        };
        let audio_args = if let Some(a) = audio.as_ref() {
            if let Some(pp) = self.audio_pp_mm.as_ref() {
                Some(pp.prepare(a)?)
            } else if let Some(pp) = self.audio_pp.as_ref() {
                Some(pp.prepare(a)?)
            } else {
                None
            }
        } else {
            None
        };

        let mut v = video_args;
        let mut a = audio_args;
        for block in &self.transformer_blocks {
            let (v_new, a_new) = block.forward(v, a, Some(&perturbations))?;
            v = v_new;
            a = a_new;
        }

        let vx = if let (Some(v), Some(table), Some(norm), Some(proj)) = (
            v.as_ref(),
            self.scale_shift_table.as_ref(),
            self.norm_out.as_ref(),
            self.proj_out.as_ref(),
        ) {
            Some(self.process_output(table, norm, proj, &v.x, &v.embedded_timestep)?)
        } else {
            None
        };
        let ax = if let (Some(a), Some(table), Some(norm), Some(proj)) = (
            a.as_ref(),
            self.audio_scale_shift_table.as_ref(),
            self.audio_norm_out.as_ref(),
            self.audio_proj_out.as_ref(),
        ) {
            Some(self.process_output(table, norm, proj, &a.x, &a.embedded_timestep)?)
        } else {
            None
        };
        Ok((vx, ax))
    }
}

pub struct X0Model {
    velocity_model: LtxModel,
}

impl X0Model {
    pub fn new(velocity_model: LtxModel) -> Self {
        Self { velocity_model }
    }

    pub fn forward(
        &self,
        video: Option<Modality>,
        audio: Option<Modality>,
        perturbations: BatchedPerturbationConfig,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.velocity_model.forward(video, audio, perturbations)
    }
}
