use candle::{Result, Tensor};

use super::attention::Attention;
use super::diffusion::{BatchedPerturbationConfig, PerturbationType};
use super::feed_forward::FeedForward;
use super::rope::LtxRopeType;
use super::transformer_args::TransformerArgs;
use super::utils::rms_norm;
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub dim: usize,
    pub heads: usize,
    pub d_head: usize,
    pub context_dim: usize,
}

pub struct BasicAVTransformerBlock {
    idx: usize,
    attn1: Option<Attention>,
    attn2: Option<Attention>,
    ff: Option<FeedForward>,
    scale_shift_table: Option<Tensor>,

    audio_attn1: Option<Attention>,
    audio_attn2: Option<Attention>,
    audio_ff: Option<FeedForward>,
    audio_scale_shift_table: Option<Tensor>,

    audio_to_video_attn: Option<Attention>,
    video_to_audio_attn: Option<Attention>,
    scale_shift_table_a2v_ca_audio: Option<Tensor>,
    scale_shift_table_a2v_ca_video: Option<Tensor>,

    norm_eps: f64,
}

impl BasicAVTransformerBlock {
    pub fn new(
        idx: usize,
        video: Option<TransformerConfig>,
        audio: Option<TransformerConfig>,
        rope_type: LtxRopeType,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let video_cfg = video.clone();
        let audio_cfg = audio.clone();
        let mut attn1 = None;
        let mut attn2 = None;
        let mut ff = None;
        let mut scale_shift_table = None;

        let mut audio_attn1 = None;
        let mut audio_attn2 = None;
        let mut audio_ff = None;
        let mut audio_scale_shift_table = None;

        let mut audio_to_video_attn = None;
        let mut video_to_audio_attn = None;
        let mut scale_shift_table_a2v_ca_audio = None;
        let mut scale_shift_table_a2v_ca_video = None;

        if let Some(video_cfg) = video_cfg.as_ref() {
            attn1 = Some(Attention::new(
                video_cfg.dim,
                None,
                video_cfg.heads,
                video_cfg.d_head,
                norm_eps,
                rope_type,
                vb.pp("attn1"),
            )?);
            attn2 = Some(Attention::new(
                video_cfg.dim,
                Some(video_cfg.context_dim),
                video_cfg.heads,
                video_cfg.d_head,
                norm_eps,
                rope_type,
                vb.pp("attn2"),
            )?);
            ff = Some(FeedForward::new(
                video_cfg.dim,
                video_cfg.dim,
                4,
                vb.pp("ff"),
            )?);
            scale_shift_table = Some(vb.get((6, video_cfg.dim), "scale_shift_table")?);
        }

        if let Some(audio_cfg) = audio_cfg.as_ref() {
            audio_attn1 = Some(Attention::new(
                audio_cfg.dim,
                None,
                audio_cfg.heads,
                audio_cfg.d_head,
                norm_eps,
                rope_type,
                vb.pp("audio_attn1"),
            )?);
            audio_attn2 = Some(Attention::new(
                audio_cfg.dim,
                Some(audio_cfg.context_dim),
                audio_cfg.heads,
                audio_cfg.d_head,
                norm_eps,
                rope_type,
                vb.pp("audio_attn2"),
            )?);
            audio_ff = Some(FeedForward::new(
                audio_cfg.dim,
                audio_cfg.dim,
                4,
                vb.pp("audio_ff"),
            )?);
            audio_scale_shift_table = Some(vb.get((6, audio_cfg.dim), "audio_scale_shift_table")?);
        }

        if let (Some(video_cfg), Some(audio_cfg)) = (video_cfg.as_ref(), audio_cfg.as_ref()) {
            audio_to_video_attn = Some(Attention::new(
                video_cfg.dim,
                Some(audio_cfg.dim),
                audio_cfg.heads,
                audio_cfg.d_head,
                norm_eps,
                rope_type,
                vb.pp("audio_to_video_attn"),
            )?);
            video_to_audio_attn = Some(Attention::new(
                audio_cfg.dim,
                Some(video_cfg.dim),
                audio_cfg.heads,
                audio_cfg.d_head,
                norm_eps,
                rope_type,
                vb.pp("video_to_audio_attn"),
            )?);
            scale_shift_table_a2v_ca_audio =
                Some(vb.get((5, audio_cfg.dim), "scale_shift_table_a2v_ca_audio")?);
            scale_shift_table_a2v_ca_video =
                Some(vb.get((5, video_cfg.dim), "scale_shift_table_a2v_ca_video")?);
        }

        Ok(Self {
            idx,
            attn1,
            attn2,
            ff,
            scale_shift_table,
            audio_attn1,
            audio_attn2,
            audio_ff,
            audio_scale_shift_table,
            audio_to_video_attn,
            video_to_audio_attn,
            scale_shift_table_a2v_ca_audio,
            scale_shift_table_a2v_ca_video,
            norm_eps,
        })
    }

    fn get_ada_values(
        &self,
        table: &Tensor,
        batch_size: usize,
        timestep: &Tensor,
        start: usize,
        len: usize,
    ) -> Result<Vec<Tensor>> {
        let num_params = table.dim(0)?;
        let embed_dim = timestep.dim(candle::D::Minus1)? / num_params;
        let t = timestep.reshape((batch_size, timestep.dim(1)?, num_params, embed_dim))?;
        let table = table.unsqueeze(0)?.unsqueeze(0)?;
        let ada = table.broadcast_add(&t)?;
        let ada = ada.narrow(2, start, len)?;
        let chunks = ada.chunk(len, 2)?;
        let mut out = Vec::with_capacity(len);
        for c in chunks {
            out.push(c.squeeze(2)?);
        }
        Ok(out)
    }

    fn get_av_ca_ada_values(
        &self,
        table: &Tensor,
        batch_size: usize,
        scale_shift_timestep: &Tensor,
        gate_timestep: &Tensor,
        num_scale_shift_values: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let scale_shift = self.get_ada_values(
            table,
            batch_size,
            scale_shift_timestep,
            0,
            num_scale_shift_values,
        )?;
        let gate =
            self.get_ada_values(table, batch_size, gate_timestep, num_scale_shift_values, 1)?;
        Ok((
            scale_shift[0].clone(),
            scale_shift[1].clone(),
            scale_shift[2].clone(),
            scale_shift[3].clone(),
            gate[0].clone(),
        ))
    }

    pub fn forward(
        &self,
        mut video: Option<TransformerArgs>,
        mut audio: Option<TransformerArgs>,
        perturbations: Option<&BatchedPerturbationConfig>,
    ) -> Result<(Option<TransformerArgs>, Option<TransformerArgs>)> {
        let perturbations = perturbations
            .cloned()
            .unwrap_or_else(BatchedPerturbationConfig::empty);
        if let Some(ref mut v) = video {
            if v.enabled && v.x.elem_count() > 0 {
                let vtable = self.scale_shift_table.as_ref().unwrap();
                let ada = self.get_ada_values(vtable, v.x.dim(0)?, &v.timesteps, 0, 3)?;
                let vshift = &ada[0];
                let vscale = &ada[1];
                let vgate = &ada[2];
                if !perturbations.all_in_batch(PerturbationType::SkipVideoSelfAttn, self.idx) {
                    let norm = rms_norm(&v.x, None, self.norm_eps)?;
                    let vscale = (vscale + 1.0)?;
                    let norm = norm.broadcast_mul(&vscale)?;
                    let norm = (norm + vshift)?;
                    let mask = perturbations.mask_like(
                        PerturbationType::SkipVideoSelfAttn,
                        self.idx,
                        &v.x,
                    )?;
                    let attn = self.attn1.as_ref().unwrap().forward(
                        &norm,
                        None,
                        None,
                        Some((&v.positional_embeddings.0, &v.positional_embeddings.1)),
                        None,
                    )?;
                    let attn = attn.broadcast_mul(vgate)?;
                    let attn = attn.broadcast_mul(&mask)?;
                    v.x = (&v.x + attn)?;
                }
                let attn2 = self.attn2.as_ref().unwrap().forward(
                    &rms_norm(&v.x, None, self.norm_eps)?,
                    Some(&v.context),
                    v.context_mask.as_ref(),
                    None,
                    None,
                )?;
                v.x = (&v.x + attn2)?;
            }
        }

        if let Some(ref mut a) = audio {
            if a.enabled && a.x.elem_count() > 0 {
                let atable = self.audio_scale_shift_table.as_ref().unwrap();
                let ada = self.get_ada_values(atable, a.x.dim(0)?, &a.timesteps, 0, 3)?;
                let ashift = &ada[0];
                let ascale = &ada[1];
                let agate = &ada[2];
                if !perturbations.all_in_batch(PerturbationType::SkipAudioSelfAttn, self.idx) {
                    let norm = rms_norm(&a.x, None, self.norm_eps)?;
                    let ascale = (ascale + 1.0)?;
                    let norm = norm.broadcast_mul(&ascale)?;
                    let norm = (norm + ashift)?;
                    let mask = perturbations.mask_like(
                        PerturbationType::SkipAudioSelfAttn,
                        self.idx,
                        &a.x,
                    )?;
                    let attn = self.audio_attn1.as_ref().unwrap().forward(
                        &norm,
                        None,
                        None,
                        Some((&a.positional_embeddings.0, &a.positional_embeddings.1)),
                        None,
                    )?;
                    let attn = attn.broadcast_mul(agate)?;
                    let attn = attn.broadcast_mul(&mask)?;
                    a.x = (&a.x + attn)?;
                }
                let attn2 = self.audio_attn2.as_ref().unwrap().forward(
                    &rms_norm(&a.x, None, self.norm_eps)?,
                    Some(&a.context),
                    a.context_mask.as_ref(),
                    None,
                    None,
                )?;
                a.x = (&a.x + attn2)?;
            }
        }

        if let (Some(ref mut v), Some(ref mut a)) = (&mut video, &mut audio) {
            if v.enabled && a.enabled && v.x.elem_count() > 0 && a.x.elem_count() > 0 {
                let vx_norm = rms_norm(&v.x, None, self.norm_eps)?;
                let ax_norm = rms_norm(&a.x, None, self.norm_eps)?;

                let (
                    scale_ca_audio_hidden_states_a2v,
                    shift_ca_audio_hidden_states_a2v,
                    scale_ca_audio_hidden_states_v2a,
                    shift_ca_audio_hidden_states_v2a,
                    gate_out_v2a,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio.as_ref().unwrap(),
                    a.x.dim(0)?,
                    a.cross_scale_shift_timestep.as_ref().unwrap(),
                    a.cross_gate_timestep.as_ref().unwrap(),
                    4,
                )?;

                let (
                    scale_ca_video_hidden_states_a2v,
                    shift_ca_video_hidden_states_a2v,
                    scale_ca_video_hidden_states_v2a,
                    shift_ca_video_hidden_states_v2a,
                    gate_out_a2v,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video.as_ref().unwrap(),
                    v.x.dim(0)?,
                    v.cross_scale_shift_timestep.as_ref().unwrap(),
                    v.cross_gate_timestep.as_ref().unwrap(),
                    4,
                )?;

                let scale_v_a2v = (scale_ca_video_hidden_states_a2v + 1.0)?;
                let scale_a_a2v = (scale_ca_audio_hidden_states_a2v + 1.0)?;
                let vx_scaled =
                    (vx_norm.broadcast_mul(&scale_v_a2v)? + &shift_ca_video_hidden_states_a2v)?;
                let ax_scaled =
                    (ax_norm.broadcast_mul(&scale_a_a2v)? + &shift_ca_audio_hidden_states_a2v)?;
                let a2v_mask =
                    perturbations.mask_like(PerturbationType::SkipA2vCrossAttn, self.idx, &v.x)?;
                let attn_a2v = self.audio_to_video_attn.as_ref().unwrap().forward(
                    &vx_scaled,
                    Some(&ax_scaled),
                    None,
                    v.cross_positional_embeddings.as_ref().map(|p| (&p.0, &p.1)),
                    a.cross_positional_embeddings.as_ref().map(|p| (&p.0, &p.1)),
                )?;
                let attn_a2v = attn_a2v.broadcast_mul(&gate_out_a2v)?;
                let attn_a2v = attn_a2v.broadcast_mul(&a2v_mask)?;
                v.x = (&v.x + attn_a2v)?;

                let scale_a_v2a = (scale_ca_audio_hidden_states_v2a + 1.0)?;
                let scale_v_v2a = (scale_ca_video_hidden_states_v2a + 1.0)?;
                let ax_scaled =
                    (ax_norm.broadcast_mul(&scale_a_v2a)? + &shift_ca_audio_hidden_states_v2a)?;
                let vx_scaled =
                    (vx_norm.broadcast_mul(&scale_v_v2a)? + &shift_ca_video_hidden_states_v2a)?;
                let v2a_mask =
                    perturbations.mask_like(PerturbationType::SkipV2aCrossAttn, self.idx, &a.x)?;
                let attn_v2a = self.video_to_audio_attn.as_ref().unwrap().forward(
                    &ax_scaled,
                    Some(&vx_scaled),
                    None,
                    a.cross_positional_embeddings.as_ref().map(|p| (&p.0, &p.1)),
                    v.cross_positional_embeddings.as_ref().map(|p| (&p.0, &p.1)),
                )?;
                let attn_v2a = attn_v2a.broadcast_mul(&gate_out_v2a)?;
                let attn_v2a = attn_v2a.broadcast_mul(&v2a_mask)?;
                a.x = (&a.x + attn_v2a)?;
            }
        }

        if let Some(ref mut v) = video {
            if v.enabled {
                let vtable = self.scale_shift_table.as_ref().unwrap();
                let ada = self.get_ada_values(vtable, v.x.dim(0)?, &v.timesteps, 3, 3)?;
                let vshift = &ada[0];
                let vscale = &ada[1];
                let vgate = &ada[2];
                let vscale = (vscale + 1.0)?;
                let vx_scaled =
                    (rms_norm(&v.x, None, self.norm_eps)?.broadcast_mul(&vscale)? + vshift)?;
                let ff = self.ff.as_ref().unwrap().forward(&vx_scaled)?;
                let ff = ff.broadcast_mul(vgate)?;
                v.x = (&v.x + ff)?;
            }
        }

        if let Some(ref mut a) = audio {
            if a.enabled {
                let atable = self.audio_scale_shift_table.as_ref().unwrap();
                let ada = self.get_ada_values(atable, a.x.dim(0)?, &a.timesteps, 3, 3)?;
                let ashift = &ada[0];
                let ascale = &ada[1];
                let agate = &ada[2];
                let ascale = (ascale + 1.0)?;
                let ax_scaled =
                    (rms_norm(&a.x, None, self.norm_eps)?.broadcast_mul(&ascale)? + ashift)?;
                let ff = self.audio_ff.as_ref().unwrap().forward(&ax_scaled)?;
                let ff = ff.broadcast_mul(agate)?;
                a.x = (&a.x + ff)?;
            }
        }

        Ok((video, audio))
    }
}
