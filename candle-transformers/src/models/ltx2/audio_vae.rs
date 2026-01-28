use candle::{Result, Tensor};
use candle_nn as nn;
use candle_nn::{Module, VarBuilder};

use super::types::AudioLatentShape;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormType {
    Group,
    Pixel,
}

impl NormType {
    pub fn from_str(name: &str) -> Result<Self> {
        match name {
            "group" => Ok(Self::Group),
            "pixel" => Ok(Self::Pixel),
            _ => candle::bail!("unsupported norm_type: {name}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalityAxis {
    None,
    Width,
    Height,
    WidthCompatibility,
}

impl CausalityAxis {
    pub fn from_str(name: &str) -> Result<Self> {
        match name {
            "width" => Ok(Self::Width),
            "height" => Ok(Self::Height),
            "width-compatibility" => Ok(Self::WidthCompatibility),
            _ => Ok(Self::None),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionType {
    Vanilla,
    Linear,
    None,
}

impl AttentionType {
    pub fn from_str(name: &str) -> Result<Self> {
        match name {
            "vanilla" => Ok(Self::Vanilla),
            "linear" => Ok(Self::Linear),
            "none" => Ok(Self::None),
            _ => candle::bail!("unsupported attention type: {name}"),
        }
    }
}

#[derive(Debug, Clone)]
struct PixelNorm {
    dim: usize,
    eps: f64,
}

impl PixelNorm {
    fn new(dim: usize, eps: f64) -> Self {
        Self { dim, eps }
    }
}

impl Module for PixelNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean_sq = x.sqr()?.mean_keepdim(self.dim)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        x.broadcast_div(&rms)
    }
}

fn build_normalization_layer(
    channels: usize,
    norm_type: NormType,
    vb: VarBuilder,
) -> Result<Box<dyn Module>> {
    match norm_type {
        NormType::Group => Ok(Box::new(nn::group_norm(32, channels, 1e-6, vb)?)),
        NormType::Pixel => Ok(Box::new(PixelNorm::new(1, 1e-6))),
    }
}

fn pad_dim_zeros(x: &Tensor, dim: usize, left: usize, right: usize) -> Result<Tensor> {
    x.pad_with_zeros(dim, left, right)
}

fn pad_2d_zeros(
    x: &Tensor,
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
) -> Result<Tensor> {
    let x = pad_dim_zeros(x, 3, pad_left, pad_right)?;
    pad_dim_zeros(&x, 2, pad_top, pad_bottom)
}

struct CausalConv2d {
    conv: nn::Conv2d,
    padding: (usize, usize, usize, usize),
}

impl CausalConv2d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        causality_axis: CausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad_h = (kernel_size - 1) * dilation;
        let pad_w = (kernel_size - 1) * dilation;
        let padding = match causality_axis {
            CausalityAxis::None => (pad_w / 2, pad_w - pad_w / 2, pad_h / 2, pad_h - pad_h / 2),
            CausalityAxis::Width | CausalityAxis::WidthCompatibility => {
                (pad_w, 0, pad_h / 2, pad_h - pad_h / 2)
            }
            CausalityAxis::Height => (pad_w / 2, pad_w - pad_w / 2, pad_h, 0),
        };
        let cfg = nn::Conv2dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = if bias {
            nn::conv2d(in_channels, out_channels, kernel_size, cfg, vb)?
        } else {
            nn::conv2d_no_bias(in_channels, out_channels, kernel_size, cfg, vb)?
        };
        Ok(Self { conv, padding })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = pad_2d_zeros(
            x,
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
        )?;
        self.conv.forward(&x)
    }
}

enum Conv2dLayer {
    Causal(CausalConv2d),
    Regular(nn::Conv2d),
}

impl Conv2dLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Conv2dLayer::Causal(c) => c.forward(x),
            Conv2dLayer::Regular(c) => c.forward(x),
        }
    }
}

fn make_conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
    causality_axis: CausalityAxis,
    vb: VarBuilder,
) -> Result<Conv2dLayer> {
    if causality_axis == CausalityAxis::None {
        let padding = kernel_size / 2;
        let cfg = nn::Conv2dConfig {
            padding,
            stride,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = if bias {
            nn::conv2d(in_channels, out_channels, kernel_size, cfg, vb)?
        } else {
            nn::conv2d_no_bias(in_channels, out_channels, kernel_size, cfg, vb)?
        };
        Ok(Conv2dLayer::Regular(conv))
    } else {
        Ok(Conv2dLayer::Causal(CausalConv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            causality_axis,
            vb,
        )?))
    }
}

struct AttnBlock {
    norm: Box<dyn Module>,
    q: Conv2dLayer,
    k: Conv2dLayer,
    v: Conv2dLayer,
    proj_out: Conv2dLayer,
}

impl AttnBlock {
    fn new(in_channels: usize, norm_type: NormType, vb: VarBuilder) -> Result<Self> {
        let norm = build_normalization_layer(in_channels, norm_type, vb.pp("norm"))?;
        let q = make_conv2d(
            in_channels,
            in_channels,
            1,
            1,
            1,
            1,
            true,
            CausalityAxis::None,
            vb.pp("q"),
        )?;
        let k = make_conv2d(
            in_channels,
            in_channels,
            1,
            1,
            1,
            1,
            true,
            CausalityAxis::None,
            vb.pp("k"),
        )?;
        let v = make_conv2d(
            in_channels,
            in_channels,
            1,
            1,
            1,
            1,
            true,
            CausalityAxis::None,
            vb.pp("v"),
        )?;
        let proj_out = make_conv2d(
            in_channels,
            in_channels,
            1,
            1,
            1,
            1,
            true,
            CausalityAxis::None,
            vb.pp("proj_out"),
        )?;
        Ok(Self {
            norm,
            q,
            k,
            v,
            proj_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        let q = self.q.forward(&h)?;
        let k = self.k.forward(&h)?;
        let v = self.v.forward(&h)?;
        let (b, c, h, w) = q.dims4()?;
        let q = q.reshape((b, c, h * w))?.transpose(1, 2)?; // b, hw, c
        let k = k.reshape((b, c, h * w))?; // b, c, hw
        let mut attn = q.matmul(&k)?;
        let scale = (c as f64).powf(-0.5);
        attn = (attn * scale)?;
        attn = candle_nn::ops::softmax_last_dim(&attn)?; // b, hw, hw
        let v = v.reshape((b, c, h * w))?; // b, c, hw
        let attn = attn.transpose(1, 2)?; // b, hw, hw
        let out = v.matmul(&attn)?; // b, c, hw
        let out = out.reshape((b, c, h, w))?;
        let out = self.proj_out.forward(&out)?;
        x + out
    }
}

enum AttnLayer {
    Block(AttnBlock),
    Identity,
}

impl AttnLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            AttnLayer::Block(b) => b.forward(x),
            AttnLayer::Identity => Ok(x.clone()),
        }
    }
}

fn make_attn(
    in_channels: usize,
    attn_type: AttentionType,
    norm_type: NormType,
    vb: VarBuilder,
) -> Result<AttnLayer> {
    match attn_type {
        AttentionType::Vanilla => Ok(AttnLayer::Block(AttnBlock::new(
            in_channels,
            norm_type,
            vb,
        )?)),
        AttentionType::None => Ok(AttnLayer::Identity),
        AttentionType::Linear => candle::bail!("linear attention not supported"),
    }
}

struct ResnetBlock {
    in_channels: usize,
    out_channels: usize,
    norm1: Box<dyn Module>,
    norm2: Box<dyn Module>,
    conv1: Conv2dLayer,
    conv2: Conv2dLayer,
    shortcut: Option<Conv2dLayer>,
    dropout: nn::Dropout,
}

impl ResnetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        dropout: f64,
        norm_type: NormType,
        causality_axis: CausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = build_normalization_layer(in_channels, norm_type, vb.pp("norm1"))?;
        let norm2 = build_normalization_layer(out_channels, norm_type, vb.pp("norm2"))?;
        let conv1 = make_conv2d(
            in_channels,
            out_channels,
            3,
            1,
            1,
            1,
            true,
            causality_axis,
            vb.pp("conv1"),
        )?;
        let conv2 = make_conv2d(
            out_channels,
            out_channels,
            3,
            1,
            1,
            1,
            true,
            causality_axis,
            vb.pp("conv2"),
        )?;
        let shortcut = if in_channels != out_channels {
            Some(make_conv2d(
                in_channels,
                out_channels,
                1,
                1,
                1,
                1,
                true,
                causality_axis,
                vb.pp("shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            in_channels,
            out_channels,
            norm1,
            norm2,
            conv1,
            conv2,
            shortcut,
            dropout: nn::Dropout::new(dropout as f32),
        })
    }

    fn forward(&self, x: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let mut h = self.norm1.forward(x)?;
        h = nn::ops::silu(&h)?;
        h = self.conv1.forward(&h)?;
        if let Some(temb) = temb {
            let temb = nn::ops::silu(temb)?;
            let temb = temb.unsqueeze(2)?.unsqueeze(3)?;
            h = h.broadcast_add(&temb)?;
        }
        h = self.norm2.forward(&h)?;
        h = nn::ops::silu(&h)?;
        h = self.dropout.forward(&h, false)?;
        h = self.conv2.forward(&h)?;
        let x = if let Some(shortcut) = &self.shortcut {
            shortcut.forward(x)?
        } else {
            x.clone()
        };
        x + h
    }
}

struct Downsample {
    conv: nn::Conv2d,
    causality_axis: CausalityAxis,
}

impl Downsample {
    fn new(in_channels: usize, causality_axis: CausalityAxis, vb: VarBuilder) -> Result<Self> {
        let cfg = nn::Conv2dConfig {
            padding: 0,
            stride: 2,
            ..Default::default()
        };
        let conv = nn::conv2d(in_channels, in_channels, 3, cfg, vb)?;
        Ok(Self {
            conv,
            causality_axis,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (pad_left, pad_right, pad_top, pad_bottom) = match self.causality_axis {
            CausalityAxis::None => (0, 1, 0, 1),
            CausalityAxis::Width => (2, 0, 0, 1),
            CausalityAxis::Height => (0, 1, 2, 0),
            CausalityAxis::WidthCompatibility => (1, 0, 0, 1),
        };
        let x = pad_2d_zeros(x, pad_left, pad_right, pad_top, pad_bottom)?;
        self.conv.forward(&x)
    }
}

struct Upsample {
    conv: Option<Conv2dLayer>,
    causality_axis: CausalityAxis,
}

impl Upsample {
    fn new(in_channels: usize, causality_axis: CausalityAxis, vb: VarBuilder) -> Result<Self> {
        let conv = make_conv2d(
            in_channels,
            in_channels,
            3,
            1,
            1,
            1,
            true,
            causality_axis,
            vb.pp("conv"),
        )?;
        Ok(Self {
            conv: Some(conv),
            causality_axis,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = repeat_interleave_2d(x, 2)?;
        let mut x = if let Some(conv) = &self.conv {
            conv.forward(&x)?
        } else {
            x
        };
        match self.causality_axis {
            CausalityAxis::None => Ok(x),
            CausalityAxis::Height => x.narrow(2, 1, x.dim(2)? - 1),
            CausalityAxis::Width => x.narrow(3, 1, x.dim(3)? - 1),
            CausalityAxis::WidthCompatibility => Ok(x),
        }
    }
}

fn repeat_interleave_2d(x: &Tensor, scale: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let x = x.unsqueeze(4)?; // b c h w 1
    let mut reps = vec![1usize; x.rank()];
    reps[4] = scale;
    let x = x.repeat(reps)?;
    let x = x.reshape((b, c, h, w * scale))?;
    let x = x.unsqueeze(3)?; // b c h 1 w*scale
    let mut reps = vec![1usize; x.rank()];
    reps[3] = scale;
    let x = x.repeat(reps)?;
    x.reshape((b, c, h * scale, w * scale))
}

struct DownStage {
    blocks: Vec<ResnetBlock>,
    attn: Vec<AttnLayer>,
    downsample: Option<Downsample>,
}

struct UpStage {
    blocks: Vec<ResnetBlock>,
    attn: Vec<AttnLayer>,
    upsample: Option<Upsample>,
}

fn build_downsampling_path(
    ch: usize,
    ch_mult: &[usize],
    num_res_blocks: usize,
    resolution: usize,
    dropout: f64,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: &[usize],
    vb: VarBuilder,
) -> Result<(Vec<DownStage>, usize)> {
    let mut down_modules = Vec::new();
    let mut curr_res = resolution;
    let mut block_in = ch;
    for (i_level, mult) in ch_mult.iter().enumerate() {
        let block_out = ch * *mult;
        let mut blocks = Vec::new();
        let mut attn = Vec::new();
        for i in 0..num_res_blocks {
            blocks.push(ResnetBlock::new(
                block_in,
                block_out,
                dropout,
                norm_type,
                causality_axis,
                vb.pp(format!("{i_level}.block.{i}")),
            )?);
            block_in = block_out;
            if attn_resolutions.contains(&curr_res) {
                attn.push(make_attn(
                    block_in,
                    attn_type,
                    norm_type,
                    vb.pp(format!("{i_level}.attn.{i}")),
                )?);
            }
        }
        let downsample = if i_level != ch_mult.len() - 1 {
            curr_res /= 2;
            Some(Downsample::new(
                block_in,
                causality_axis,
                vb.pp(format!("{i_level}.downsample")),
            )?)
        } else {
            None
        };
        down_modules.push(DownStage {
            blocks,
            attn,
            downsample,
        });
    }
    Ok((down_modules, block_in))
}

fn build_upsampling_path(
    ch: usize,
    ch_mult: &[usize],
    num_res_blocks: usize,
    resolution: usize,
    dropout: f64,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: &[usize],
    initial_block_channels: usize,
    vb: VarBuilder,
) -> Result<(Vec<UpStage>, usize)> {
    let mut up_modules: Vec<UpStage> = Vec::new();
    let mut block_in = initial_block_channels;
    let mut curr_res = resolution / (2usize.pow((ch_mult.len() - 1) as u32));

    for (level_idx, mult) in ch_mult.iter().enumerate().rev() {
        let mut blocks = Vec::new();
        let mut attn = Vec::new();
        let block_out = ch * *mult;
        for i in 0..(num_res_blocks + 1) {
            blocks.push(ResnetBlock::new(
                block_in,
                block_out,
                dropout,
                norm_type,
                causality_axis,
                vb.pp(format!("{level_idx}.block.{i}")),
            )?);
            block_in = block_out;
            if attn_resolutions.contains(&curr_res) {
                attn.push(make_attn(
                    block_in,
                    attn_type,
                    norm_type,
                    vb.pp(format!("{level_idx}.attn.{i}")),
                )?);
            }
        }
        let upsample = if level_idx != 0 {
            curr_res *= 2;
            Some(Upsample::new(
                block_in,
                causality_axis,
                vb.pp(format!("{level_idx}.upsample")),
            )?)
        } else {
            None
        };
        up_modules.insert(
            0,
            UpStage {
                blocks,
                attn,
                upsample,
            },
        );
    }
    Ok((up_modules, block_in))
}

struct PerChannelStatistics {
    std_of_means: Tensor,
    mean_of_means: Tensor,
}

impl PerChannelStatistics {
    fn new(latent_channels: usize, vb: VarBuilder) -> Result<Self> {
        let std_of_means = vb.get(latent_channels, "std-of-means")?;
        let mean_of_means = vb.get(latent_channels, "mean-of-means")?;
        Ok(Self {
            std_of_means,
            mean_of_means,
        })
    }

    fn un_normalize(&self, x: &Tensor) -> Result<Tensor> {
        match x.rank() {
            3 => {
                let std = self.std_of_means.reshape((1, 1, ()))?;
                let mean = self.mean_of_means.reshape((1, 1, ()))?;
                (x.broadcast_mul(&std)? + &mean)
            }
            4 => {
                let std = self.std_of_means.reshape((1, (), 1, 1))?;
                let mean = self.mean_of_means.reshape((1, (), 1, 1))?;
                (x.broadcast_mul(&std)? + &mean)
            }
            _ => candle::bail!("unsupported rank for audio normalization"),
        }
    }

    fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        match x.rank() {
            3 => {
                let std = self.std_of_means.reshape((1, 1, ()))?;
                let mean = self.mean_of_means.reshape((1, 1, ()))?;
                (x.broadcast_sub(&mean)?).broadcast_div(&std)
            }
            4 => {
                let std = self.std_of_means.reshape((1, (), 1, 1))?;
                let mean = self.mean_of_means.reshape((1, (), 1, 1))?;
                (x.broadcast_sub(&mean)?).broadcast_div(&std)
            }
            _ => candle::bail!("unsupported rank for audio normalization"),
        }
    }
}

struct AudioPatchifier;

impl AudioPatchifier {
    fn patchify(&self, latents: &Tensor) -> Result<Tensor> {
        let (b, c, t, f) = latents.dims4()?;
        let latents = latents.permute((0, 2, 1, 3))?;
        latents.reshape((b, t, c * f))
    }

    fn unpatchify(&self, latents: &Tensor, output_shape: AudioLatentShape) -> Result<Tensor> {
        let latents = latents.reshape((
            output_shape.batch,
            output_shape.frames,
            output_shape.channels,
            output_shape.mel_bins,
        ))?;
        latents.permute((0, 2, 1, 3))
    }
}

struct MidBlock {
    block1: ResnetBlock,
    attn: AttnLayer,
    block2: ResnetBlock,
}

impl MidBlock {
    fn new(
        channels: usize,
        dropout: f64,
        norm_type: NormType,
        causality_axis: CausalityAxis,
        attn_type: AttentionType,
        add_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let block1 = ResnetBlock::new(
            channels,
            channels,
            dropout,
            norm_type,
            causality_axis,
            vb.pp("block1"),
        )?;
        let attn = if add_attention {
            make_attn(channels, attn_type, norm_type, vb.pp("attn"))?
        } else {
            AttnLayer::Identity
        };
        let block2 = ResnetBlock::new(
            channels,
            channels,
            dropout,
            norm_type,
            causality_axis,
            vb.pp("block2"),
        )?;
        Ok(Self {
            block1,
            attn,
            block2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block1.forward(x, None)?;
        let x = self.attn.forward(&x)?;
        self.block2.forward(&x, None)
    }
}

pub struct AudioEncoder {
    per_channel_statistics: PerChannelStatistics,
    patchifier: AudioPatchifier,
    conv_in: Conv2dLayer,
    down: Vec<DownStage>,
    mid: MidBlock,
    norm_out: Box<dyn Module>,
    conv_out: Conv2dLayer,
    num_resolutions: usize,
    num_res_blocks: usize,
    attn_resolutions: Vec<usize>,
    ch_mult: Vec<usize>,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    z_channels: usize,
    double_z: bool,
    mel_bins: usize,
}

impl AudioEncoder {
    pub fn new(vb: VarBuilder, cfg: AudioVaeConfig) -> Result<Self> {
        let per_channel_statistics =
            PerChannelStatistics::new(cfg.ch, vb.pp("per_channel_statistics"))?;
        let patchifier = AudioPatchifier;
        let conv_in = make_conv2d(
            cfg.in_channels,
            cfg.ch,
            3,
            1,
            1,
            1,
            true,
            cfg.causality_axis,
            vb.pp("conv_in"),
        )?;
        let (down, block_in) = build_downsampling_path(
            cfg.ch,
            &cfg.ch_mult,
            cfg.num_res_blocks,
            cfg.resolution,
            cfg.dropout,
            cfg.norm_type,
            cfg.causality_axis,
            cfg.attn_type,
            &cfg.attn_resolutions,
            vb.pp("down"),
        )?;
        let mid = MidBlock::new(
            block_in,
            cfg.dropout,
            cfg.norm_type,
            cfg.causality_axis,
            cfg.attn_type,
            cfg.mid_block_add_attention,
            vb.pp("mid"),
        )?;
        let norm_out = build_normalization_layer(block_in, cfg.norm_type, vb.pp("norm_out"))?;
        let conv_out = make_conv2d(
            block_in,
            if cfg.double_z {
                2 * cfg.z_channels
            } else {
                cfg.z_channels
            },
            3,
            1,
            1,
            1,
            true,
            cfg.causality_axis,
            vb.pp("conv_out"),
        )?;
        Ok(Self {
            per_channel_statistics,
            patchifier,
            conv_in,
            down,
            mid,
            norm_out,
            conv_out,
            num_resolutions: cfg.ch_mult.len(),
            num_res_blocks: cfg.num_res_blocks,
            attn_resolutions: cfg.attn_resolutions.clone(),
            ch_mult: cfg.ch_mult.clone(),
            norm_type: cfg.norm_type,
            causality_axis: cfg.causality_axis,
            z_channels: cfg.z_channels,
            double_z: cfg.double_z,
            mel_bins: cfg.mel_bins,
        })
    }

    pub fn forward(&self, spectrogram: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(spectrogram)?;
        for stage in self.down.iter() {
            for (idx, block) in stage.blocks.iter().enumerate() {
                h = block.forward(&h, None)?;
                if idx < stage.attn.len() {
                    h = stage.attn[idx].forward(&h)?;
                }
            }
            if let Some(down) = &stage.downsample {
                h = down.forward(&h)?;
            }
        }
        h = self.mid.forward(&h)?;
        h = self.norm_out.forward(&h)?;
        h = nn::ops::silu(&h)?;
        h = self.conv_out.forward(&h)?;
        let means = h.narrow(1, 0, self.z_channels)?;
        let latent_shape = AudioLatentShape {
            batch: means.dim(0)?,
            channels: means.dim(1)?,
            frames: means.dim(2)?,
            mel_bins: means.dim(3)?,
        };
        let patched = self.patchifier.patchify(&means)?;
        let normalized = self.per_channel_statistics.normalize(&patched)?;
        self.patchifier.unpatchify(&normalized, latent_shape)
    }
}

pub struct AudioDecoder {
    per_channel_statistics: PerChannelStatistics,
    patchifier: AudioPatchifier,
    conv_in: Conv2dLayer,
    mid: MidBlock,
    up: Vec<UpStage>,
    norm_out: Box<dyn Module>,
    conv_out: Conv2dLayer,
    out_ch: usize,
    mel_bins: usize,
    causality_axis: CausalityAxis,
}

impl AudioDecoder {
    pub fn new(vb: VarBuilder, cfg: AudioVaeConfig) -> Result<Self> {
        let per_channel_statistics =
            PerChannelStatistics::new(cfg.ch, vb.pp("per_channel_statistics"))?;
        let patchifier = AudioPatchifier;
        let base_block_channels = cfg.ch * cfg.ch_mult[cfg.ch_mult.len() - 1];
        let conv_in = make_conv2d(
            cfg.z_channels,
            base_block_channels,
            3,
            1,
            1,
            1,
            true,
            cfg.causality_axis,
            vb.pp("conv_in"),
        )?;
        let mid = MidBlock::new(
            base_block_channels,
            cfg.dropout,
            cfg.norm_type,
            cfg.causality_axis,
            cfg.attn_type,
            cfg.mid_block_add_attention,
            vb.pp("mid"),
        )?;
        let (up, final_block_channels) = build_upsampling_path(
            cfg.ch,
            &cfg.ch_mult,
            cfg.num_res_blocks,
            cfg.resolution,
            cfg.dropout,
            cfg.norm_type,
            cfg.causality_axis,
            cfg.attn_type,
            &cfg.attn_resolutions,
            base_block_channels,
            vb.pp("up"),
        )?;
        let norm_out =
            build_normalization_layer(final_block_channels, cfg.norm_type, vb.pp("norm_out"))?;
        let conv_out = make_conv2d(
            final_block_channels,
            cfg.out_ch,
            3,
            1,
            1,
            1,
            true,
            cfg.causality_axis,
            vb.pp("conv_out"),
        )?;
        Ok(Self {
            per_channel_statistics,
            patchifier,
            conv_in,
            mid,
            up,
            norm_out,
            conv_out,
            out_ch: cfg.out_ch,
            mel_bins: cfg.mel_bins,
            causality_axis: cfg.causality_axis,
        })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let latent_shape = AudioLatentShape {
            batch: sample.dim(0)?,
            channels: sample.dim(1)?,
            frames: sample.dim(2)?,
            mel_bins: sample.dim(3)?,
        };
        let sample = self.patchifier.patchify(sample)?;
        let sample = self.per_channel_statistics.un_normalize(&sample)?;
        let mut sample = self.patchifier.unpatchify(&sample, latent_shape)?;

        let mut h = self.conv_in.forward(&sample)?;
        h = self.mid.forward(&h)?;
        for stage in self.up.iter().rev() {
            for (idx, block) in stage.blocks.iter().enumerate() {
                h = block.forward(&h, None)?;
                if idx < stage.attn.len() {
                    h = stage.attn[idx].forward(&h)?;
                }
            }
            if let Some(up) = &stage.upsample {
                h = up.forward(&h)?;
            }
        }
        h = self.norm_out.forward(&h)?;
        h = nn::ops::silu(&h)?;
        h = self.conv_out.forward(&h)?;
        Ok(h)
    }
}

#[derive(Debug, Clone)]
pub struct AudioVaeConfig {
    pub ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attn_resolutions: Vec<usize>,
    pub dropout: f64,
    pub in_channels: usize,
    pub resolution: usize,
    pub z_channels: usize,
    pub double_z: bool,
    pub attn_type: AttentionType,
    pub mid_block_add_attention: bool,
    pub norm_type: NormType,
    pub causality_axis: CausalityAxis,
    pub out_ch: usize,
    pub mel_bins: usize,
}

impl AudioVaeConfig {
    pub fn from_config_value(value: &serde_json::Value) -> Result<Self> {
        let cfg = value.get("audio_vae").unwrap_or(value);
        let model_params = cfg.get("model").and_then(|v| v.get("params"));
        let ddconfig = model_params.and_then(|v| v.get("ddconfig")).unwrap_or(cfg);
        let preprocessing = cfg.get("preprocessing").unwrap_or(&serde_json::Value::Null);
        let _stft = preprocessing
            .get("stft")
            .unwrap_or(&serde_json::Value::Null);
        let mel = preprocessing.get("mel").unwrap_or(&serde_json::Value::Null);
        let variables = cfg.get("variables").unwrap_or(&serde_json::Value::Null);
        let mel_bins = ddconfig
            .get("mel_bins")
            .or_else(|| mel.get("n_mel_channels"))
            .or_else(|| variables.get("mel_bins"))
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        Ok(Self {
            ch: ddconfig.get("ch").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
            ch_mult: ddconfig
                .get("ch_mult")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|v| v as usize))
                        .collect()
                })
                .unwrap_or_else(|| vec![1, 2, 4, 8]),
            num_res_blocks: ddconfig
                .get("num_res_blocks")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize,
            attn_resolutions: ddconfig
                .get("attn_resolutions")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|v| v as usize))
                        .collect()
                })
                .unwrap_or_else(|| vec![8, 16, 32]),
            dropout: ddconfig
                .get("dropout")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            in_channels: ddconfig
                .get("in_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize,
            resolution: ddconfig
                .get("resolution")
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as usize,
            z_channels: ddconfig
                .get("z_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize,
            double_z: ddconfig
                .get("double_z")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            attn_type: AttentionType::from_str(
                ddconfig
                    .get("attn_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("vanilla"),
            )?,
            mid_block_add_attention: ddconfig
                .get("mid_block_add_attention")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            norm_type: NormType::from_str(
                ddconfig
                    .get("norm_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("pixel"),
            )?,
            causality_axis: CausalityAxis::from_str(
                ddconfig
                    .get("causality_axis")
                    .and_then(|v| v.as_str())
                    .unwrap_or("height"),
            )?,
            out_ch: ddconfig.get("out_ch").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            mel_bins,
        })
    }
}

pub struct Vocoder {
    conv_pre: nn::Conv1d,
    ups: Vec<nn::ConvTranspose1d>,
    resblocks: Vec<ResBlock1D>,
    conv_post: nn::Conv1d,
    num_kernels: usize,
}

#[derive(Debug, Clone)]
struct ResBlock1D {
    convs1: Vec<nn::Conv1d>,
    convs2: Vec<nn::Conv1d>,
    convs: Vec<nn::Conv1d>,
    use_pair: bool,
}

impl ResBlock1D {
    fn new(
        kernel_size: usize,
        dilations: &[usize],
        channels: usize,
        use_pair: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        if use_pair {
            let mut convs1 = Vec::new();
            let mut convs2 = Vec::new();
            for (i, d) in dilations.iter().enumerate() {
                convs1.push(conv1d_same(
                    channels,
                    kernel_size,
                    *d,
                    vb.pp(format!("convs1.{i}")),
                )?);
                convs2.push(conv1d_same(
                    channels,
                    kernel_size,
                    1,
                    vb.pp(format!("convs2.{i}")),
                )?);
            }
            Ok(Self {
                convs1,
                convs2,
                convs: Vec::new(),
                use_pair: true,
            })
        } else {
            let mut convs = Vec::new();
            for (i, d) in dilations.iter().enumerate() {
                convs.push(conv1d_same(
                    channels,
                    kernel_size,
                    *d,
                    vb.pp(format!("convs.{i}")),
                )?);
            }
            Ok(Self {
                convs1: Vec::new(),
                convs2: Vec::new(),
                convs,
                use_pair: false,
            })
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.use_pair {
            let mut x = x.clone();
            for (conv1, conv2) in self.convs1.iter().zip(self.convs2.iter()) {
                let mut xt = candle_nn::ops::leaky_relu(&x, 0.1)?;
                xt = conv1.forward(&xt)?;
                xt = candle_nn::ops::leaky_relu(&xt, 0.1)?;
                xt = conv2.forward(&xt)?;
                x = (&xt + &x)?;
            }
            Ok(x)
        } else {
            let mut x = x.clone();
            for conv in self.convs.iter() {
                let mut xt = candle_nn::ops::leaky_relu(&x, 0.1)?;
                xt = conv.forward(&xt)?;
                x = (&xt + &x)?;
            }
            Ok(x)
        }
    }
}

fn conv1d_same(
    channels: usize,
    kernel_size: usize,
    dilation: usize,
    vb: VarBuilder,
) -> Result<nn::Conv1d> {
    let padding = dilation * (kernel_size - 1) / 2;
    let cfg = nn::Conv1dConfig {
        padding,
        dilation,
        stride: 1,
        ..Default::default()
    };
    nn::conv1d(channels, channels, kernel_size, cfg, vb)
}

impl Vocoder {
    pub fn new(vb: VarBuilder, cfg: VocoderConfig) -> Result<Self> {
        let in_channels = if cfg.stereo { 128 } else { 64 };
        let conv_pre = nn::conv1d(
            in_channels,
            cfg.upsample_initial_channel,
            7,
            nn::Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_pre"),
        )?;
        let mut ups = Vec::new();
        for (i, (stride, kernel)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_ch = cfg.upsample_initial_channel / (2usize.pow(i as u32));
            let out_ch = cfg.upsample_initial_channel / (2usize.pow((i + 1) as u32));
            let padding = (kernel - stride) / 2;
            let conv = nn::conv_transpose1d(
                in_ch,
                out_ch,
                *kernel,
                nn::ConvTranspose1dConfig {
                    stride: *stride,
                    padding,
                    ..Default::default()
                },
                vb.pp(format!("ups.{i}")),
            )?;
            ups.push(conv);
        }
        let mut resblocks = Vec::new();
        for (i, _up) in ups.iter().enumerate() {
            let ch = cfg.upsample_initial_channel / (2usize.pow((i + 1) as u32));
            for (j, kernel_size) in cfg.resblock_kernel_sizes.iter().enumerate() {
                let dilations = &cfg.resblock_dilation_sizes[j];
                let use_pair = cfg.resblock == "1";
                resblocks.push(ResBlock1D::new(
                    *kernel_size,
                    dilations,
                    ch,
                    use_pair,
                    vb.pp(format!("resblocks.{}.{}", i, j)),
                )?);
            }
        }
        let out_channels = if cfg.stereo { 2 } else { 1 };
        let final_channels =
            cfg.upsample_initial_channel / (2usize.pow(cfg.upsample_rates.len() as u32));
        let conv_post = nn::conv1d(
            final_channels,
            out_channels,
            7,
            nn::Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_post"),
        )?;
        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
            num_kernels: cfg.resblock_kernel_sizes.len(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.transpose(2, 3)?; // b c t f -> b c f t
        if x.rank() == 4 {
            let (b, s, c, t) = x.dims4()?;
            let x2 = x.reshape((b, s * c, t))?;
            x = x2;
        }
        let mut x = self.conv_pre.forward(&x)?;
        for (i, up) in self.ups.iter().enumerate() {
            x = candle_nn::ops::leaky_relu(&x, 0.1)?;
            x = up.forward(&x)?;
            let start = i * self.num_kernels;
            let end = start + self.num_kernels;
            let mut acc: Option<Tensor> = None;
            for idx in start..end {
                let out = self.resblocks[idx].forward(&x)?;
                acc = Some(if let Some(acc) = acc {
                    (&acc + &out)?
                } else {
                    out
                });
            }
            let num = (end - start) as f64;
            let denom = Tensor::full(num, (), x.device())?.to_dtype(x.dtype())?;
            x = acc.expect("resblock accum").broadcast_div(&denom)?;
        }
        x = candle_nn::ops::leaky_relu(&x, 0.1)?;
        x = self.conv_post.forward(&x)?;
        x.tanh()
    }
}

#[derive(Debug, Clone)]
pub struct VocoderConfig {
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_initial_channel: usize,
    pub stereo: bool,
    pub resblock: String,
}

impl VocoderConfig {
    pub fn from_config_value(value: &serde_json::Value) -> Result<Self> {
        let cfg = value.get("vocoder").unwrap_or(value);
        Ok(Self {
            resblock_kernel_sizes: cfg
                .get("resblock_kernel_sizes")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|v| v as usize))
                        .collect()
                })
                .unwrap_or_else(|| vec![3, 7, 11]),
            upsample_rates: cfg
                .get("upsample_rates")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|v| v as usize))
                        .collect()
                })
                .unwrap_or_else(|| vec![6, 5, 2, 2, 2]),
            upsample_kernel_sizes: cfg
                .get("upsample_kernel_sizes")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|v| v as usize))
                        .collect()
                })
                .unwrap_or_else(|| vec![16, 15, 8, 4, 4]),
            resblock_dilation_sizes: cfg
                .get("resblock_dilation_sizes")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|inner| {
                            inner.as_array().map(|vals| {
                                vals.iter()
                                    .filter_map(|v| v.as_u64().map(|v| v as usize))
                                    .collect()
                            })
                        })
                        .collect()
                })
                .unwrap_or_else(|| vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]),
            upsample_initial_channel: cfg
                .get("upsample_initial_channel")
                .and_then(|v| v.as_u64())
                .unwrap_or(1024) as usize,
            stereo: cfg.get("stereo").and_then(|v| v.as_bool()).unwrap_or(true),
            resblock: cfg
                .get("resblock")
                .and_then(|v| v.as_str())
                .unwrap_or("1")
                .to_string(),
        })
    }
}

pub fn decode_audio(latent: &Tensor, decoder: &AudioDecoder, vocoder: &Vocoder) -> Result<Tensor> {
    let decoded = decoder.forward(latent)?;
    let decoded = vocoder.forward(&decoded)?;
    Ok(decoded.squeeze(0)?)
}
