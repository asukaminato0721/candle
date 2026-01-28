use candle::{DType, Result, Tensor};
use candle_nn as nn;
use candle_nn::{Module, VarBuilder};

use super::timestep_embedding::PixArtAlphaCombinedTimestepSizeEmbeddings;
use super::types::SpatioTemporalScaleFactors;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormLayerType {
    GroupNorm,
    PixelNorm,
}

impl NormLayerType {
    pub fn from_str(name: &str) -> Result<Self> {
        match name {
            "group_norm" => Ok(Self::GroupNorm),
            "pixel_norm" => Ok(Self::PixelNorm),
            _ => candle::bail!("unsupported norm_layer: {name}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogVarianceType {
    PerChannel,
    Uniform,
    Constant,
    None,
}

impl LogVarianceType {
    pub fn from_str(name: &str) -> Result<Self> {
        match name {
            "per_channel" => Ok(Self::PerChannel),
            "uniform" => Ok(Self::Uniform),
            "constant" => Ok(Self::Constant),
            "none" => Ok(Self::None),
            _ => candle::bail!("unsupported latent_log_var: {name}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingModeType {
    Zeros,
    Reflect,
    Replicate,
}

impl PaddingModeType {
    pub fn from_str(name: &str) -> Result<Self> {
        match name {
            "zeros" => Ok(Self::Zeros),
            "reflect" => Ok(Self::Reflect),
            "replicate" => Ok(Self::Replicate),
            _ => candle::bail!("unsupported padding_mode: {name}"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TilingConfig;

#[derive(Clone, Debug)]
pub struct BlockSpec {
    pub name: String,
    pub num_layers: usize,
    pub multiplier: Option<usize>,
    pub inject_noise: bool,
    pub attention_head_dim: Option<usize>,
    pub residual: bool,
}

fn parse_blocks(value: &serde_json::Value) -> Result<Vec<BlockSpec>> {
    let mut blocks = Vec::new();
    let Some(arr) = value.as_array() else {
        return Ok(blocks);
    };
    for item in arr.iter() {
        let Some(pair) = item.as_array() else {
            candle::bail!("block spec must be array")
        };
        if pair.len() != 2 {
            candle::bail!("block spec must have 2 elements")
        }
        let name = pair[0]
            .as_str()
            .ok_or_else(|| candle::Error::msg("block name must be string"))?
            .to_string();
        let params = &pair[1];
        let mut spec = BlockSpec {
            name,
            num_layers: 1,
            multiplier: None,
            inject_noise: false,
            attention_head_dim: None,
            residual: false,
        };
        if let Some(n) = params.as_u64() {
            spec.num_layers = n as usize;
        } else if let Some(obj) = params.as_object() {
            if let Some(n) = obj.get("num_layers").and_then(|v| v.as_u64()) {
                spec.num_layers = n as usize;
            }
            if let Some(m) = obj.get("multiplier").and_then(|v| v.as_u64()) {
                spec.multiplier = Some(m as usize);
            }
            if let Some(v) = obj.get("inject_noise").and_then(|v| v.as_bool()) {
                spec.inject_noise = v;
            }
            if let Some(v) = obj.get("attention_head_dim").and_then(|v| v.as_u64()) {
                spec.attention_head_dim = Some(v as usize);
            }
            if let Some(v) = obj.get("residual").and_then(|v| v.as_bool()) {
                spec.residual = v;
            }
        } else {
            candle::bail!("block params must be int or object")
        }
        blocks.push(spec);
    }
    Ok(blocks)
}

#[derive(Clone, Debug)]
pub struct VideoEncoderConfig {
    pub convolution_dimensions: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub encoder_blocks: Vec<BlockSpec>,
    pub patch_size: usize,
    pub norm_layer: NormLayerType,
    pub latent_log_var: LogVarianceType,
    pub encoder_spatial_padding_mode: PaddingModeType,
}

impl VideoEncoderConfig {
    pub fn from_config_value(value: &serde_json::Value) -> Result<Self> {
        let cfg = value.get("vae").unwrap_or(value);
        let convolution_dimensions = cfg.get("dims").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let in_channels = cfg.get("in_channels").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let out_channels = cfg
            .get("latent_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let encoder_blocks = parse_blocks(
            cfg.get("encoder_blocks")
                .unwrap_or(&serde_json::Value::Null),
        )?;
        let patch_size = cfg.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
        let norm_layer = NormLayerType::from_str(
            cfg.get("norm_layer")
                .and_then(|v| v.as_str())
                .unwrap_or("pixel_norm"),
        )?;
        let latent_log_var = LogVarianceType::from_str(
            cfg.get("latent_log_var")
                .and_then(|v| v.as_str())
                .unwrap_or("uniform"),
        )?;
        let encoder_spatial_padding_mode = PaddingModeType::from_str(
            cfg.get("encoder_spatial_padding_mode")
                .and_then(|v| v.as_str())
                .unwrap_or("zeros"),
        )?;
        Ok(Self {
            convolution_dimensions,
            in_channels,
            out_channels,
            encoder_blocks,
            patch_size,
            norm_layer,
            latent_log_var,
            encoder_spatial_padding_mode,
        })
    }
}

#[derive(Clone, Debug)]
pub struct VideoDecoderConfig {
    pub convolution_dimensions: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub decoder_blocks: Vec<BlockSpec>,
    pub patch_size: usize,
    pub norm_layer: NormLayerType,
    pub causal: bool,
    pub timestep_conditioning: bool,
    pub decoder_spatial_padding_mode: PaddingModeType,
}

impl VideoDecoderConfig {
    pub fn from_config_value(value: &serde_json::Value) -> Result<Self> {
        let cfg = value.get("vae").unwrap_or(value);
        let convolution_dimensions = cfg.get("dims").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let in_channels = cfg
            .get("latent_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let out_channels = cfg
            .get("out_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let decoder_blocks = parse_blocks(
            cfg.get("decoder_blocks")
                .unwrap_or(&serde_json::Value::Null),
        )?;
        let patch_size = cfg.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
        let norm_layer = NormLayerType::from_str(
            cfg.get("norm_layer")
                .and_then(|v| v.as_str())
                .unwrap_or("pixel_norm"),
        )?;
        let causal = cfg
            .get("causal_decoder")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let timestep_conditioning = cfg
            .get("timestep_conditioning")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let decoder_spatial_padding_mode = PaddingModeType::from_str(
            cfg.get("decoder_spatial_padding_mode")
                .and_then(|v| v.as_str())
                .unwrap_or("reflect"),
        )?;
        Ok(Self {
            convolution_dimensions,
            in_channels,
            out_channels,
            decoder_blocks,
            patch_size,
            norm_layer,
            causal,
            timestep_conditioning,
            decoder_spatial_padding_mode,
        })
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

#[derive(Debug, Clone)]
pub struct PerChannelStatistics {
    std_of_means: Tensor,
    mean_of_means: Tensor,
}

impl PerChannelStatistics {
    pub fn new(latent_channels: usize, vb: VarBuilder) -> Result<Self> {
        let std_of_means = vb.get(latent_channels, "std-of-means")?;
        let mean_of_means = vb.get(latent_channels, "mean-of-means")?;
        Ok(Self {
            std_of_means,
            mean_of_means,
        })
    }

    pub fn un_normalize(&self, x: &Tensor) -> Result<Tensor> {
        let std = self.std_of_means.reshape((1, (), 1, 1, 1))?;
        let mean = self.mean_of_means.reshape((1, (), 1, 1, 1))?;
        (x.broadcast_mul(&std)? + &mean)
    }

    pub fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let std = self.std_of_means.reshape((1, (), 1, 1, 1))?;
        let mean = self.mean_of_means.reshape((1, (), 1, 1, 1))?;
        (x.broadcast_sub(&mean)?).broadcast_div(&std)
    }
}

fn pad_dim(
    x: &Tensor,
    dim: usize,
    left: usize,
    right: usize,
    mode: PaddingModeType,
) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let size = x.dim(dim)?;
    if matches!(mode, PaddingModeType::Reflect) && (left >= size || right >= size) {
        candle::bail!("reflect padding larger than input size")
    }
    match mode {
        PaddingModeType::Zeros => x.pad_with_zeros(dim, left, right),
        PaddingModeType::Reflect => {
            let mut parts: Vec<Tensor> = Vec::new();
            if left > 0 {
                let slice = x.narrow(dim, 1, left)?.flip(&[dim])?;
                parts.push(slice);
            }
            parts.push(x.clone());
            if right > 0 {
                let start = size - right - 1;
                let slice = x.narrow(dim, start, right)?.flip(&[dim])?;
                parts.push(slice);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, dim)
        }
        PaddingModeType::Replicate => {
            let mut parts: Vec<Tensor> = Vec::new();
            if left > 0 {
                let mut repeats = vec![1usize; x.rank()];
                repeats[dim] = left;
                let slice = x.narrow(dim, 0, 1)?.repeat(repeats)?;
                parts.push(slice);
            }
            parts.push(x.clone());
            if right > 0 {
                let mut repeats = vec![1usize; x.rank()];
                repeats[dim] = right;
                let slice = x.narrow(dim, size - 1, 1)?.repeat(repeats)?;
                parts.push(slice);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, dim)
        }
    }
}

fn pad_spatial_5d(x: &Tensor, pad_h: usize, pad_w: usize, mode: PaddingModeType) -> Result<Tensor> {
    let x = pad_dim(x, 3, pad_h, pad_h, mode)?;
    pad_dim(&x, 4, pad_w, pad_w, mode)
}

fn pad_temporal_5d(
    x: &Tensor,
    pad_left: usize,
    pad_right: usize,
    mode: PaddingModeType,
) -> Result<Tensor> {
    pad_dim(x, 2, pad_left, pad_right, mode)
}

#[derive(Debug, Clone)]
pub(crate) struct Conv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
    padding: (usize, usize, usize),
    padding_mode: PaddingModeType,
}

impl Conv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
        bias: bool,
        padding_mode: PaddingModeType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(
            (
                out_channels,
                in_channels / groups,
                kernel_size,
                kernel_size,
                kernel_size,
            ),
            "weight",
        )?;
        let bias = if bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            dilation,
            groups,
            padding,
            padding_mode,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (pad_t, pad_h, pad_w) = self.padding;
        let mut x = x.clone();
        if pad_t > 0 {
            x = pad_temporal_5d(&x, pad_t, pad_t, self.padding_mode)?;
        }
        if pad_h > 0 || pad_w > 0 {
            x = pad_spatial_5d(&x, pad_h, pad_w, self.padding_mode)?;
        }

        let (b, c_in, t, _h, _w) = x.dims5()?;
        let (c_out, c_in_k, k_t, _k_h, _k_w) = self.weight.dims5()?;
        if c_in != c_in_k * self.groups {
            candle::bail!(
                "conv3d in_channels mismatch: {c_in} vs {c_in_k} * {groups}",
                groups = self.groups
            );
        }
        let (stride_t, stride_h, stride_w) = self.stride;
        let (dilation_t, dilation_h, dilation_w) = self.dilation;
        if stride_h != stride_w || dilation_h != dilation_w {
            candle::bail!("conv3d requires equal spatial stride/dilation")
        }
        let effective_k_t = (k_t - 1) * dilation_t + 1;
        if t < effective_k_t {
            candle::bail!("conv3d temporal kernel larger than input")
        }
        let t_out = (t - effective_k_t) / stride_t + 1;

        let mut outs: Vec<Tensor> = Vec::with_capacity(t_out);
        for t_idx in 0..t_out {
            let mut acc: Option<Tensor> = None;
            for kt in 0..k_t {
                let t_in = t_idx * stride_t + kt * dilation_t;
                let xs = x.narrow(2, t_in, 1)?.squeeze(2)?;
                let w = self.weight.narrow(2, kt, 1)?.squeeze(2)?;
                let ys = xs.conv2d(&w, 0, stride_h, dilation_h, self.groups)?;
                acc = Some(if let Some(acc) = acc {
                    (&acc + &ys)?
                } else {
                    ys
                });
            }
            let mut out_t = acc.expect("conv3d accum missing");
            if let Some(bias) = &self.bias {
                let bias = bias.reshape((1, c_out, 1, 1))?;
                out_t = out_t.broadcast_add(&bias)?;
            }
            outs.push(out_t.unsqueeze(2)?);
        }
        Tensor::cat(&outs, 2)
    }
}

#[derive(Debug, Clone)]
struct CausalConv3d {
    conv: Conv3d,
    time_kernel_size: usize,
    padding_mode: PaddingModeType,
}

impl CausalConv3d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
        bias: bool,
        spatial_padding_mode: PaddingModeType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad_h = kernel_size / 2;
        let pad_w = kernel_size / 2;
        let conv = Conv3d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            (0, pad_h, pad_w),
            dilation,
            groups,
            bias,
            spatial_padding_mode,
            vb,
        )?;
        Ok(Self {
            conv,
            time_kernel_size: kernel_size,
            padding_mode: spatial_padding_mode,
        })
    }

    fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let x = if causal {
            pad_temporal_5d(x, self.time_kernel_size - 1, 0, self.padding_mode)?
        } else {
            let pad = (self.time_kernel_size - 1) / 2;
            pad_temporal_5d(x, pad, pad, self.padding_mode)?
        };
        self.conv.forward(&x)
    }
}

#[derive(Debug, Clone)]
enum ConvNd {
    Conv2d(nn::Conv2d),
    Conv3d(Conv3d),
    CausalConv3d(CausalConv3d),
}

impl ConvNd {
    fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        match self {
            ConvNd::Conv2d(conv) => conv.forward(x),
            ConvNd::Conv3d(conv) => conv.forward(x),
            ConvNd::CausalConv3d(conv) => conv.forward(x, causal),
        }
    }
}

fn make_conv_nd(
    dims: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
    bias: bool,
    causal: bool,
    spatial_padding_mode: PaddingModeType,
    vb: VarBuilder,
) -> Result<ConvNd> {
    match dims {
        2 => {
            let cfg = nn::Conv2dConfig {
                padding: padding.1,
                stride: stride.1,
                dilation: dilation.1,
                groups,
                ..Default::default()
            };
            let conv = if bias {
                nn::conv2d(in_channels, out_channels, kernel_size, cfg, vb)?
            } else {
                nn::conv2d_no_bias(in_channels, out_channels, kernel_size, cfg, vb)?
            };
            Ok(ConvNd::Conv2d(conv))
        }
        3 => {
            if causal {
                Ok(ConvNd::CausalConv3d(CausalConv3d::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    dilation,
                    groups,
                    bias,
                    spatial_padding_mode,
                    vb,
                )?))
            } else {
                Ok(ConvNd::Conv3d(Conv3d::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                    spatial_padding_mode,
                    vb,
                )?))
            }
        }
        _ => candle::bail!("unsupported conv dims: {dims}"),
    }
}

fn make_linear_nd(
    dims: usize,
    in_channels: usize,
    out_channels: usize,
    bias: bool,
    vb: VarBuilder,
) -> Result<ConvNd> {
    match dims {
        2 => {
            let cfg = nn::Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            };
            let conv = if bias {
                nn::conv2d(in_channels, out_channels, 1, cfg, vb)?
            } else {
                nn::conv2d_no_bias(in_channels, out_channels, 1, cfg, vb)?
            };
            Ok(ConvNd::Conv2d(conv))
        }
        3 => Ok(ConvNd::Conv3d(Conv3d::new(
            in_channels,
            out_channels,
            1,
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            1,
            bias,
            PaddingModeType::Zeros,
            vb,
        )?)),
        _ => candle::bail!("unsupported linear dims: {dims}"),
    }
}

fn patchify(x: &Tensor, patch_size_hw: usize, patch_size_t: usize) -> Result<Tensor> {
    if patch_size_hw == 1 && patch_size_t == 1 {
        return Ok(x.clone());
    }
    match x.rank() {
        4 => {
            let (b, c, h, w) = x.dims4()?;
            let h2 = h / patch_size_hw;
            let w2 = w / patch_size_hw;
            let x = x.reshape((b, c, h2, patch_size_hw, w2, patch_size_hw))?;
            let x = x.permute((0, 1, 3, 5, 2, 4))?;
            x.reshape((b, c * patch_size_hw * patch_size_hw, h2, w2))
        }
        5 => {
            let (b, c, f, h, w) = x.dims5()?;
            let f2 = f / patch_size_t;
            let h2 = h / patch_size_hw;
            let w2 = w / patch_size_hw;
            let x = x.reshape(vec![
                b,
                c,
                f2,
                patch_size_t,
                h2,
                patch_size_hw,
                w2,
                patch_size_hw,
            ])?;
            let x = x.permute(vec![0, 1, 3, 5, 7, 2, 4, 6])?;
            x.reshape((
                b,
                c * patch_size_t * patch_size_hw * patch_size_hw,
                f2,
                h2,
                w2,
            ))
        }
        _ => candle::bail!("patchify expects 4d or 5d"),
    }
}

fn unpatchify(x: &Tensor, patch_size_hw: usize, patch_size_t: usize) -> Result<Tensor> {
    if patch_size_hw == 1 && patch_size_t == 1 {
        return Ok(x.clone());
    }
    match x.rank() {
        4 => {
            let (b, c, h, w) = x.dims4()?;
            let c0 = c / (patch_size_hw * patch_size_hw);
            let x = x.reshape((b, c0, patch_size_hw, patch_size_hw, h, w))?;
            let x = x.permute((0, 1, 4, 2, 5, 3))?;
            x.reshape((b, c0, h * patch_size_hw, w * patch_size_hw))
        }
        5 => {
            let (b, c, f, h, w) = x.dims5()?;
            let c0 = c / (patch_size_t * patch_size_hw * patch_size_hw);
            let x = x.reshape(vec![
                b,
                c0,
                patch_size_t,
                patch_size_hw,
                patch_size_hw,
                f,
                h,
                w,
            ])?;
            let x = x.permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?;
            x.reshape((
                b,
                c0,
                f * patch_size_t,
                h * patch_size_hw,
                w * patch_size_hw,
            ))
        }
        _ => candle::bail!("unpatchify expects 4d or 5d"),
    }
}

struct ResnetBlock3D {
    in_channels: usize,
    out_channels: usize,
    norm1: Box<dyn Module>,
    norm2: Box<dyn Module>,
    norm3: Box<dyn Module>,
    conv1: ConvNd,
    conv2: ConvNd,
    conv_shortcut: Option<ConvNd>,
    dropout: nn::Dropout,
    inject_noise: bool,
    per_channel_scale1: Option<Tensor>,
    per_channel_scale2: Option<Tensor>,
    timestep_conditioning: bool,
    scale_shift_table: Option<Tensor>,
}

impl ResnetBlock3D {
    fn new(
        dims: usize,
        in_channels: usize,
        out_channels: usize,
        dropout: f64,
        groups: usize,
        eps: f64,
        norm_layer: NormLayerType,
        inject_noise: bool,
        timestep_conditioning: bool,
        spatial_padding_mode: PaddingModeType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1: Box<dyn Module> = match norm_layer {
            NormLayerType::GroupNorm => {
                Box::new(nn::group_norm(groups, in_channels, eps, vb.pp("norm1"))?)
            }
            NormLayerType::PixelNorm => Box::new(PixelNorm::new(1, 1e-6)),
        };
        let norm2: Box<dyn Module> = match norm_layer {
            NormLayerType::GroupNorm => {
                Box::new(nn::group_norm(groups, out_channels, eps, vb.pp("norm2"))?)
            }
            NormLayerType::PixelNorm => Box::new(PixelNorm::new(1, 1e-6)),
        };
        let conv1 = make_conv_nd(
            dims,
            in_channels,
            out_channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            spatial_padding_mode,
            vb.pp("conv1"),
        )?;
        let conv2 = make_conv_nd(
            dims,
            out_channels,
            out_channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            spatial_padding_mode,
            vb.pp("conv2"),
        )?;
        let conv_shortcut = if in_channels != out_channels {
            Some(make_linear_nd(
                dims,
                in_channels,
                out_channels,
                true,
                vb.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        let norm3: Box<dyn Module> = if in_channels != out_channels {
            Box::new(nn::group_norm(1, in_channels, eps, vb.pp("norm3"))?)
        } else {
            Box::new(candle_nn::ops::Identity)
        };
        let per_channel_scale1 = if inject_noise {
            Some(vb.get((in_channels, 1, 1), "per_channel_scale1")?)
        } else {
            None
        };
        let per_channel_scale2 = if inject_noise {
            Some(vb.get((in_channels, 1, 1), "per_channel_scale2")?)
        } else {
            None
        };
        let scale_shift_table = if timestep_conditioning {
            Some(vb.get((4, in_channels), "scale_shift_table")?)
        } else {
            None
        };
        Ok(Self {
            in_channels,
            out_channels,
            norm1,
            norm2,
            norm3,
            conv1,
            conv2,
            conv_shortcut,
            dropout: nn::Dropout::new(dropout as f32),
            inject_noise,
            per_channel_scale1,
            per_channel_scale2,
            timestep_conditioning,
            scale_shift_table,
        })
    }

    fn feed_spatial_noise(
        &self,
        hidden_states: &Tensor,
        per_channel_scale: &Tensor,
    ) -> Result<Tensor> {
        let (_b, _c, _t, h, w) = hidden_states.dims5()?;
        let spatial_noise = Tensor::randn(0f32, 1f32, (h, w), hidden_states.device())?
            .to_dtype(hidden_states.dtype())?;
        let spatial_noise = spatial_noise.reshape((1, 1, 1, h, w))?;
        let per_channel_scale = per_channel_scale.reshape((1, self.in_channels, 1, 1, 1))?;
        let noise = spatial_noise.broadcast_mul(&per_channel_scale)?;
        hidden_states + noise
    }

    fn forward(
        &self,
        input_tensor: &Tensor,
        causal: bool,
        timestep: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch_size = input_tensor.dim(0)?;
        let mut hidden_states = self.norm1.forward(input_tensor)?;
        if self.timestep_conditioning {
            let timestep = timestep.ok_or_else(|| candle::Error::msg("timestep missing"))?;
            let scale_shift_table = self
                .scale_shift_table
                .as_ref()
                .ok_or_else(|| candle::Error::msg("scale_shift_table missing"))?;
            let t = timestep.reshape((batch_size, 4, self.in_channels, 1, 1, 1))?;
            let table = scale_shift_table.reshape((1, 4, self.in_channels, 1, 1, 1))?;
            let ada = table.broadcast_add(&t)?;
            let shift1 = ada.narrow(1, 0, 1)?.squeeze(1)?;
            let scale1 = ada.narrow(1, 1, 1)?.squeeze(1)?;
            let shift2 = ada.narrow(1, 2, 1)?.squeeze(1)?;
            let scale2 = ada.narrow(1, 3, 1)?.squeeze(1)?;
            hidden_states = hidden_states.broadcast_mul(&(&scale1 + 1.0)?)?;
            hidden_states = hidden_states.broadcast_add(&shift1)?;
            hidden_states = nn::ops::silu(&hidden_states)?;
            hidden_states = self.conv1.forward(&hidden_states, causal)?;
            if self.inject_noise {
                if let Some(scale) = &self.per_channel_scale1 {
                    hidden_states = self.feed_spatial_noise(&hidden_states, scale)?;
                }
            }
            hidden_states = self.norm2.forward(&hidden_states)?;
            hidden_states = hidden_states.broadcast_mul(&(&scale2 + 1.0)?)?;
            hidden_states = hidden_states.broadcast_add(&shift2)?;
            hidden_states = nn::ops::silu(&hidden_states)?;
        } else {
            hidden_states = nn::ops::silu(&hidden_states)?;
            hidden_states = self.conv1.forward(&hidden_states, causal)?;
            if self.inject_noise {
                if let Some(scale) = &self.per_channel_scale1 {
                    hidden_states = self.feed_spatial_noise(&hidden_states, scale)?;
                }
            }
            hidden_states = self.norm2.forward(&hidden_states)?;
            hidden_states = nn::ops::silu(&hidden_states)?;
        }
        hidden_states = self.dropout.forward(&hidden_states, false)?;
        hidden_states = self.conv2.forward(&hidden_states, causal)?;
        if self.inject_noise {
            if let Some(scale) = &self.per_channel_scale2 {
                hidden_states = self.feed_spatial_noise(&hidden_states, scale)?;
            }
        }
        let input_tensor = self.norm3.forward(input_tensor)?;
        let input_tensor = if let Some(conv) = &self.conv_shortcut {
            conv.forward(&input_tensor, causal)?
        } else {
            input_tensor.clone()
        };
        input_tensor + hidden_states
    }
}

struct UNetMidBlock3D {
    res_blocks: Vec<ResnetBlock3D>,
    timestep_conditioning: bool,
    time_embedder: Option<PixArtAlphaCombinedTimestepSizeEmbeddings>,
}

impl UNetMidBlock3D {
    fn new(
        dims: usize,
        in_channels: usize,
        num_layers: usize,
        resnet_eps: f64,
        resnet_groups: usize,
        norm_layer: NormLayerType,
        inject_noise: bool,
        timestep_conditioning: bool,
        spatial_padding_mode: PaddingModeType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let time_embedder = if timestep_conditioning {
            Some(PixArtAlphaCombinedTimestepSizeEmbeddings::new(
                in_channels * 4,
                0,
                vb.pp("time_embedder"),
            )?)
        } else {
            None
        };
        let mut res_blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            res_blocks.push(ResnetBlock3D::new(
                dims,
                in_channels,
                in_channels,
                0.0,
                resnet_groups,
                resnet_eps,
                norm_layer,
                inject_noise,
                timestep_conditioning,
                spatial_padding_mode,
                vb.pp(format!("resnets.{i}")),
            )?);
        }
        Ok(Self {
            res_blocks,
            timestep_conditioning,
            time_embedder,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        causal: bool,
        timestep: Option<&Tensor>,
    ) -> Result<Tensor> {
        let timestep_embed = if self.timestep_conditioning {
            let timestep = timestep.ok_or_else(|| candle::Error::msg("timestep missing"))?;
            let batch = timestep.dim(0)?;
            let embedder = self
                .time_embedder
                .as_ref()
                .ok_or_else(|| candle::Error::msg("time_embedder missing"))?;
            let t = embedder.forward(timestep, hidden_states.dtype())?;
            Some(t.reshape((batch, (), 1, 1, 1))?)
        } else {
            None
        };
        let mut hs = hidden_states.clone();
        for block in self.res_blocks.iter() {
            hs = block.forward(&hs, causal, timestep_embed.as_ref())?;
        }
        Ok(hs)
    }
}

#[derive(Debug, Clone)]
struct SpaceToDepthDownsample {
    stride: (usize, usize, usize),
    group_size: usize,
    conv: ConvNd,
}

impl SpaceToDepthDownsample {
    fn new(
        dims: usize,
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize, usize),
        spatial_padding_mode: PaddingModeType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let group_size = in_channels * stride.0 * stride.1 * stride.2 / out_channels;
        let conv = make_conv_nd(
            dims,
            in_channels,
            out_channels / (stride.0 * stride.1 * stride.2),
            3,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            spatial_padding_mode,
            vb.pp("conv"),
        )?;
        Ok(Self {
            stride,
            group_size,
            conv,
        })
    }

    fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let mut x = x.clone();
        if self.stride.0 == 2 {
            let first = x.narrow(2, 0, 1)?;
            x = Tensor::cat(&[&first, &x], 2)?;
        }
        let (b, c, d, h, w) = x.dims5()?;
        let d2 = d / self.stride.0;
        let h2 = h / self.stride.1;
        let w2 = w / self.stride.2;
        let x_in = x.reshape(vec![
            b,
            c,
            d2,
            self.stride.0,
            h2,
            self.stride.1,
            w2,
            self.stride.2,
        ])?;
        let x_in = x_in.permute(vec![0, 1, 3, 5, 7, 2, 4, 6])?;
        let x_in = x_in.reshape((
            b,
            c * self.stride.0 * self.stride.1 * self.stride.2,
            d2,
            h2,
            w2,
        ))?;
        let out_channels = (c * self.stride.0 * self.stride.1 * self.stride.2) / self.group_size;
        let x_in = x_in.reshape((b, out_channels, self.group_size, d2, h2, w2))?;
        let x_in = x_in.mean(2)?;

        let mut y = self.conv.forward(&x, causal)?;
        let (b2, c2, d2, h2, w2) = y.dims5()?;
        let y = y.reshape(vec![
            b2,
            c2,
            d2,
            self.stride.0,
            h2,
            self.stride.1,
            w2,
            self.stride.2,
        ])?;
        let y = y.permute(vec![0, 1, 3, 5, 7, 2, 4, 6])?;
        let y = y.reshape((
            b2,
            c2 * self.stride.0 * self.stride.1 * self.stride.2,
            d2,
            h2,
            w2,
        ))?;
        y + x_in
    }
}

#[derive(Debug, Clone)]
struct DepthToSpaceUpsample {
    stride: (usize, usize, usize),
    conv: ConvNd,
    residual: bool,
    out_channels_reduction_factor: usize,
}

impl DepthToSpaceUpsample {
    fn new(
        dims: usize,
        in_channels: usize,
        stride: (usize, usize, usize),
        residual: bool,
        out_channels_reduction_factor: usize,
        spatial_padding_mode: PaddingModeType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_channels =
            (stride.0 * stride.1 * stride.2) * in_channels / out_channels_reduction_factor;
        let conv = make_conv_nd(
            dims,
            in_channels,
            out_channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            spatial_padding_mode,
            vb.pp("conv"),
        )?;
        Ok(Self {
            stride,
            conv,
            residual,
            out_channels_reduction_factor,
        })
    }

    fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let mut x_in = None;
        if self.residual {
            let (b, c, d, h, w) = x.dims5()?;
            let c0 = c / (self.stride.0 * self.stride.1 * self.stride.2);
            let x_res = x.reshape(vec![
                b,
                c0,
                self.stride.0,
                self.stride.1,
                self.stride.2,
                d,
                h,
                w,
            ])?;
            let x_res = x_res.permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?;
            let x_res = x_res.reshape((
                b,
                c0,
                d * self.stride.0,
                h * self.stride.1,
                w * self.stride.2,
            ))?;
            let num_repeat = (self.stride.0 * self.stride.1 * self.stride.2)
                / self.out_channels_reduction_factor;
            let mut repeats = vec![1usize; x_res.rank()];
            repeats[1] = num_repeat;
            let x_res = x_res.repeat(repeats)?;
            let x_res = if self.stride.0 == 2 {
                x_res.narrow(2, 1, x_res.dim(2)? - 1)?
            } else {
                x_res
            };
            x_in = Some(x_res);
        }
        let mut y = self.conv.forward(x, causal)?;
        let (b, c, d, h, w) = y.dims5()?;
        let y = y.reshape(vec![
            b,
            c / (self.stride.0 * self.stride.1 * self.stride.2),
            self.stride.0,
            self.stride.1,
            self.stride.2,
            d,
            h,
            w,
        ])?;
        let mut y = y.permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?;
        y = y.reshape((
            b,
            c / (self.stride.0 * self.stride.1 * self.stride.2),
            d * self.stride.0,
            h * self.stride.1,
            w * self.stride.2,
        ))?;
        if self.stride.0 == 2 {
            y = y.narrow(2, 1, y.dim(2)? - 1)?;
        }
        if let Some(x_in) = x_in {
            y + x_in
        } else {
            Ok(y)
        }
    }
}

struct EncoderBlock {
    block: EncoderBlockKind,
}

enum EncoderBlockKind {
    Mid(UNetMidBlock3D),
    Resnet(ResnetBlock3D),
    Conv(ConvNd),
    SpaceToDepth(SpaceToDepthDownsample),
}

impl EncoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.block {
            EncoderBlockKind::Mid(b) => b.forward(x, true, None),
            EncoderBlockKind::Resnet(b) => b.forward(x, true, None),
            EncoderBlockKind::Conv(b) => b.forward(x, true),
            EncoderBlockKind::SpaceToDepth(b) => b.forward(x, true),
        }
    }
}

struct DecoderBlock {
    block: DecoderBlockKind,
}

enum DecoderBlockKind {
    Mid(UNetMidBlock3D),
    Resnet(ResnetBlock3D),
    Upsample(DepthToSpaceUpsample),
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor, causal: bool, timestep: Option<&Tensor>) -> Result<Tensor> {
        match &self.block {
            DecoderBlockKind::Mid(b) => b.forward(x, causal, timestep),
            DecoderBlockKind::Resnet(b) => b.forward(x, causal, timestep),
            DecoderBlockKind::Upsample(b) => b.forward(x, causal),
        }
    }
}

fn make_encoder_block(
    block: &BlockSpec,
    in_channels: usize,
    dims: usize,
    norm_layer: NormLayerType,
    norm_num_groups: usize,
    spatial_padding_mode: PaddingModeType,
    vb: VarBuilder,
) -> Result<(EncoderBlock, usize)> {
    let mut out_channels = in_channels;
    let block_kind = match block.name.as_str() {
        "res_x" => EncoderBlockKind::Mid(UNetMidBlock3D::new(
            dims,
            in_channels,
            block.num_layers,
            1e-6,
            norm_num_groups,
            norm_layer,
            false,
            false,
            spatial_padding_mode,
            vb,
        )?),
        "res_x_y" => {
            let mult = block.multiplier.unwrap_or(2);
            out_channels = in_channels * mult;
            EncoderBlockKind::Resnet(ResnetBlock3D::new(
                dims,
                in_channels,
                out_channels,
                0.0,
                norm_num_groups,
                1e-6,
                norm_layer,
                false,
                false,
                spatial_padding_mode,
                vb,
            )?)
        }
        "compress_time" => EncoderBlockKind::Conv(make_conv_nd(
            dims,
            in_channels,
            out_channels,
            3,
            (2, 1, 1),
            (0, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            spatial_padding_mode,
            vb,
        )?),
        "compress_space" => EncoderBlockKind::Conv(make_conv_nd(
            dims,
            in_channels,
            out_channels,
            3,
            (1, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            spatial_padding_mode,
            vb,
        )?),
        "compress_all" => EncoderBlockKind::Conv(make_conv_nd(
            dims,
            in_channels,
            out_channels,
            3,
            (2, 2, 2),
            (0, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            spatial_padding_mode,
            vb,
        )?),
        "compress_all_x_y" => {
            let mult = block.multiplier.unwrap_or(2);
            out_channels = in_channels * mult;
            EncoderBlockKind::Conv(make_conv_nd(
                dims,
                in_channels,
                out_channels,
                3,
                (2, 2, 2),
                (0, 1, 1),
                (1, 1, 1),
                1,
                true,
                true,
                spatial_padding_mode,
                vb,
            )?)
        }
        "compress_all_res" => {
            let mult = block.multiplier.unwrap_or(2);
            out_channels = in_channels * mult;
            EncoderBlockKind::SpaceToDepth(SpaceToDepthDownsample::new(
                dims,
                in_channels,
                out_channels,
                (2, 2, 2),
                spatial_padding_mode,
                vb,
            )?)
        }
        "compress_space_res" => {
            let mult = block.multiplier.unwrap_or(2);
            out_channels = in_channels * mult;
            EncoderBlockKind::SpaceToDepth(SpaceToDepthDownsample::new(
                dims,
                in_channels,
                out_channels,
                (1, 2, 2),
                spatial_padding_mode,
                vb,
            )?)
        }
        "compress_time_res" => {
            let mult = block.multiplier.unwrap_or(2);
            out_channels = in_channels * mult;
            EncoderBlockKind::SpaceToDepth(SpaceToDepthDownsample::new(
                dims,
                in_channels,
                out_channels,
                (2, 1, 1),
                spatial_padding_mode,
                vb,
            )?)
        }
        _ => candle::bail!("unknown encoder block: {}", block.name),
    };
    Ok((EncoderBlock { block: block_kind }, out_channels))
}

fn make_decoder_block(
    block: &BlockSpec,
    in_channels: usize,
    dims: usize,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: usize,
    spatial_padding_mode: PaddingModeType,
    vb: VarBuilder,
) -> Result<(DecoderBlock, usize)> {
    let mut out_channels = in_channels;
    let block_kind = match block.name.as_str() {
        "res_x" => DecoderBlockKind::Mid(UNetMidBlock3D::new(
            dims,
            in_channels,
            block.num_layers,
            1e-6,
            norm_num_groups,
            norm_layer,
            block.inject_noise,
            timestep_conditioning,
            spatial_padding_mode,
            vb,
        )?),
        "attn_res_x" => DecoderBlockKind::Mid(UNetMidBlock3D::new(
            dims,
            in_channels,
            block.num_layers,
            1e-6,
            norm_num_groups,
            norm_layer,
            block.inject_noise,
            timestep_conditioning,
            spatial_padding_mode,
            vb,
        )?),
        "res_x_y" => {
            let mult = block.multiplier.unwrap_or(2);
            out_channels = in_channels / mult;
            DecoderBlockKind::Resnet(ResnetBlock3D::new(
                dims,
                in_channels,
                out_channels,
                0.0,
                norm_num_groups,
                1e-6,
                norm_layer,
                block.inject_noise,
                false,
                spatial_padding_mode,
                vb,
            )?)
        }
        "compress_time" => DecoderBlockKind::Upsample(DepthToSpaceUpsample::new(
            dims,
            in_channels,
            (2, 1, 1),
            false,
            1,
            spatial_padding_mode,
            vb,
        )?),
        "compress_space" => DecoderBlockKind::Upsample(DepthToSpaceUpsample::new(
            dims,
            in_channels,
            (1, 2, 2),
            false,
            1,
            spatial_padding_mode,
            vb,
        )?),
        "compress_all" => {
            let mult = block.multiplier.unwrap_or(1);
            out_channels = in_channels / mult;
            DecoderBlockKind::Upsample(DepthToSpaceUpsample::new(
                dims,
                in_channels,
                (2, 2, 2),
                block.residual,
                mult,
                spatial_padding_mode,
                vb,
            )?)
        }
        _ => candle::bail!("unknown decoder block: {}", block.name),
    };
    Ok((DecoderBlock { block: block_kind }, out_channels))
}

pub struct VideoEncoder {
    patch_size: usize,
    latent_channels: usize,
    latent_log_var: LogVarianceType,
    conv_in: ConvNd,
    down_blocks: Vec<EncoderBlock>,
    conv_norm_out: Box<dyn Module>,
    conv_out: ConvNd,
    per_channel_statistics: PerChannelStatistics,
}

impl VideoEncoder {
    pub fn new(vb: VarBuilder, cfg: VideoEncoderConfig) -> Result<Self> {
        if cfg.convolution_dimensions != 3 {
            candle::bail!("only dims=3 supported for video VAE");
        }
        let in_channels = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        let mut feature_channels = cfg.out_channels;
        let conv_in = make_conv_nd(
            cfg.convolution_dimensions,
            in_channels,
            feature_channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            cfg.encoder_spatial_padding_mode,
            vb.pp("conv_in"),
        )?;
        let mut down_blocks = Vec::new();
        let vb_blocks = vb.pp("down_blocks");
        for (idx, block) in cfg.encoder_blocks.iter().enumerate() {
            let (block, out_ch) = make_encoder_block(
                block,
                feature_channels,
                cfg.convolution_dimensions,
                cfg.norm_layer,
                32,
                cfg.encoder_spatial_padding_mode,
                vb_blocks.pp(idx),
            )?;
            feature_channels = out_ch;
            down_blocks.push(block);
        }
        let conv_norm_out: Box<dyn Module> = match cfg.norm_layer {
            NormLayerType::GroupNorm => Box::new(nn::group_norm(
                32,
                feature_channels,
                1e-6,
                vb.pp("conv_norm_out"),
            )?),
            NormLayerType::PixelNorm => Box::new(PixelNorm::new(1, 1e-6)),
        };
        let conv_out_channels = match cfg.latent_log_var {
            LogVarianceType::PerChannel => cfg.out_channels * 2,
            LogVarianceType::Uniform | LogVarianceType::Constant => cfg.out_channels + 1,
            LogVarianceType::None => cfg.out_channels,
        };
        let conv_out = make_conv_nd(
            cfg.convolution_dimensions,
            feature_channels,
            conv_out_channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            cfg.encoder_spatial_padding_mode,
            vb.pp("conv_out"),
        )?;
        let per_channel_statistics =
            PerChannelStatistics::new(cfg.out_channels, vb.pp("per_channel_statistics"))?;
        Ok(Self {
            patch_size: cfg.patch_size,
            latent_channels: cfg.out_channels,
            latent_log_var: cfg.latent_log_var,
            conv_in,
            down_blocks,
            conv_norm_out,
            conv_out,
            per_channel_statistics,
        })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let frames_count = sample.dim(2)?;
        if (frames_count.saturating_sub(1)) % 8 != 0 {
            candle::bail!("invalid number of frames, expected 1 + 8 * k")
        }
        let sample = patchify(sample, self.patch_size, 1)?;
        let mut sample = self.conv_in.forward(&sample, true)?;
        for block in self.down_blocks.iter() {
            sample = block.forward(&sample)?;
        }
        sample = self.conv_norm_out.forward(&sample)?;
        sample = nn::ops::silu(&sample)?;
        sample = self.conv_out.forward(&sample, true)?;

        let sample = match self.latent_log_var {
            LogVarianceType::Uniform => {
                let channels = sample.dim(1)?;
                if channels < 2 {
                    candle::bail!("invalid channel count for UNIFORM logvar")
                }
                let means = sample.narrow(1, 0, channels - 1)?;
                let logvar = sample.narrow(1, channels - 1, 1)?;
                let repeat = means.dim(1)?;
                let logvar = logvar.repeat((1, repeat, 1, 1, 1))?;
                Tensor::cat(&[&means, &logvar], 1)?
            }
            LogVarianceType::Constant => {
                let channels = sample.dim(1)?;
                let means = sample.narrow(1, 0, channels - 1)?;
                let approx_ln_0 = -30.0f32;
                let logvar = Tensor::full(approx_ln_0, means.shape(), means.device())?
                    .to_dtype(means.dtype())?;
                Tensor::cat(&[&means, &logvar], 1)?
            }
            _ => sample,
        };
        let means = sample.narrow(1, 0, self.latent_channels)?;
        self.per_channel_statistics.normalize(&means)
    }
}

pub struct VideoDecoder {
    patch_size: usize,
    causal: bool,
    timestep_conditioning: bool,
    conv_in: ConvNd,
    up_blocks: Vec<DecoderBlock>,
    conv_norm_out: Box<dyn Module>,
    conv_out: ConvNd,
    per_channel_statistics: PerChannelStatistics,
    decode_noise_scale: f64,
    decode_timestep: f64,
    timestep_scale_multiplier: Option<Tensor>,
    last_time_embedder: Option<PixArtAlphaCombinedTimestepSizeEmbeddings>,
    last_scale_shift_table: Option<Tensor>,
    pub video_downscale_factors: SpatioTemporalScaleFactors,
}

impl VideoDecoder {
    pub fn new(vb: VarBuilder, cfg: VideoDecoderConfig) -> Result<Self> {
        if cfg.convolution_dimensions != 3 {
            candle::bail!("only dims=3 supported for video VAE");
        }
        let mut feature_channels = cfg.in_channels;
        for block in cfg.decoder_blocks.iter().rev() {
            if block.name == "res_x_y" {
                let mult = block.multiplier.unwrap_or(2);
                feature_channels *= mult;
            }
            if block.name == "compress_all" {
                let mult = block.multiplier.unwrap_or(1);
                feature_channels *= mult;
            }
        }
        let conv_in = make_conv_nd(
            cfg.convolution_dimensions,
            cfg.in_channels,
            feature_channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            cfg.decoder_spatial_padding_mode,
            vb.pp("conv_in"),
        )?;
        let mut up_blocks = Vec::new();
        let vb_blocks = vb.pp("up_blocks");
        for (idx, block) in cfg.decoder_blocks.iter().rev().enumerate() {
            let (block, out_ch) = make_decoder_block(
                block,
                feature_channels,
                cfg.convolution_dimensions,
                cfg.norm_layer,
                cfg.timestep_conditioning,
                32,
                cfg.decoder_spatial_padding_mode,
                vb_blocks.pp(idx),
            )?;
            feature_channels = out_ch;
            up_blocks.push(block);
        }
        let conv_norm_out: Box<dyn Module> = match cfg.norm_layer {
            NormLayerType::GroupNorm => Box::new(nn::group_norm(
                32,
                feature_channels,
                1e-6,
                vb.pp("conv_norm_out"),
            )?),
            NormLayerType::PixelNorm => Box::new(PixelNorm::new(1, 1e-6)),
        };
        let out_channels = cfg.out_channels * cfg.patch_size * cfg.patch_size;
        let conv_out = make_conv_nd(
            cfg.convolution_dimensions,
            feature_channels,
            out_channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            true,
            cfg.decoder_spatial_padding_mode,
            vb.pp("conv_out"),
        )?;
        let per_channel_statistics =
            PerChannelStatistics::new(cfg.in_channels, vb.pp("per_channel_statistics"))?;
        let timestep_scale_multiplier = if cfg.timestep_conditioning {
            Some(vb.get((), "timestep_scale_multiplier")?)
        } else {
            None
        };
        let last_time_embedder = if cfg.timestep_conditioning {
            Some(PixArtAlphaCombinedTimestepSizeEmbeddings::new(
                feature_channels * 2,
                0,
                vb.pp("last_time_embedder"),
            )?)
        } else {
            None
        };
        let last_scale_shift_table = if cfg.timestep_conditioning {
            Some(vb.get((2, feature_channels), "last_scale_shift_table")?)
        } else {
            None
        };
        Ok(Self {
            patch_size: cfg.patch_size,
            causal: cfg.causal,
            timestep_conditioning: cfg.timestep_conditioning,
            conv_in,
            up_blocks,
            conv_norm_out,
            conv_out,
            per_channel_statistics,
            decode_noise_scale: 0.025,
            decode_timestep: 0.05,
            timestep_scale_multiplier,
            last_time_embedder,
            last_scale_shift_table,
            video_downscale_factors: SpatioTemporalScaleFactors::default(),
        })
    }

    pub fn forward(&self, sample: &Tensor, timestep: Option<&Tensor>) -> Result<Tensor> {
        let batch_size = sample.dim(0)?;
        let mut sample = sample.clone();
        if self.timestep_conditioning {
            let noise = Tensor::randn(0f32, 1f32, sample.shape(), sample.device())?
                .to_dtype(sample.dtype())?;
            let scale = Tensor::full(self.decode_noise_scale as f32, (), sample.device())?
                .to_dtype(sample.dtype())?;
            let noise = noise.broadcast_mul(&scale)?;
            sample = (noise + ((1.0 - self.decode_noise_scale) * &sample)?)?;
        }
        sample = self.per_channel_statistics.un_normalize(&sample)?;
        let timestep = if self.timestep_conditioning {
            match timestep {
                Some(t) => t.clone(),
                None => Tensor::full(self.decode_timestep as f32, (batch_size,), sample.device())?
                    .to_dtype(sample.dtype())?,
            }
        } else {
            Tensor::zeros((batch_size,), sample.dtype(), sample.device())?
        };
        let scaled_timestep = if self.timestep_conditioning {
            let scale = self
                .timestep_scale_multiplier
                .as_ref()
                .ok_or_else(|| candle::Error::msg("timestep_scale_multiplier missing"))?;
            timestep.broadcast_mul(scale)?
        } else {
            timestep
        };

        let mut sample = self.conv_in.forward(&sample, self.causal)?;
        for block in self.up_blocks.iter() {
            sample = block.forward(&sample, self.causal, Some(&scaled_timestep))?;
        }
        sample = self.conv_norm_out.forward(&sample)?;
        if self.timestep_conditioning {
            let embedder = self
                .last_time_embedder
                .as_ref()
                .ok_or_else(|| candle::Error::msg("last_time_embedder missing"))?;
            let embedded = embedder.forward(&scaled_timestep.flatten_all()?, sample.dtype())?;
            let embedded = embedded.reshape((batch_size, embedded.dim(1)?, 1, 1, 1))?;
            let table = self
                .last_scale_shift_table
                .as_ref()
                .ok_or_else(|| candle::Error::msg("last_scale_shift_table missing"))?;
            let feat = table.dim(1)?;
            let table = table.reshape(vec![1, 2, feat, 1, 1, 1])?;
            let embedded = embedded.reshape(vec![batch_size, 2, feat, 1, 1, 1])?;
            let ada = table.broadcast_add(&embedded)?;
            let shift = ada.narrow(1, 0, 1)?.squeeze(1)?;
            let scale = ada.narrow(1, 1, 1)?.squeeze(1)?;
            sample = sample.broadcast_mul(&(&scale + 1.0)?)?;
            sample = sample.broadcast_add(&shift)?;
        }
        sample = nn::ops::silu(&sample)?;
        sample = self.conv_out.forward(&sample, self.causal)?;
        unpatchify(&sample, self.patch_size, 1)
    }
}

pub fn decode_video(
    latent: &Tensor,
    decoder: &VideoDecoder,
    _tiling: Option<&TilingConfig>,
) -> Result<Tensor> {
    let decoded = decoder.forward(latent, None)?;
    let decoded = ((decoded + 1.0)? / 2.0)?;
    let decoded = decoded.clamp(0.0, 1.0)?;
    let decoded = (decoded * 255.0)?;
    let decoded = decoded.to_dtype(DType::U8)?;
    let decoded = decoded.transpose(1, 2)?; // B, F, C, H, W
    let decoded = decoded.permute((0, 1, 3, 4, 2))?; // B, F, H, W, C
    Ok(decoded)
}

pub fn upsample_latent(
    latent: &Tensor,
    video_encoder: &VideoEncoder,
    upsampler: &super::upsampler::LatentUpsampler,
) -> Result<Tensor> {
    let latent = video_encoder.per_channel_statistics.un_normalize(latent)?;
    let latent = upsampler.forward(&latent)?;
    video_encoder.per_channel_statistics.normalize(&latent)
}
