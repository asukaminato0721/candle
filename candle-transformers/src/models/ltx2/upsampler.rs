use candle::{Module, Result, Tensor};
use candle_nn as nn;
use candle_nn::VarBuilder;

use super::video_vae::{Conv3d, PaddingModeType};

#[derive(Debug, Clone)]
struct ResBlock {
    conv1: ConvLayer,
    norm1: nn::GroupNorm,
    conv2: ConvLayer,
    norm2: nn::GroupNorm,
}

impl ResBlock {
    fn new(channels: usize, mid_channels: usize, dims: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = ConvLayer::new(dims, channels, mid_channels, 3, vb.pp("conv1"))?;
        let norm1 = nn::group_norm(32, mid_channels, 1e-6, vb.pp("norm1"))?;
        let conv2 = ConvLayer::new(dims, mid_channels, channels, 3, vb.pp("conv2"))?;
        let norm2 = nn::group_norm(32, channels, 1e-6, vb.pp("norm2"))?;
        Ok(Self {
            conv1,
            norm1,
            conv2,
            norm2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let mut x = self.conv1.forward(x)?;
        x = self.norm1.forward(&x)?;
        x = nn::ops::silu(&x)?;
        x = self.conv2.forward(&x)?;
        x = self.norm2.forward(&x)?;
        nn::ops::silu(&(x + residual)?)
    }
}

#[derive(Debug, Clone)]
struct PixelShuffleND {
    dims: usize,
    upscale: (usize, usize, usize),
}

impl PixelShuffleND {
    fn new(dims: usize, upscale: (usize, usize, usize)) -> Self {
        Self { dims, upscale }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self.dims {
            3 => {
                let (b, c, d, h, w) = x.dims5()?;
                let c0 = c / (self.upscale.0 * self.upscale.1 * self.upscale.2);
                let x = x.reshape(vec![
                    b,
                    c0,
                    self.upscale.0,
                    self.upscale.1,
                    self.upscale.2,
                    d,
                    h,
                    w,
                ])?;
                let x = x.permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?;
                x.reshape((
                    b,
                    c0,
                    d * self.upscale.0,
                    h * self.upscale.1,
                    w * self.upscale.2,
                ))
            }
            2 => {
                let (b, c, h, w) = x.dims4()?;
                let c0 = c / (self.upscale.0 * self.upscale.1);
                let x = x.reshape((b, c0, self.upscale.0, self.upscale.1, h, w))?;
                let x = x.permute((0, 1, 4, 2, 5, 3))?;
                x.reshape((b, c0, h * self.upscale.0, w * self.upscale.1))
            }
            1 => {
                let (b, c, d, h, w) = x.dims5()?;
                let c0 = c / self.upscale.0;
                let x = x.reshape((b, c0, self.upscale.0, d, h, w))?;
                let x = x.permute((0, 1, 3, 2, 4, 5))?;
                x.reshape((b, c0, d * self.upscale.0, h, w))
            }
            _ => candle::bail!("unsupported pixel shuffle dims: {}", self.dims),
        }
    }
}

#[derive(Debug, Clone)]
struct BlurDownsample {
    stride: usize,
    kernel_size: usize,
}

impl BlurDownsample {
    fn new(stride: usize, kernel_size: usize) -> Self {
        Self {
            stride,
            kernel_size,
        }
    }

    fn kernel(&self, device: &candle::Device) -> Result<Tensor> {
        let mut coeffs = Vec::with_capacity(self.kernel_size);
        for k in 0..self.kernel_size {
            let v = binomial(self.kernel_size - 1, k) as f32;
            coeffs.push(v);
        }
        let mut kernel = vec![0f32; self.kernel_size * self.kernel_size];
        let sum: f32 = coeffs.iter().sum();
        for i in 0..self.kernel_size {
            for j in 0..self.kernel_size {
                kernel[i * self.kernel_size + j] = (coeffs[i] * coeffs[j]) / (sum * sum);
            }
        }
        Tensor::from_vec(kernel, (self.kernel_size, self.kernel_size), device)
    }

    fn forward_2d(&self, x: &Tensor) -> Result<Tensor> {
        if self.stride == 1 {
            return Ok(x.clone());
        }
        let c = x.dim(1)?;
        let kernel =
            self.kernel(x.device())?
                .reshape((1, 1, self.kernel_size, self.kernel_size))?;
        let kernel = kernel.repeat((c, 1, 1, 1))?;
        x.conv2d(&kernel, self.kernel_size / 2, self.stride, 1, c)
    }
}

fn binomial(n: usize, k: usize) -> usize {
    let mut res = 1usize;
    for i in 1..=k {
        res = res * (n + 1 - i) / i;
    }
    res
}

#[derive(Debug, Clone)]
struct SpatialRationalResampler {
    scale: f64,
    num: usize,
    den: usize,
    conv: nn::Conv2d,
    pixel_shuffle: PixelShuffleND,
    blur_down: BlurDownsample,
}

impl SpatialRationalResampler {
    fn new(mid_channels: usize, scale: f64, vb: VarBuilder) -> Result<Self> {
        let (num, den) = rational_for_scale(scale)?;
        let conv = nn::conv2d(
            mid_channels,
            (num * num) * mid_channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        Ok(Self {
            scale,
            num,
            den,
            conv,
            pixel_shuffle: PixelShuffleND::new(2, (num, num, 1)),
            blur_down: BlurDownsample::new(den, 5),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, f, _h, _w) = x.dims5()?;
        let x = x.permute((0, 2, 1, 3, 4))?;
        let x = x.reshape((b * f, (), x.dim(3)?, x.dim(4)?))?;
        let x = self.conv.forward(&x)?;
        let x = self.pixel_shuffle.forward(&x)?;
        let x = self.blur_down.forward_2d(&x)?;
        let (bf, c, h, w) = x.dims4()?;
        let x = x.reshape((b, f, c, h, w))?;
        x.permute((0, 2, 1, 3, 4))
    }
}

fn rational_for_scale(scale: f64) -> Result<(usize, usize)> {
    match scale {
        s if (s - 0.75).abs() < 1e-6 => Ok((3, 4)),
        s if (s - 1.5).abs() < 1e-6 => Ok((3, 2)),
        s if (s - 2.0).abs() < 1e-6 => Ok((2, 1)),
        s if (s - 4.0).abs() < 1e-6 => Ok((4, 1)),
        _ => candle::bail!("unsupported rational scale: {scale}"),
    }
}

#[derive(Debug, Clone)]
struct ConvLayer {
    dims: usize,
    conv2d: Option<nn::Conv2d>,
    conv3d: Option<Conv3d>,
}

impl ConvLayer {
    fn new(
        dims: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        match dims {
            2 => Ok(Self {
                dims,
                conv2d: Some(nn::conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    nn::Conv2dConfig {
                        padding: kernel_size / 2,
                        ..Default::default()
                    },
                    vb,
                )?),
                conv3d: None,
            }),
            3 => Ok(Self {
                dims,
                conv2d: None,
                conv3d: Some(Conv3d::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    (1, 1, 1),
                    (1, 1, 1),
                    (1, 1, 1),
                    1,
                    true,
                    PaddingModeType::Zeros,
                    vb,
                )?),
            }),
            _ => candle::bail!("unsupported conv dims: {dims}"),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self.dims {
            2 => self
                .conv2d
                .as_ref()
                .ok_or_else(|| candle::Error::msg("missing conv2d"))?
                .forward(x),
            3 => self
                .conv3d
                .as_ref()
                .ok_or_else(|| candle::Error::msg("missing conv3d"))?
                .forward(x),
            _ => candle::bail!("unsupported conv dims"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LatentUpsampler {
    in_channels: usize,
    mid_channels: usize,
    num_blocks_per_stage: usize,
    dims: usize,
    spatial_upsample: bool,
    temporal_upsample: bool,
    spatial_scale: f64,
    rational_resampler: bool,
    initial_conv: ConvLayer,
    initial_norm: nn::GroupNorm,
    res_blocks: Vec<ResBlock>,
    upsampler: UpsamplerKind,
    post_res_blocks: Vec<ResBlock>,
    final_conv: ConvLayer,
}

#[derive(Debug, Clone)]
enum UpsamplerKind {
    PixelShuffle(PixelShuffleND, ConvLayer),
    PixelShuffle3d(PixelShuffleND, ConvLayer),
    SpatialRational(SpatialRationalResampler),
}

impl LatentUpsampler {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        mid_channels: usize,
        num_blocks_per_stage: usize,
        dims: usize,
        spatial_upsample: bool,
        temporal_upsample: bool,
        spatial_scale: f64,
        rational_resampler: bool,
    ) -> Result<Self> {
        let initial_conv =
            ConvLayer::new(dims, in_channels, mid_channels, 3, vb.pp("initial_conv"))?;
        let initial_norm = nn::group_norm(32, mid_channels, 1e-6, vb.pp("initial_norm"))?;
        let mut res_blocks = Vec::new();
        for i in 0..num_blocks_per_stage {
            res_blocks.push(ResBlock::new(
                mid_channels,
                mid_channels,
                dims,
                vb.pp(format!("res_blocks.{i}")),
            )?);
        }
        let upsampler = if spatial_upsample && temporal_upsample {
            let conv = ConvLayer::new(
                3,
                mid_channels,
                8 * mid_channels,
                3,
                vb.pp("upsampler.conv"),
            )?;
            UpsamplerKind::PixelShuffle3d(PixelShuffleND::new(3, (2, 2, 2)), conv)
        } else if spatial_upsample {
            if rational_resampler {
                UpsamplerKind::SpatialRational(SpatialRationalResampler::new(
                    mid_channels,
                    spatial_scale,
                    vb.pp("upsampler"),
                )?)
            } else {
                let conv = ConvLayer::new(
                    2,
                    mid_channels,
                    4 * mid_channels,
                    3,
                    vb.pp("upsampler.conv"),
                )?;
                UpsamplerKind::PixelShuffle(PixelShuffleND::new(2, (2, 2, 1)), conv)
            }
        } else if temporal_upsample {
            let conv = ConvLayer::new(
                3,
                mid_channels,
                2 * mid_channels,
                3,
                vb.pp("upsampler.conv"),
            )?;
            UpsamplerKind::PixelShuffle3d(PixelShuffleND::new(1, (2, 1, 1)), conv)
        } else {
            candle::bail!("either spatial_upsample or temporal_upsample must be true")
        };
        let mut post_res_blocks = Vec::new();
        for i in 0..num_blocks_per_stage {
            post_res_blocks.push(ResBlock::new(
                mid_channels,
                mid_channels,
                dims,
                vb.pp(format!("post_res_blocks.{i}")),
            )?);
        }
        let final_conv = ConvLayer::new(dims, mid_channels, in_channels, 3, vb.pp("final_conv"))?;
        Ok(Self {
            in_channels,
            mid_channels,
            num_blocks_per_stage,
            dims,
            spatial_upsample,
            temporal_upsample,
            spatial_scale,
            rational_resampler,
            initial_conv,
            initial_norm,
            res_blocks,
            upsampler,
            post_res_blocks,
            final_conv,
        })
    }

    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let (b, _c, f, _h, _w) = latent.dims5()?;
        let mut x = match self.dims {
            2 => {
                let x = latent.permute((0, 2, 1, 3, 4))?;
                x.reshape((b * f, (), x.dim(3)?, x.dim(4)?))?
            }
            _ => latent.clone(),
        };
        x = self.initial_conv.forward(&x)?;
        x = self.initial_norm.forward(&x)?;
        x = nn::ops::silu(&x)?;
        for block in self.res_blocks.iter() {
            x = block.forward(&x)?;
        }

        x = match &self.upsampler {
            UpsamplerKind::PixelShuffle(shuffle, conv) => {
                if self.dims == 3 {
                    let x2 = x.permute((0, 2, 1, 3, 4))?;
                    let x2 = x2.reshape((b * f, (), x2.dim(3)?, x2.dim(4)?))?;
                    let y = conv.forward(&x2)?;
                    let y = shuffle.forward(&y)?;
                    let (bf, c2, h2, w2) = y.dims4()?;
                    let y = y.reshape((b, f, c2, h2, w2))?;
                    y.permute((0, 2, 1, 3, 4))?
                } else {
                    let y = conv.forward(&x)?;
                    shuffle.forward(&y)?
                }
            }
            UpsamplerKind::PixelShuffle3d(shuffle, conv) => {
                let y = conv.forward(&x)?;
                shuffle.forward(&y)?
            }
            UpsamplerKind::SpatialRational(resampler) => resampler.forward(&x)?,
        };
        if self.temporal_upsample && x.rank() == 5 {
            let t = x.dim(2)?;
            if t > 1 {
                x = x.narrow(2, 1, t - 1)?;
            }
        }

        for block in self.post_res_blocks.iter() {
            x = block.forward(&x)?;
        }
        x = self.final_conv.forward(&x)?;

        if self.dims == 2 {
            let (bf, c, h, w) = x.dims4()?;
            let x = x.reshape((b, f, c, h, w))?;
            x.permute((0, 2, 1, 3, 4))
        } else {
            Ok(x)
        }
    }
}
