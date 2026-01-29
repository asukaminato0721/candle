use candle::{DType, Result, Tensor};
use candle_nn::ops;
use candle_nn::{
    conv2d, conv2d_no_bias, conv_transpose2d_no_bias, group_norm, Conv2d, Conv2dConfig,
    ConvTranspose2d, ConvTranspose2dConfig, GroupNorm, Module, VarBuilder,
};

use crate::ops::{apply_color_change, apply_rgb_change, GridChangeApplier};

#[derive(Clone)]
pub struct InstanceNorm2d {
    gn: GroupNorm,
}

impl InstanceNorm2d {
    pub fn new(num_channels: usize, vb: VarBuilder) -> Result<Self> {
        let gn = group_norm(num_channels, num_channels, 1e-5, vb)?;
        Ok(Self { gn })
    }
}

impl Module for InstanceNorm2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.gn.forward(xs)
    }
}

#[derive(Clone, Copy)]
pub enum Act {
    Relu,
    LeakyRelu(f64),
    Sigmoid,
    Tanh,
}

impl Act {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Relu => xs.relu(),
            Self::LeakyRelu(slope) => ops::leaky_relu(xs, *slope),
            Self::Sigmoid => ops::sigmoid(xs),
            Self::Tanh => xs.tanh(),
        }
    }
}

fn conv2d_cfg(_kernel: usize, stride: usize, padding: usize, groups: usize) -> Conv2dConfig {
    Conv2dConfig {
        padding,
        stride,
        dilation: 1,
        groups,
        cudnn_fwd_algo: None,
    }
}

fn conv_transpose2d_cfg(
    _kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> ConvTranspose2dConfig {
    ConvTranspose2dConfig {
        padding,
        output_padding: 0,
        stride,
        dilation: 1,
        groups,
    }
}

fn conv2d_block(
    vb: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    act: Act,
) -> Result<ConvBlock> {
    let conv = conv2d_no_bias(
        in_ch,
        out_ch,
        kernel,
        conv2d_cfg(kernel, stride, padding, 1),
        vb.pp("0"),
    )?;
    let norm = InstanceNorm2d::new(out_ch, vb.pp("1"))?;
    Ok(ConvBlock { conv, norm, act })
}

fn conv2d_bias(
    vb: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Conv2d> {
    conv2d(
        in_ch,
        out_ch,
        kernel,
        conv2d_cfg(kernel, stride, padding, 1),
        vb,
    )
}

fn conv2d_no_bias_layer(
    vb: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Conv2d> {
    conv2d_no_bias(
        in_ch,
        out_ch,
        kernel,
        conv2d_cfg(kernel, stride, padding, 1),
        vb,
    )
}

fn conv2d_no_bias_layer_groups(
    vb: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<Conv2d> {
    conv2d_no_bias(
        in_ch,
        out_ch,
        kernel,
        conv2d_cfg(kernel, stride, padding, groups),
        vb,
    )
}

fn conv_transpose2d_no_bias_layer(
    vb: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<ConvTranspose2d> {
    conv_transpose2d_no_bias(
        in_ch,
        out_ch,
        kernel,
        conv_transpose2d_cfg(kernel, stride, padding, 1),
        vb,
    )
}

fn conv_transpose2d_no_bias_layer_groups(
    vb: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<ConvTranspose2d> {
    conv_transpose2d_no_bias(
        in_ch,
        out_ch,
        kernel,
        conv_transpose2d_cfg(kernel, stride, padding, groups),
        vb,
    )
}

pub struct ConvBlock {
    conv: Conv2d,
    norm: InstanceNorm2d,
    act: Act,
}

impl Module for ConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = self.norm.forward(&xs)?;
        self.act.forward(&xs)
    }
}

pub struct DownsampleBlock {
    conv: Conv2d,
    norm: InstanceNorm2d,
    act: Act,
}

impl Module for DownsampleBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = self.norm.forward(&xs)?;
        self.act.forward(&xs)
    }
}

pub struct UpsampleBlock {
    conv: ConvTranspose2d,
    norm: InstanceNorm2d,
    act: Act,
}

impl Module for UpsampleBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = self.norm.forward(&xs)?;
        self.act.forward(&xs)
    }
}

pub struct ResnetBlock {
    conv1: Conv2d,
    norm1: InstanceNorm2d,
    act: Act,
    conv2: Conv2d,
    norm2: InstanceNorm2d,
}

impl ResnetBlock {
    pub fn new(num_channels: usize, act: Act, vb: VarBuilder) -> Result<Self> {
        let conv1 = conv2d_no_bias_layer(
            vb.pp("resnet_path").pp("0"),
            num_channels,
            num_channels,
            3,
            1,
            1,
        )?;
        let norm1 = InstanceNorm2d::new(num_channels, vb.pp("resnet_path").pp("1"))?;
        let conv2 = conv2d_no_bias_layer(
            vb.pp("resnet_path").pp("3"),
            num_channels,
            num_channels,
            3,
            1,
            1,
        )?;
        let norm2 = InstanceNorm2d::new(num_channels, vb.pp("resnet_path").pp("4"))?;
        Ok(Self {
            conv1,
            norm1,
            act,
            conv2,
            norm2,
        })
    }
}

impl Module for ResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y = self.conv1.forward(xs)?;
        let y = self.norm1.forward(&y)?;
        let y = self.act.forward(&y)?;
        let y = self.conv2.forward(&y)?;
        let y = self.norm2.forward(&y)?;
        xs.broadcast_add(&y)
    }
}

pub struct SeparableConv {
    depthwise: Conv2d,
    pointwise: Conv2d,
}

impl SeparableConv {
    pub fn new(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let depthwise =
            conv2d_no_bias_layer_groups(vb.pp("0"), in_ch, in_ch, kernel, stride, padding, in_ch)?;
        let pointwise = conv2d_no_bias_layer(vb.pp("1"), in_ch, out_ch, 1, 1, 0)?;
        Ok(Self {
            depthwise,
            pointwise,
        })
    }
}

impl Module for SeparableConv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.depthwise.forward(xs)?;
        self.pointwise.forward(&xs)
    }
}

pub struct SeparableConvBlock {
    conv: SeparableConv,
    norm: InstanceNorm2d,
    act: Act,
}

impl SeparableConvBlock {
    pub fn new(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        act: Act,
    ) -> Result<Self> {
        let conv = SeparableConv::new(vb.pp("0"), in_ch, out_ch, kernel, stride, padding)?;
        let norm = InstanceNorm2d::new(out_ch, vb.pp("2"))?;
        Ok(Self { conv, norm, act })
    }
}

impl Module for SeparableConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = self.norm.forward(&xs)?;
        self.act.forward(&xs)
    }
}

pub struct SeparableDownsampleBlock {
    conv: SeparableConv,
    norm: InstanceNorm2d,
    act: Act,
}

impl SeparableDownsampleBlock {
    pub fn new(vb: VarBuilder, in_ch: usize, out_ch: usize, act: Act) -> Result<Self> {
        let conv = SeparableConv::new(vb.pp("0"), in_ch, out_ch, 4, 2, 1)?;
        let norm = InstanceNorm2d::new(out_ch, vb.pp("2"))?;
        Ok(Self { conv, norm, act })
    }
}

impl Module for SeparableDownsampleBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = self.norm.forward(&xs)?;
        self.act.forward(&xs)
    }
}

pub struct SeparableUpsampleBlock {
    depthwise: ConvTranspose2d,
    pointwise: Conv2d,
    norm: InstanceNorm2d,
    act: Act,
}

impl SeparableUpsampleBlock {
    pub fn new(vb: VarBuilder, in_ch: usize, out_ch: usize, act: Act) -> Result<Self> {
        let depthwise =
            conv_transpose2d_no_bias_layer_groups(vb.pp("0"), in_ch, in_ch, 4, 2, 1, in_ch)?;
        let pointwise = conv2d_no_bias_layer(vb.pp("1"), in_ch, out_ch, 1, 1, 0)?;
        let norm = InstanceNorm2d::new(out_ch, vb.pp("2"))?;
        Ok(Self {
            depthwise,
            pointwise,
            norm,
            act,
        })
    }
}

impl Module for SeparableUpsampleBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.depthwise.forward(xs)?;
        let xs = self.pointwise.forward(&xs)?;
        let xs = self.norm.forward(&xs)?;
        self.act.forward(&xs)
    }
}

pub struct ResnetBlockSeparable {
    conv1: SeparableConv,
    norm1: InstanceNorm2d,
    act: Act,
    conv2: SeparableConv,
    norm2: InstanceNorm2d,
}

impl ResnetBlockSeparable {
    pub fn new(num_channels: usize, act: Act, vb: VarBuilder) -> Result<Self> {
        let conv1 = SeparableConv::new(
            vb.pp("resnet_path").pp("0"),
            num_channels,
            num_channels,
            3,
            1,
            1,
        )?;
        let norm1 = InstanceNorm2d::new(num_channels, vb.pp("resnet_path").pp("1"))?;
        let conv2 = SeparableConv::new(
            vb.pp("resnet_path").pp("3"),
            num_channels,
            num_channels,
            3,
            1,
            1,
        )?;
        let norm2 = InstanceNorm2d::new(num_channels, vb.pp("resnet_path").pp("4"))?;
        Ok(Self {
            conv1,
            norm1,
            act,
            conv2,
            norm2,
        })
    }
}

impl Module for ResnetBlockSeparable {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y = self.conv1.forward(xs)?;
        let y = self.norm1.forward(&y)?;
        let y = self.act.forward(&y)?;
        let y = self.conv2.forward(&y)?;
        let y = self.norm2.forward(&y)?;
        xs.broadcast_add(&y)
    }
}

#[derive(Clone, Copy)]
pub enum UpsampleMode {
    Nearest,
    Bilinear,
}

fn upsample2d(xs: &Tensor, mode: UpsampleMode) -> Result<Tensor> {
    let (_n, _c, h, w) = xs.dims4()?;
    let h2 = h * 2;
    let w2 = w * 2;
    match mode {
        UpsampleMode::Nearest => xs.upsample_nearest2d(h2, w2),
        UpsampleMode::Bilinear => xs.upsample_bilinear2d(h2, w2, false),
    }
}

pub struct PoserEncoderDecoder00Args {
    pub image_size: usize,
    pub input_image_channels: usize,
    pub output_image_channels: usize,
    pub num_pose_params: usize,
    pub start_channels: usize,
    pub bottleneck_image_size: usize,
    pub num_bottleneck_blocks: usize,
    pub max_channels: usize,
}

enum DownBlock {
    Conv(ConvBlock),
    Down(DownsampleBlock),
}

impl DownBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(b) => b.forward(xs),
            Self::Down(b) => b.forward(xs),
        }
    }
}

enum BottleneckBlock {
    Conv(ConvBlock),
    Res(ResnetBlock),
}

impl BottleneckBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(b) => b.forward(xs),
            Self::Res(b) => b.forward(xs),
        }
    }
}

pub struct PoserEncoderDecoder00 {
    args: PoserEncoderDecoder00Args,
    downsample_blocks: Vec<DownBlock>,
    bottleneck_blocks: Vec<BottleneckBlock>,
    upsample_blocks: Vec<UpsampleBlock>,
}

impl PoserEncoderDecoder00 {
    pub fn new(args: PoserEncoderDecoder00Args, vb: VarBuilder, act: Act) -> Result<Self> {
        let mut downsample_blocks = Vec::new();
        let mut current_image_size = args.image_size;
        let mut current_num_channels = args.start_channels;
        let num_levels = (args.image_size / args.bottleneck_image_size).ilog2() as usize + 1;

        downsample_blocks.push(DownBlock::Conv(conv2d_block(
            vb.pp("downsample_blocks").pp("0"),
            args.input_image_channels,
            args.start_channels,
            3,
            1,
            1,
            act,
        )?));
        let mut idx = 1;
        while current_image_size > args.bottleneck_image_size {
            let next_image_size = current_image_size / 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let block = DownsampleBlock {
                conv: conv2d_no_bias_layer(
                    vb.pp("downsample_blocks").pp(idx.to_string()).pp("0"),
                    current_num_channels,
                    next_num_channels,
                    4,
                    2,
                    1,
                )?,
                norm: InstanceNorm2d::new(
                    next_num_channels,
                    vb.pp("downsample_blocks").pp(idx.to_string()).pp("1"),
                )?,
                act,
            };
            downsample_blocks.push(DownBlock::Down(block));
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            idx += 1;
        }
        if downsample_blocks.len() != num_levels {
            candle::bail!("unexpected num_levels in PoserEncoderDecoder00");
        }

        let mut bottleneck_blocks = Vec::new();
        let bottleneck0 = conv2d_block(
            vb.pp("bottleneck_blocks").pp("0"),
            current_num_channels + args.num_pose_params,
            current_num_channels,
            3,
            1,
            1,
            act,
        )?;
        bottleneck_blocks.push(BottleneckBlock::Conv(bottleneck0));
        for i in 1..args.num_bottleneck_blocks {
            let res = ResnetBlock::new(
                current_num_channels,
                act,
                vb.pp("bottleneck_blocks").pp(i.to_string()),
            )?;
            bottleneck_blocks.push(BottleneckBlock::Res(res));
        }

        let mut upsample_blocks = Vec::new();
        let mut up_idx = 0;
        while current_image_size < args.image_size {
            let next_image_size = current_image_size * 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let conv = conv_transpose2d_no_bias_layer(
                vb.pp("upsample_blocks").pp(up_idx.to_string()).pp("0"),
                current_num_channels,
                next_num_channels,
                4,
                2,
                1,
            )?;
            let norm = InstanceNorm2d::new(
                next_num_channels,
                vb.pp("upsample_blocks").pp(up_idx.to_string()).pp("1"),
            )?;
            let block = UpsampleBlock { conv, norm, act };
            upsample_blocks.push(block);
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            up_idx += 1;
        }

        Ok(Self {
            args,
            downsample_blocks,
            bottleneck_blocks,
            upsample_blocks,
        })
    }

    pub fn forward(&self, image: &Tensor, pose: Option<&Tensor>) -> Result<Vec<Tensor>> {
        if self.args.num_pose_params != 0 && pose.is_none() {
            candle::bail!("pose required");
        }
        if self.args.num_pose_params == 0 && pose.is_some() {
            candle::bail!("pose not expected");
        }
        let mut outputs = Vec::new();
        let mut feature = image.clone();
        outputs.push(feature.clone());
        for block in &self.downsample_blocks {
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        if let Some(pose) = pose {
            let (n, c) = pose.dims2()?;
            let pose = pose.reshape((n, c, 1, 1))?.broadcast_as((
                n,
                c,
                self.args.bottleneck_image_size,
                self.args.bottleneck_image_size,
            ))?;
            feature = Tensor::cat(&[&feature, &pose], 1)?;
        }
        for block in &self.bottleneck_blocks {
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        for block in &self.upsample_blocks {
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        outputs.reverse();
        Ok(outputs)
    }
}

enum SepDownBlock {
    Conv(SeparableConvBlock),
    Down(SeparableDownsampleBlock),
}

impl SepDownBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(b) => b.forward(xs),
            Self::Down(b) => b.forward(xs),
        }
    }
}

enum SepBottleneckBlock {
    Conv(SeparableConvBlock),
    Res(ResnetBlockSeparable),
}

impl SepBottleneckBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(b) => b.forward(xs),
            Self::Res(b) => b.forward(xs),
        }
    }
}

pub struct PoserEncoderDecoder00Separable {
    args: PoserEncoderDecoder00Args,
    downsample_blocks: Vec<SepDownBlock>,
    bottleneck_blocks: Vec<SepBottleneckBlock>,
    upsample_blocks: Vec<SeparableUpsampleBlock>,
}

impl PoserEncoderDecoder00Separable {
    pub fn new(args: PoserEncoderDecoder00Args, vb: VarBuilder, act: Act) -> Result<Self> {
        let mut downsample_blocks = Vec::new();
        let mut current_image_size = args.image_size;
        let mut current_num_channels = args.start_channels;
        let num_levels = (args.image_size / args.bottleneck_image_size).ilog2() as usize + 1;

        downsample_blocks.push(SepDownBlock::Conv(SeparableConvBlock::new(
            vb.pp("downsample_blocks").pp("0"),
            args.input_image_channels,
            args.start_channels,
            3,
            1,
            1,
            act,
        )?));
        let mut idx = 1;
        while current_image_size > args.bottleneck_image_size {
            let next_image_size = current_image_size / 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let block = SeparableDownsampleBlock::new(
                vb.pp("downsample_blocks").pp(idx.to_string()),
                current_num_channels,
                next_num_channels,
                act,
            )?;
            downsample_blocks.push(SepDownBlock::Down(block));
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            idx += 1;
        }
        if downsample_blocks.len() != num_levels {
            candle::bail!("unexpected num_levels in PoserEncoderDecoder00Separable");
        }

        let mut bottleneck_blocks = Vec::new();
        let bottleneck0 = SeparableConvBlock::new(
            vb.pp("bottleneck_blocks").pp("0"),
            current_num_channels + args.num_pose_params,
            current_num_channels,
            3,
            1,
            1,
            act,
        )?;
        bottleneck_blocks.push(SepBottleneckBlock::Conv(bottleneck0));
        for i in 1..args.num_bottleneck_blocks {
            let res = ResnetBlockSeparable::new(
                current_num_channels,
                act,
                vb.pp("bottleneck_blocks").pp(i.to_string()),
            )?;
            bottleneck_blocks.push(SepBottleneckBlock::Res(res));
        }

        let mut upsample_blocks = Vec::new();
        let mut up_idx = 0;
        while current_image_size < args.image_size {
            let next_image_size = current_image_size * 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let block = SeparableUpsampleBlock::new(
                vb.pp("upsample_blocks").pp(up_idx.to_string()),
                current_num_channels,
                next_num_channels,
                act,
            )?;
            upsample_blocks.push(block);
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            up_idx += 1;
        }

        Ok(Self {
            args,
            downsample_blocks,
            bottleneck_blocks,
            upsample_blocks,
        })
    }

    pub fn forward(&self, image: &Tensor, pose: Option<&Tensor>) -> Result<Vec<Tensor>> {
        if self.args.num_pose_params != 0 && pose.is_none() {
            candle::bail!("pose required");
        }
        if self.args.num_pose_params == 0 && pose.is_some() {
            candle::bail!("pose not expected");
        }
        let mut outputs = Vec::new();
        let mut feature = image.clone();
        outputs.push(feature.clone());
        for block in &self.downsample_blocks {
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        if let Some(pose) = pose {
            let (n, c) = pose.dims2()?;
            let pose = pose.reshape((n, c, 1, 1))?.broadcast_as((
                n,
                c,
                self.args.bottleneck_image_size,
                self.args.bottleneck_image_size,
            ))?;
            feature = Tensor::cat(&[&feature, &pose], 1)?;
        }
        for block in &self.bottleneck_blocks {
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        for block in &self.upsample_blocks {
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        outputs.reverse();
        Ok(outputs)
    }
}

pub struct ResizeConvEncoderDecoderArgs {
    pub image_size: usize,
    pub input_channels: usize,
    pub start_channels: usize,
    pub bottleneck_image_size: usize,
    pub num_bottleneck_blocks: usize,
    pub max_channels: usize,
    pub upsample_mode: UpsampleMode,
}

pub struct ResizeConvEncoderDecoder {
    args: ResizeConvEncoderDecoderArgs,
    downsample_blocks: Vec<DownBlock>,
    bottleneck_blocks: Vec<ResnetBlock>,
    upsample_blocks: Vec<ConvBlock>,
}

impl ResizeConvEncoderDecoder {
    pub fn new(args: ResizeConvEncoderDecoderArgs, vb: VarBuilder, act: Act) -> Result<Self> {
        let mut downsample_blocks = Vec::new();
        let mut current_image_size = args.image_size;
        let mut current_num_channels = args.start_channels;
        let num_levels = (args.image_size / args.bottleneck_image_size).ilog2() as usize + 1;

        downsample_blocks.push(DownBlock::Conv(conv2d_block(
            vb.pp("downsample_blocks").pp("0"),
            args.input_channels,
            args.start_channels,
            7,
            1,
            3,
            act,
        )?));
        let mut idx = 1;
        while current_image_size > args.bottleneck_image_size {
            let next_image_size = current_image_size / 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let block = DownsampleBlock {
                conv: conv2d_no_bias_layer(
                    vb.pp("downsample_blocks").pp(idx.to_string()).pp("0"),
                    current_num_channels,
                    next_num_channels,
                    4,
                    2,
                    1,
                )?,
                norm: InstanceNorm2d::new(
                    next_num_channels,
                    vb.pp("downsample_blocks").pp(idx.to_string()).pp("1"),
                )?,
                act,
            };
            downsample_blocks.push(DownBlock::Down(block));
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            idx += 1;
        }
        if downsample_blocks.len() != num_levels {
            candle::bail!("unexpected num_levels in ResizeConvEncoderDecoder");
        }

        let mut bottleneck_blocks = Vec::new();
        for i in 0..args.num_bottleneck_blocks {
            let res = ResnetBlock::new(
                current_num_channels,
                act,
                vb.pp("bottleneck_blocks").pp(i.to_string()),
            )?;
            bottleneck_blocks.push(res);
        }

        let mut upsample_blocks = Vec::new();
        let mut up_idx = 0;
        while current_image_size < args.image_size {
            let next_image_size = current_image_size * 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let block = conv2d_block(
                vb.pp("upsample_blocks").pp(up_idx.to_string()).pp("1"),
                current_num_channels,
                next_num_channels,
                3,
                1,
                1,
                act,
            )?;
            upsample_blocks.push(block);
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            up_idx += 1;
        }

        Ok(Self {
            args,
            downsample_blocks,
            bottleneck_blocks,
            upsample_blocks,
        })
    }

    pub fn forward(&self, feature: &Tensor) -> Result<Vec<Tensor>> {
        let mut feature = feature.clone();
        for block in &self.downsample_blocks {
            feature = block.forward(&feature)?;
        }
        for block in &self.bottleneck_blocks {
            feature = block.forward(&feature)?;
        }
        let mut outputs = vec![feature.clone()];
        for block in &self.upsample_blocks {
            feature = upsample2d(&feature, self.args.upsample_mode)?;
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        Ok(outputs)
    }
}

pub struct ResizeConvEncoderDecoderSeparable {
    args: ResizeConvEncoderDecoderArgs,
    downsample_blocks: Vec<SepDownBlock>,
    bottleneck_blocks: Vec<ResnetBlockSeparable>,
    upsample_blocks: Vec<SeparableConvBlock>,
}

impl ResizeConvEncoderDecoderSeparable {
    pub fn new(args: ResizeConvEncoderDecoderArgs, vb: VarBuilder, act: Act) -> Result<Self> {
        let mut downsample_blocks = Vec::new();
        let mut current_image_size = args.image_size;
        let mut current_num_channels = args.start_channels;
        let num_levels = (args.image_size / args.bottleneck_image_size).ilog2() as usize + 1;

        downsample_blocks.push(SepDownBlock::Conv(SeparableConvBlock::new(
            vb.pp("downsample_blocks").pp("0"),
            args.input_channels,
            args.start_channels,
            7,
            1,
            3,
            act,
        )?));
        let mut idx = 1;
        while current_image_size > args.bottleneck_image_size {
            let next_image_size = current_image_size / 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let block = SeparableDownsampleBlock::new(
                vb.pp("downsample_blocks").pp(idx.to_string()),
                current_num_channels,
                next_num_channels,
                act,
            )?;
            downsample_blocks.push(SepDownBlock::Down(block));
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            idx += 1;
        }
        if downsample_blocks.len() != num_levels {
            candle::bail!("unexpected num_levels in ResizeConvEncoderDecoderSeparable");
        }

        let mut bottleneck_blocks = Vec::new();
        for i in 0..args.num_bottleneck_blocks {
            let res = ResnetBlockSeparable::new(
                current_num_channels,
                act,
                vb.pp("bottleneck_blocks").pp(i.to_string()),
            )?;
            bottleneck_blocks.push(res);
        }

        let mut upsample_blocks = Vec::new();
        let mut up_idx = 0;
        while current_image_size < args.image_size {
            let next_image_size = current_image_size * 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let block = SeparableConvBlock::new(
                vb.pp("upsample_blocks").pp(up_idx.to_string()).pp("1"),
                current_num_channels,
                next_num_channels,
                3,
                1,
                1,
                act,
            )?;
            upsample_blocks.push(block);
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            up_idx += 1;
        }

        Ok(Self {
            args,
            downsample_blocks,
            bottleneck_blocks,
            upsample_blocks,
        })
    }

    pub fn forward(&self, feature: &Tensor) -> Result<Vec<Tensor>> {
        let mut feature = feature.clone();
        for block in &self.downsample_blocks {
            feature = block.forward(&feature)?;
        }
        for block in &self.bottleneck_blocks {
            feature = block.forward(&feature)?;
        }
        let mut outputs = vec![feature.clone()];
        for block in &self.upsample_blocks {
            feature = upsample2d(&feature, self.args.upsample_mode)?;
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        Ok(outputs)
    }
}

pub struct ResizeConvUNetArgs {
    pub image_size: usize,
    pub input_channels: usize,
    pub start_channels: usize,
    pub bottleneck_image_size: usize,
    pub num_bottleneck_blocks: usize,
    pub max_channels: usize,
    pub upsample_mode: UpsampleMode,
}

pub struct ResizeConvUNet {
    args: ResizeConvUNetArgs,
    downsample_blocks: Vec<DownBlock>,
    bottleneck_blocks: Vec<ResnetBlock>,
    upsample_blocks: Vec<ConvBlock>,
    size_to_channel: std::collections::HashMap<usize, usize>,
}

impl ResizeConvUNet {
    pub fn new(args: ResizeConvUNetArgs, vb: VarBuilder, act: Act) -> Result<Self> {
        let mut downsample_blocks = Vec::new();
        let mut current_image_size = args.image_size;
        let mut current_num_channels = args.start_channels;
        let mut size_to_channel = std::collections::HashMap::new();

        downsample_blocks.push(DownBlock::Conv(conv2d_block(
            vb.pp("downsample_blocks").pp("0"),
            args.input_channels,
            args.start_channels,
            3,
            1,
            1,
            act,
        )?));
        size_to_channel.insert(current_image_size, current_num_channels);
        let mut idx = 1;
        while current_image_size > args.bottleneck_image_size {
            let next_image_size = current_image_size / 2;
            let next_num_channels = (current_num_channels * 2).min(args.max_channels);
            let block = DownsampleBlock {
                conv: conv2d_no_bias_layer(
                    vb.pp("downsample_blocks").pp(idx.to_string()).pp("0"),
                    current_num_channels,
                    next_num_channels,
                    4,
                    2,
                    1,
                )?,
                norm: InstanceNorm2d::new(
                    next_num_channels,
                    vb.pp("downsample_blocks").pp(idx.to_string()).pp("1"),
                )?,
                act,
            };
            downsample_blocks.push(DownBlock::Down(block));
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            size_to_channel.insert(current_image_size, current_num_channels);
            idx += 1;
        }

        let mut bottleneck_blocks = Vec::new();
        for i in 0..args.num_bottleneck_blocks {
            let res = ResnetBlock::new(
                current_num_channels,
                act,
                vb.pp("bottleneck_blocks").pp(i.to_string()),
            )?;
            bottleneck_blocks.push(res);
        }

        let mut upsample_blocks = Vec::new();
        let mut up_idx = 0;
        while current_image_size < args.image_size {
            let next_image_size = current_image_size * 2;
            let next_num_channels = *size_to_channel
                .get(&next_image_size)
                .ok_or_else(|| candle::Error::Msg("missing size_to_channel".to_string()))?;
            let block = conv2d_block(
                vb.pp("upsample_blocks").pp(up_idx.to_string()),
                current_num_channels + next_num_channels,
                next_num_channels,
                3,
                1,
                1,
                act,
            )?;
            upsample_blocks.push(block);
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            up_idx += 1;
        }

        Ok(Self {
            args,
            downsample_blocks,
            bottleneck_blocks,
            upsample_blocks,
            size_to_channel,
        })
    }

    pub fn forward(&self, feature: &Tensor) -> Result<Vec<Tensor>> {
        let mut feature = feature.clone();
        let mut downsampled = Vec::new();
        for block in &self.downsample_blocks {
            feature = block.forward(&feature)?;
            downsampled.push(feature.clone());
        }
        for block in &self.bottleneck_blocks {
            feature = block.forward(&feature)?;
        }
        let mut outputs = vec![feature.clone()];
        for (i, block) in self.upsample_blocks.iter().enumerate() {
            feature = upsample2d(&feature, self.args.upsample_mode)?;
            let skip = downsampled
                .get(downsampled.len().saturating_sub(i + 2))
                .ok_or_else(|| candle::Error::Msg("missing skip".to_string()))?;
            feature = Tensor::cat(&[&feature, skip], 1)?;
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        Ok(outputs)
    }
}

pub struct ResizeConvUNetSeparable {
    args: ResizeConvUNetArgs,
    downsample_blocks: Vec<SepDownBlock>,
    bottleneck_blocks: Vec<ResnetBlockSeparable>,
    upsample_blocks: Vec<SeparableConvBlock>,
    size_to_channel: std::collections::HashMap<usize, usize>,
}

impl ResizeConvUNetSeparable {
    pub fn new(args: ResizeConvUNetArgs, vb: VarBuilder, act: Act) -> Result<Self> {
        let mut downsample_blocks = Vec::new();
        let mut current_image_size = args.image_size;
        let mut current_num_channels = args.start_channels;
        let mut size_to_channel = std::collections::HashMap::new();

        downsample_blocks.push(SepDownBlock::Conv(SeparableConvBlock::new(
            vb.pp("downsample_blocks").pp("0"),
            args.input_channels,
            args.start_channels,
            3,
            1,
            1,
            act,
        )?));
        size_to_channel.insert(current_image_size, current_num_channels);
        let mut idx = 1;
        while current_image_size > args.bottleneck_image_size {
            let next_image_size = current_image_size / 2;
            let next_num_channels = (current_num_channels * 2).min(args.max_channels);
            let block = SeparableDownsampleBlock::new(
                vb.pp("downsample_blocks").pp(idx.to_string()),
                current_num_channels,
                next_num_channels,
                act,
            )?;
            downsample_blocks.push(SepDownBlock::Down(block));
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            size_to_channel.insert(current_image_size, current_num_channels);
            idx += 1;
        }

        let mut bottleneck_blocks = Vec::new();
        for i in 0..args.num_bottleneck_blocks {
            let res = ResnetBlockSeparable::new(
                current_num_channels,
                act,
                vb.pp("bottleneck_blocks").pp(i.to_string()),
            )?;
            bottleneck_blocks.push(res);
        }

        let mut upsample_blocks = Vec::new();
        let mut up_idx = 0;
        while current_image_size < args.image_size {
            let next_image_size = current_image_size * 2;
            let next_num_channels = *size_to_channel
                .get(&next_image_size)
                .ok_or_else(|| candle::Error::Msg("missing size_to_channel".to_string()))?;
            let block = SeparableConvBlock::new(
                vb.pp("upsample_blocks").pp(up_idx.to_string()),
                current_num_channels + next_num_channels,
                next_num_channels,
                3,
                1,
                1,
                act,
            )?;
            upsample_blocks.push(block);
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            up_idx += 1;
        }

        Ok(Self {
            args,
            downsample_blocks,
            bottleneck_blocks,
            upsample_blocks,
            size_to_channel,
        })
    }

    pub fn forward(&self, feature: &Tensor) -> Result<Vec<Tensor>> {
        let mut feature = feature.clone();
        let mut downsampled = Vec::new();
        for block in &self.downsample_blocks {
            feature = block.forward(&feature)?;
            downsampled.push(feature.clone());
        }
        for block in &self.bottleneck_blocks {
            feature = block.forward(&feature)?;
        }
        let mut outputs = vec![feature.clone()];
        for (i, block) in self.upsample_blocks.iter().enumerate() {
            feature = upsample2d(&feature, self.args.upsample_mode)?;
            let skip = downsampled
                .get(downsampled.len().saturating_sub(i + 2))
                .ok_or_else(|| candle::Error::Msg("missing skip".to_string()))?;
            feature = Tensor::cat(&[&feature, skip], 1)?;
            feature = block.forward(&feature)?;
            outputs.push(feature.clone());
        }
        Ok(outputs)
    }
}

pub struct ConvAct {
    conv: Conv2d,
    act: Act,
}

impl ConvAct {
    pub fn new(vb: VarBuilder, in_ch: usize, out_ch: usize, act: Act) -> Result<Self> {
        let conv = conv2d_bias(vb.pp("0"), in_ch, out_ch, 3, 1, 1)?;
        Ok(Self { conv, act })
    }
}

impl Module for ConvAct {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        self.act.forward(&xs)
    }
}

fn act_sigmoid() -> Act {
    Act::Sigmoid
}

fn act_tanh() -> Act {
    Act::Tanh
}

pub struct EyebrowDecomposer00 {
    body: PoserEncoderDecoder00,
    background_layer_alpha: ConvAct,
    background_layer_color_change: ConvAct,
    eyebrow_layer_alpha: ConvAct,
    eyebrow_layer_color_change: ConvAct,
}

impl EyebrowDecomposer00 {
    pub const EYEBROW_LAYER_INDEX: usize = 0;
    pub const BACKGROUND_LAYER_INDEX: usize = 3;

    pub fn new(args: PoserEncoderDecoder00Args, vb: VarBuilder) -> Result<Self> {
        let act = Act::Relu;
        let start_channels = args.start_channels;
        let output_channels = args.output_image_channels;
        let body = PoserEncoderDecoder00::new(args, vb.pp("body"), act)?;
        let background_layer_alpha = ConvAct::new(
            vb.pp("background_layer_alpha"),
            start_channels,
            1,
            act_sigmoid(),
        )?;
        let background_layer_color_change = ConvAct::new(
            vb.pp("background_layer_color_change"),
            start_channels,
            output_channels,
            act_tanh(),
        )?;
        let eyebrow_layer_alpha = ConvAct::new(
            vb.pp("eyebrow_layer_alpha"),
            start_channels,
            1,
            act_sigmoid(),
        )?;
        let eyebrow_layer_color_change = ConvAct::new(
            vb.pp("eyebrow_layer_color_change"),
            start_channels,
            output_channels,
            act_tanh(),
        )?;
        Ok(Self {
            body,
            background_layer_alpha,
            background_layer_color_change,
            eyebrow_layer_alpha,
            eyebrow_layer_color_change,
        })
    }

    pub fn forward(&self, image: &Tensor) -> Result<Vec<Tensor>> {
        let feature = self.body.forward(image, None)?[0].clone();
        let background_layer_alpha = self.background_layer_alpha.forward(&feature)?;
        let background_layer_color_change = self.background_layer_color_change.forward(&feature)?;
        let background_layer_1 = apply_color_change(
            &background_layer_alpha,
            &background_layer_color_change,
            image,
        )?;
        let eyebrow_layer_alpha = self.eyebrow_layer_alpha.forward(&feature)?;
        let eyebrow_layer_color_change = self.eyebrow_layer_color_change.forward(&feature)?;
        let eyebrow_layer =
            apply_color_change(&eyebrow_layer_alpha, image, &eyebrow_layer_color_change)?;
        Ok(vec![
            eyebrow_layer,
            eyebrow_layer_alpha,
            eyebrow_layer_color_change,
            background_layer_1,
            background_layer_alpha,
            background_layer_color_change,
        ])
    }
}

pub struct EyebrowMorphingCombiner00 {
    body: PoserEncoderDecoder00,
    morphed_eyebrow_layer_grid_change: Conv2d,
    morphed_eyebrow_layer_alpha: ConvAct,
    morphed_eyebrow_layer_color_change: ConvAct,
    combine_alpha: ConvAct,
    grid_change_applier: std::sync::Mutex<GridChangeApplier>,
}

impl EyebrowMorphingCombiner00 {
    pub const EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX: usize = 2;

    pub fn new(args: PoserEncoderDecoder00Args, vb: VarBuilder) -> Result<Self> {
        let act = Act::Relu;
        let start_channels = args.start_channels;
        let output_channels = args.output_image_channels;
        let body = PoserEncoderDecoder00::new(args, vb.pp("body"), act)?;
        let morphed_eyebrow_layer_grid_change = conv2d_no_bias_layer(
            vb.pp("morphed_eyebrow_layer_grid_change"),
            start_channels,
            2,
            3,
            1,
            1,
        )?;
        let morphed_eyebrow_layer_alpha = ConvAct::new(
            vb.pp("morphed_eyebrow_layer_alpha"),
            start_channels,
            1,
            act_sigmoid(),
        )?;
        let morphed_eyebrow_layer_color_change = ConvAct::new(
            vb.pp("morphed_eyebrow_layer_color_change"),
            start_channels,
            output_channels,
            act_tanh(),
        )?;
        let combine_alpha = ConvAct::new(vb.pp("combine_alpha"), start_channels, 1, act_sigmoid())?;
        Ok(Self {
            body,
            morphed_eyebrow_layer_grid_change,
            morphed_eyebrow_layer_alpha,
            morphed_eyebrow_layer_color_change,
            combine_alpha,
            grid_change_applier: std::sync::Mutex::new(GridChangeApplier::new()),
        })
    }

    pub fn forward(
        &self,
        background_layer: &Tensor,
        eyebrow_layer: &Tensor,
        pose: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let combined = Tensor::cat(&[background_layer, eyebrow_layer], 1)?;
        let feature = self.body.forward(&combined, Some(pose))?[0].clone();
        let grid_change = self.morphed_eyebrow_layer_grid_change.forward(&feature)?;
        let alpha = self.morphed_eyebrow_layer_alpha.forward(&feature)?;
        let color_change = self.morphed_eyebrow_layer_color_change.forward(&feature)?;
        let warped = {
            let mut applier = self.grid_change_applier.lock().unwrap();
            applier.apply(&grid_change, eyebrow_layer)?
        };
        let morphed = apply_color_change(&alpha, &color_change, &warped)?;
        let combine_alpha = self.combine_alpha.forward(&feature)?;
        let eyebrow_image = apply_rgb_change(&combine_alpha, &morphed, background_layer)?;
        let combine_alpha2 = (morphed.narrow(1, 3, 1)? + 1.0)?;
        let combine_alpha2 = (&combine_alpha2 / 2.0)?;
        let eyebrow_image_no_combine_alpha =
            apply_rgb_change(&combine_alpha2, &morphed, background_layer)?;
        Ok(vec![
            eyebrow_image,
            combine_alpha,
            eyebrow_image_no_combine_alpha,
            morphed,
            alpha,
            color_change,
            warped,
            grid_change,
        ])
    }
}

pub struct EyebrowDecomposer03 {
    body: PoserEncoderDecoder00Separable,
    background_layer_alpha: ConvAct,
    background_layer_color_change: ConvAct,
    eyebrow_layer_alpha: ConvAct,
    eyebrow_layer_color_change: ConvAct,
}

impl EyebrowDecomposer03 {
    pub const EYEBROW_LAYER_INDEX: usize = 0;
    pub const BACKGROUND_LAYER_INDEX: usize = 3;

    pub fn new(args: PoserEncoderDecoder00Args, vb: VarBuilder) -> Result<Self> {
        let act = Act::Relu;
        let start_channels = args.start_channels;
        let output_channels = args.output_image_channels;
        let body = PoserEncoderDecoder00Separable::new(args, vb.pp("body"), act)?;
        let background_layer_alpha = ConvAct::new(
            vb.pp("background_layer_alpha"),
            start_channels,
            1,
            act_sigmoid(),
        )?;
        let background_layer_color_change = ConvAct::new(
            vb.pp("background_layer_color_change"),
            start_channels,
            output_channels,
            act_tanh(),
        )?;
        let eyebrow_layer_alpha = ConvAct::new(
            vb.pp("eyebrow_layer_alpha"),
            start_channels,
            1,
            act_sigmoid(),
        )?;
        let eyebrow_layer_color_change = ConvAct::new(
            vb.pp("eyebrow_layer_color_change"),
            start_channels,
            output_channels,
            act_tanh(),
        )?;
        Ok(Self {
            body,
            background_layer_alpha,
            background_layer_color_change,
            eyebrow_layer_alpha,
            eyebrow_layer_color_change,
        })
    }

    pub fn forward(&self, image: &Tensor) -> Result<Vec<Tensor>> {
        let feature = self.body.forward(image, None)?[0].clone();
        let background_layer_alpha = self.background_layer_alpha.forward(&feature)?;
        let background_layer_color_change = self.background_layer_color_change.forward(&feature)?;
        let background_layer_1 = apply_color_change(
            &background_layer_alpha,
            &background_layer_color_change,
            image,
        )?;
        let eyebrow_layer_alpha = self.eyebrow_layer_alpha.forward(&feature)?;
        let eyebrow_layer_color_change = self.eyebrow_layer_color_change.forward(&feature)?;
        let eyebrow_layer =
            apply_color_change(&eyebrow_layer_alpha, image, &eyebrow_layer_color_change)?;
        Ok(vec![
            eyebrow_layer,
            eyebrow_layer_alpha,
            eyebrow_layer_color_change,
            background_layer_1,
            background_layer_alpha,
            background_layer_color_change,
        ])
    }
}

pub struct EyebrowMorphingCombiner03 {
    body: PoserEncoderDecoder00Separable,
    morphed_eyebrow_layer_grid_change: Conv2d,
    morphed_eyebrow_layer_alpha: ConvAct,
    morphed_eyebrow_layer_color_change: ConvAct,
    combine_alpha: ConvAct,
    grid_change_applier: std::sync::Mutex<GridChangeApplier>,
}

impl EyebrowMorphingCombiner03 {
    pub const EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX: usize = 2;

    pub fn new(args: PoserEncoderDecoder00Args, vb: VarBuilder) -> Result<Self> {
        let act = Act::Relu;
        let start_channels = args.start_channels;
        let output_channels = args.output_image_channels;
        let body = PoserEncoderDecoder00Separable::new(args, vb.pp("body"), act)?;
        let morphed_eyebrow_layer_grid_change = conv2d_no_bias_layer(
            vb.pp("morphed_eyebrow_layer_grid_change"),
            start_channels,
            2,
            3,
            1,
            1,
        )?;
        let morphed_eyebrow_layer_alpha = ConvAct::new(
            vb.pp("morphed_eyebrow_layer_alpha"),
            start_channels,
            1,
            act_sigmoid(),
        )?;
        let morphed_eyebrow_layer_color_change = ConvAct::new(
            vb.pp("morphed_eyebrow_layer_color_change"),
            start_channels,
            output_channels,
            act_tanh(),
        )?;
        let combine_alpha = ConvAct::new(vb.pp("combine_alpha"), start_channels, 1, act_sigmoid())?;
        Ok(Self {
            body,
            morphed_eyebrow_layer_grid_change,
            morphed_eyebrow_layer_alpha,
            morphed_eyebrow_layer_color_change,
            combine_alpha,
            grid_change_applier: std::sync::Mutex::new(GridChangeApplier::new()),
        })
    }

    pub fn forward(
        &self,
        background_layer: &Tensor,
        eyebrow_layer: &Tensor,
        pose: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let combined = Tensor::cat(&[background_layer, eyebrow_layer], 1)?;
        let feature = self.body.forward(&combined, Some(pose))?[0].clone();
        let grid_change = self.morphed_eyebrow_layer_grid_change.forward(&feature)?;
        let alpha = self.morphed_eyebrow_layer_alpha.forward(&feature)?;
        let color_change = self.morphed_eyebrow_layer_color_change.forward(&feature)?;
        let warped = {
            let mut applier = self.grid_change_applier.lock().unwrap();
            applier.apply(&grid_change, eyebrow_layer)?
        };
        let morphed = apply_color_change(&alpha, &color_change, &warped)?;
        let combine_alpha = self.combine_alpha.forward(&feature)?;
        let eyebrow_image = apply_rgb_change(&combine_alpha, &morphed, background_layer)?;
        let combine_alpha2 = (morphed.narrow(1, 3, 1)? + 1.0)?;
        let combine_alpha2 = (&combine_alpha2 / 2.0)?;
        let eyebrow_image_no_combine_alpha =
            apply_rgb_change(&combine_alpha2, &morphed, background_layer)?;
        Ok(vec![
            eyebrow_image,
            combine_alpha,
            eyebrow_image_no_combine_alpha,
            morphed,
            alpha,
            color_change,
            warped,
            grid_change,
        ])
    }
}

pub struct FaceMorpher08Args {
    pub image_size: usize,
    pub image_channels: usize,
    pub num_expression_params: usize,
    pub start_channels: usize,
    pub bottleneck_image_size: usize,
    pub num_bottleneck_blocks: usize,
    pub max_channels: usize,
}

pub struct FaceMorpher08 {
    args: FaceMorpher08Args,
    downsample_blocks: Vec<DownBlock>,
    bottleneck_blocks: Vec<BottleneckBlock>,
    upsample_blocks: Vec<UpsampleBlock>,
    iris_mouth_grid_change: Conv2d,
    iris_mouth_color_change: ConvAct,
    iris_mouth_alpha: ConvAct,
    eye_color_change: ConvAct,
    eye_alpha: ConvAct,
    grid_change_applier: std::sync::Mutex<GridChangeApplier>,
}

impl FaceMorpher08 {
    pub fn new(args: FaceMorpher08Args, vb: VarBuilder, act: Act) -> Result<Self> {
        let mut downsample_blocks = Vec::new();
        let mut current_image_size = args.image_size;
        let mut current_num_channels = args.start_channels;
        let num_levels = (args.image_size / args.bottleneck_image_size).ilog2() as usize + 1;

        downsample_blocks.push(DownBlock::Conv(conv2d_block(
            vb.pp("downsample_blocks").pp("0"),
            args.image_channels,
            args.start_channels,
            3,
            1,
            1,
            act,
        )?));
        let mut idx = 1;
        while current_image_size > args.bottleneck_image_size {
            let next_image_size = current_image_size / 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let block = DownsampleBlock {
                conv: conv2d_no_bias_layer(
                    vb.pp("downsample_blocks").pp(idx.to_string()).pp("0"),
                    current_num_channels,
                    next_num_channels,
                    4,
                    2,
                    1,
                )?,
                norm: InstanceNorm2d::new(
                    next_num_channels,
                    vb.pp("downsample_blocks").pp(idx.to_string()).pp("1"),
                )?,
                act,
            };
            downsample_blocks.push(DownBlock::Down(block));
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            idx += 1;
        }
        if downsample_blocks.len() != num_levels {
            candle::bail!("unexpected num_levels in FaceMorpher08");
        }

        let mut bottleneck_blocks = Vec::new();
        let bottleneck0 = conv2d_block(
            vb.pp("bottleneck_blocks").pp("0"),
            current_num_channels + args.num_expression_params,
            current_num_channels,
            3,
            1,
            1,
            act,
        )?;
        bottleneck_blocks.push(BottleneckBlock::Conv(bottleneck0));
        for i in 1..args.num_bottleneck_blocks {
            let res = ResnetBlock::new(
                current_num_channels,
                act,
                vb.pp("bottleneck_blocks").pp(i.to_string()),
            )?;
            bottleneck_blocks.push(BottleneckBlock::Res(res));
        }

        let mut upsample_blocks = Vec::new();
        let mut up_idx = 0;
        while current_image_size < args.image_size {
            let next_image_size = current_image_size * 2;
            let next_num_channels =
                (args.start_channels * (args.image_size / next_image_size)).min(args.max_channels);
            let conv = conv_transpose2d_no_bias_layer(
                vb.pp("upsample_blocks").pp(up_idx.to_string()).pp("0"),
                current_num_channels,
                next_num_channels,
                4,
                2,
                1,
            )?;
            let norm = InstanceNorm2d::new(
                next_num_channels,
                vb.pp("upsample_blocks").pp(up_idx.to_string()).pp("1"),
            )?;
            let block = UpsampleBlock { conv, norm, act };
            upsample_blocks.push(block);
            current_image_size = next_image_size;
            current_num_channels = next_num_channels;
            up_idx += 1;
        }

        let iris_mouth_grid_change = conv2d_no_bias_layer(
            vb.pp("iris_mouth_grid_change"),
            args.start_channels,
            2,
            3,
            1,
            1,
        )?;
        let iris_mouth_color_change = ConvAct::new(
            vb.pp("iris_mouth_color_change"),
            args.start_channels,
            args.image_channels,
            act_tanh(),
        )?;
        let iris_mouth_alpha = ConvAct::new(
            vb.pp("iris_mouth_alpha"),
            args.start_channels,
            1,
            act_sigmoid(),
        )?;
        let eye_color_change = ConvAct::new(
            vb.pp("eye_color_change"),
            args.start_channels,
            args.image_channels,
            act_tanh(),
        )?;
        let eye_alpha = ConvAct::new(vb.pp("eye_alpha"), args.start_channels, 1, act_sigmoid())?;

        Ok(Self {
            args,
            downsample_blocks,
            bottleneck_blocks,
            upsample_blocks,
            iris_mouth_grid_change,
            iris_mouth_color_change,
            iris_mouth_alpha,
            eye_color_change,
            eye_alpha,
            grid_change_applier: std::sync::Mutex::new(GridChangeApplier::new()),
        })
    }

    pub fn forward(&self, image: &Tensor, pose: &Tensor) -> Result<Vec<Tensor>> {
        let mut feature = image.clone();
        for block in &self.downsample_blocks {
            feature = block.forward(&feature)?;
        }
        let (n, c) = pose.dims2()?;
        let pose = pose.reshape((n, c, 1, 1))?.broadcast_as((
            n,
            c,
            self.args.bottleneck_image_size,
            self.args.bottleneck_image_size,
        ))?;
        feature = Tensor::cat(&[&feature, &pose], 1)?;
        for block in &self.bottleneck_blocks {
            feature = block.forward(&feature)?;
        }
        for block in &self.upsample_blocks {
            feature = block.forward(&feature)?;
        }
        let iris_mouth_grid_change = self.iris_mouth_grid_change.forward(&feature)?;
        let iris_mouth_image_0 = {
            let mut applier = self.grid_change_applier.lock().unwrap();
            applier.apply(&iris_mouth_grid_change, image)?
        };
        let iris_mouth_color_change = self.iris_mouth_color_change.forward(&feature)?;
        let iris_mouth_alpha = self.iris_mouth_alpha.forward(&feature)?;
        let iris_mouth_image_1 = apply_color_change(
            &iris_mouth_alpha,
            &iris_mouth_color_change,
            &iris_mouth_image_0,
        )?;
        let eye_color_change = self.eye_color_change.forward(&feature)?;
        let eye_alpha = self.eye_alpha.forward(&feature)?;
        let output_image = apply_color_change(&eye_alpha, &eye_color_change, &iris_mouth_image_1)?;
        Ok(vec![
            output_image,
            eye_alpha,
            eye_color_change,
            iris_mouth_image_1,
            iris_mouth_alpha,
            iris_mouth_color_change,
            iris_mouth_image_0,
        ])
    }
}

pub struct FaceMorpher09 {
    body: PoserEncoderDecoder00Separable,
    iris_mouth_grid_change: Conv2d,
    iris_mouth_color_change: ConvAct,
    iris_mouth_alpha: ConvAct,
    eye_color_change: ConvAct,
    eye_alpha: ConvAct,
    grid_change_applier: std::sync::Mutex<GridChangeApplier>,
}

impl FaceMorpher09 {
    pub fn new(args: FaceMorpher08Args, vb: VarBuilder, act: Act) -> Result<Self> {
        let body = PoserEncoderDecoder00Separable::new(
            PoserEncoderDecoder00Args {
                image_size: args.image_size,
                input_image_channels: args.image_channels,
                output_image_channels: args.image_channels,
                num_pose_params: args.num_expression_params,
                start_channels: args.start_channels,
                bottleneck_image_size: args.bottleneck_image_size,
                num_bottleneck_blocks: args.num_bottleneck_blocks,
                max_channels: args.max_channels,
            },
            vb.pp("body"),
            act,
        )?;

        let iris_mouth_grid_change = conv2d_no_bias_layer(
            vb.pp("iris_mouth_grid_change"),
            args.start_channels,
            2,
            3,
            1,
            1,
        )?;
        let iris_mouth_color_change = ConvAct::new(
            vb.pp("iris_mouth_color_change"),
            args.start_channels,
            args.image_channels,
            act_tanh(),
        )?;
        let iris_mouth_alpha = ConvAct::new(
            vb.pp("iris_mouth_alpha"),
            args.start_channels,
            1,
            act_sigmoid(),
        )?;
        let eye_color_change = ConvAct::new(
            vb.pp("eye_color_change"),
            args.start_channels,
            args.image_channels,
            act_tanh(),
        )?;
        let eye_alpha = ConvAct::new(vb.pp("eye_alpha"), args.start_channels, 1, act_sigmoid())?;

        Ok(Self {
            body,
            iris_mouth_grid_change,
            iris_mouth_color_change,
            iris_mouth_alpha,
            eye_color_change,
            eye_alpha,
            grid_change_applier: std::sync::Mutex::new(GridChangeApplier::new()),
        })
    }

    pub fn forward(&self, image: &Tensor, pose: &Tensor) -> Result<Vec<Tensor>> {
        let feature = self.body.forward(image, Some(pose))?[0].clone();
        let iris_mouth_grid_change = self.iris_mouth_grid_change.forward(&feature)?;
        let iris_mouth_image_0 = {
            let mut applier = self.grid_change_applier.lock().unwrap();
            applier.apply(&iris_mouth_grid_change, image)?
        };
        let iris_mouth_color_change = self.iris_mouth_color_change.forward(&feature)?;
        let iris_mouth_alpha = self.iris_mouth_alpha.forward(&feature)?;
        let iris_mouth_image_1 = apply_color_change(
            &iris_mouth_alpha,
            &iris_mouth_color_change,
            &iris_mouth_image_0,
        )?;
        let eye_color_change = self.eye_color_change.forward(&feature)?;
        let eye_alpha = self.eye_alpha.forward(&feature)?;
        let output_image = apply_color_change(&eye_alpha, &eye_color_change, &iris_mouth_image_1)?;
        Ok(vec![
            output_image,
            eye_alpha,
            eye_color_change,
            iris_mouth_image_1,
            iris_mouth_alpha,
            iris_mouth_color_change,
            iris_mouth_image_0,
        ])
    }
}

pub struct TwoAlgoFaceBodyRotator05 {
    encoder_decoder: ResizeConvEncoderDecoder,
    direct_creator: ConvAct,
    grid_change_creator: Conv2d,
    grid_change_applier: std::sync::Mutex<GridChangeApplier>,
}

impl TwoAlgoFaceBodyRotator05 {
    pub const WARPED_IMAGE_INDEX: usize = 1;
    pub const GRID_CHANGE_INDEX: usize = 2;

    pub fn new(args: ResizeConvEncoderDecoderArgs, vb: VarBuilder, act: Act) -> Result<Self> {
        let start_channels = args.start_channels;
        let encoder_decoder = ResizeConvEncoderDecoder::new(args, vb.pp("encoder_decoder"), act)?;
        let direct_creator = ConvAct::new(vb.pp("direct_creator"), start_channels, 4, act_tanh())?;
        let grid_change_creator =
            conv2d_no_bias_layer(vb.pp("grid_change_creator"), start_channels, 2, 3, 1, 1)?;
        Ok(Self {
            encoder_decoder,
            direct_creator,
            grid_change_creator,
            grid_change_applier: std::sync::Mutex::new(GridChangeApplier::new()),
        })
    }

    pub fn forward(&self, image: &Tensor, pose: &Tensor) -> Result<Vec<Tensor>> {
        let (n, c) = pose.dims2()?;
        let pose = pose.reshape((n, c, 1, 1))?.broadcast_as((
            n,
            c,
            self.encoder_decoder.args.image_size,
            self.encoder_decoder.args.image_size,
        ))?;
        let feature = Tensor::cat(&[image, &pose], 1)?;
        let feature = self.encoder_decoder.forward(&feature)?;
        let feature = feature
            .last()
            .ok_or_else(|| candle::Error::Msg("missing feature".to_string()))?;
        let grid_change = self.grid_change_creator.forward(feature)?;
        let direct_image = self.direct_creator.forward(feature)?;
        let warped_image = {
            let mut applier = self.grid_change_applier.lock().unwrap();
            applier.apply(&grid_change, image)?
        };
        Ok(vec![direct_image, warped_image, grid_change])
    }
}

pub struct Editor07 {
    body: ResizeConvUNet,
    color_change_creator: ConvAct,
    alpha_creator: ConvAct,
    grid_change_creator: Conv2d,
    grid_change_applier: std::sync::Mutex<GridChangeApplier>,
}

impl Editor07 {
    pub fn new(args: ResizeConvUNetArgs, vb: VarBuilder, act: Act) -> Result<Self> {
        let start_channels = args.start_channels;
        let body = ResizeConvUNet::new(args, vb.pp("body"), act)?;
        let color_change_creator =
            ConvAct::new(vb.pp("color_change_creator"), start_channels, 4, act_tanh())?;
        let alpha_creator = ConvAct::new(vb.pp("alpha_creator"), start_channels, 4, act_sigmoid())?;
        let grid_change_creator =
            conv2d_no_bias_layer(vb.pp("grid_change_creator"), start_channels, 2, 3, 1, 1)?;
        Ok(Self {
            body,
            color_change_creator,
            alpha_creator,
            grid_change_creator,
            grid_change_applier: std::sync::Mutex::new(GridChangeApplier::new()),
        })
    }

    pub fn forward(
        &self,
        input_original_image: &Tensor,
        input_warped_image: &Tensor,
        input_grid_change: &Tensor,
        pose: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let (n, c) = pose.dims2()?;
        let pose = pose.reshape((n, c, 1, 1))?.broadcast_as((
            n,
            c,
            self.body.args.image_size,
            self.body.args.image_size,
        ))?;
        let feature = Tensor::cat(
            &[
                input_original_image,
                input_warped_image,
                input_grid_change,
                &pose,
            ],
            1,
        )?;
        let feature = self.body.forward(&feature)?;
        let feature = feature
            .last()
            .ok_or_else(|| candle::Error::Msg("missing feature".to_string()))?;
        let output_grid_change =
            input_grid_change.broadcast_add(&self.grid_change_creator.forward(feature)?)?;
        let output_color_change = self.color_change_creator.forward(feature)?;
        let output_color_change_alpha = self.alpha_creator.forward(feature)?;
        let output_warped_image = {
            let mut applier = self.grid_change_applier.lock().unwrap();
            applier.apply(&output_grid_change, input_original_image)?
        };
        let output_color_changed = apply_color_change(
            &output_color_change_alpha,
            &output_color_change,
            &output_warped_image,
        )?;
        Ok(vec![
            output_color_changed,
            output_color_change_alpha,
            output_color_change,
            output_warped_image,
            output_grid_change,
        ])
    }
}

pub struct TwoAlgoFaceBodyRotator05Separable {
    encoder_decoder: ResizeConvEncoderDecoderSeparable,
    direct_creator: ConvAct,
    grid_change_creator: Conv2d,
    grid_change_applier: std::sync::Mutex<GridChangeApplier>,
}

impl TwoAlgoFaceBodyRotator05Separable {
    pub const WARPED_IMAGE_INDEX: usize = 1;
    pub const GRID_CHANGE_INDEX: usize = 2;

    pub fn new(args: ResizeConvEncoderDecoderArgs, vb: VarBuilder, act: Act) -> Result<Self> {
        let start_channels = args.start_channels;
        let encoder_decoder =
            ResizeConvEncoderDecoderSeparable::new(args, vb.pp("encoder_decoder"), act)?;
        let direct_creator = ConvAct::new(vb.pp("direct_creator"), start_channels, 4, act_tanh())?;
        let grid_change_creator =
            conv2d_no_bias_layer(vb.pp("grid_change_creator"), start_channels, 2, 3, 1, 1)?;
        Ok(Self {
            encoder_decoder,
            direct_creator,
            grid_change_creator,
            grid_change_applier: std::sync::Mutex::new(GridChangeApplier::new()),
        })
    }

    pub fn forward(&self, image: &Tensor, pose: &Tensor) -> Result<Vec<Tensor>> {
        let (n, c) = pose.dims2()?;
        let pose = pose.reshape((n, c, 1, 1))?.broadcast_as((
            n,
            c,
            self.encoder_decoder.args.image_size,
            self.encoder_decoder.args.image_size,
        ))?;
        let feature = Tensor::cat(&[image, &pose], 1)?;
        let feature = self.encoder_decoder.forward(&feature)?;
        let feature = feature
            .last()
            .ok_or_else(|| candle::Error::Msg("missing feature".to_string()))?;
        let grid_change = self.grid_change_creator.forward(feature)?;
        let direct_image = self.direct_creator.forward(feature)?;
        let warped_image = {
            let mut applier = self.grid_change_applier.lock().unwrap();
            applier.apply(&grid_change, image)?
        };
        Ok(vec![direct_image, warped_image, grid_change])
    }
}

pub struct Editor07Separable {
    body: ResizeConvUNetSeparable,
    color_change_creator: ConvAct,
    alpha_creator: ConvAct,
    grid_change_creator: Conv2d,
    grid_change_applier: std::sync::Mutex<GridChangeApplier>,
}

impl Editor07Separable {
    pub fn new(args: ResizeConvUNetArgs, vb: VarBuilder, act: Act) -> Result<Self> {
        let start_channels = args.start_channels;
        let body = ResizeConvUNetSeparable::new(args, vb.pp("body"), act)?;
        let color_change_creator =
            ConvAct::new(vb.pp("color_change_creator"), start_channels, 4, act_tanh())?;
        let alpha_creator = ConvAct::new(vb.pp("alpha_creator"), start_channels, 4, act_sigmoid())?;
        let grid_change_creator =
            conv2d_no_bias_layer(vb.pp("grid_change_creator"), start_channels, 2, 3, 1, 1)?;
        Ok(Self {
            body,
            color_change_creator,
            alpha_creator,
            grid_change_creator,
            grid_change_applier: std::sync::Mutex::new(GridChangeApplier::new()),
        })
    }

    pub fn forward(
        &self,
        input_original_image: &Tensor,
        input_warped_image: &Tensor,
        input_grid_change: &Tensor,
        pose: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let (n, c) = pose.dims2()?;
        let pose = pose.reshape((n, c, 1, 1))?.broadcast_as((
            n,
            c,
            self.body.args.image_size,
            self.body.args.image_size,
        ))?;
        let feature = Tensor::cat(
            &[
                input_original_image,
                input_warped_image,
                input_grid_change,
                &pose,
            ],
            1,
        )?;
        let feature = self.body.forward(&feature)?;
        let feature = feature
            .last()
            .ok_or_else(|| candle::Error::Msg("missing feature".to_string()))?;
        let output_grid_change =
            input_grid_change.broadcast_add(&self.grid_change_creator.forward(feature)?)?;
        let output_color_change = self.color_change_creator.forward(feature)?;
        let output_color_change_alpha = self.alpha_creator.forward(feature)?;
        let output_warped_image = {
            let mut applier = self.grid_change_applier.lock().unwrap();
            applier.apply(&output_grid_change, input_original_image)?
        };
        let output_color_changed = apply_color_change(
            &output_color_change_alpha,
            &output_color_change,
            &output_warped_image,
        )?;
        Ok(vec![
            output_color_changed,
            output_color_change_alpha,
            output_color_change,
            output_warped_image,
            output_grid_change,
        ])
    }
}

pub fn build_standard_float_modules(
    model_dir: &std::path::Path,
    device: &candle::Device,
    dtype: DType,
) -> Result<StandardFloatModules> {
    let vb_ed = VarBuilder::from_pth(model_dir.join("eyebrow_decomposer.pt"), dtype, device)?;
    let vb_emc = VarBuilder::from_pth(
        model_dir.join("eyebrow_morphing_combiner.pt"),
        dtype,
        device,
    )?;
    let vb_fm = VarBuilder::from_pth(model_dir.join("face_morpher.pt"), dtype, device)?;
    let vb_rot = VarBuilder::from_pth(
        model_dir.join("two_algo_face_body_rotator.pt"),
        dtype,
        device,
    )?;
    let vb_edit = VarBuilder::from_pth(model_dir.join("editor.pt"), dtype, device)?;

    let eyebrow_decomposer = EyebrowDecomposer00::new(
        PoserEncoderDecoder00Args {
            image_size: 128,
            input_image_channels: 4,
            output_image_channels: 4,
            num_pose_params: 0,
            start_channels: 64,
            bottleneck_image_size: 16,
            num_bottleneck_blocks: 6,
            max_channels: 512,
        },
        vb_ed,
    )?;

    let eyebrow_morphing_combiner = EyebrowMorphingCombiner00::new(
        PoserEncoderDecoder00Args {
            image_size: 128,
            input_image_channels: 8,
            output_image_channels: 4,
            num_pose_params: 12,
            start_channels: 64,
            bottleneck_image_size: 16,
            num_bottleneck_blocks: 6,
            max_channels: 512,
        },
        vb_emc,
    )?;

    let face_morpher = FaceMorpher08::new(
        FaceMorpher08Args {
            image_size: 192,
            image_channels: 4,
            num_expression_params: 27,
            start_channels: 64,
            bottleneck_image_size: 24,
            num_bottleneck_blocks: 6,
            max_channels: 512,
        },
        vb_fm,
        Act::Relu,
    )?;

    let rotator = TwoAlgoFaceBodyRotator05::new(
        ResizeConvEncoderDecoderArgs {
            image_size: 256,
            input_channels: 4 + 6,
            start_channels: 64,
            bottleneck_image_size: 32,
            num_bottleneck_blocks: 6,
            max_channels: 512,
            upsample_mode: UpsampleMode::Nearest,
        },
        vb_rot,
        Act::LeakyRelu(0.1),
    )?;

    let editor = Editor07::new(
        ResizeConvUNetArgs {
            image_size: 512,
            input_channels: 2 * 4 + 6 + 2,
            start_channels: 32,
            bottleneck_image_size: 64,
            num_bottleneck_blocks: 6,
            max_channels: 512,
            upsample_mode: UpsampleMode::Nearest,
        },
        vb_edit,
        Act::LeakyRelu(0.1),
    )?;

    Ok(StandardFloatModules {
        eyebrow_decomposer,
        eyebrow_morphing_combiner,
        face_morpher,
        rotator,
        editor,
    })
}

pub fn build_separable_modules(
    model_dir: &std::path::Path,
    device: &candle::Device,
    dtype: DType,
) -> Result<SeparableModules> {
    let vb_ed = VarBuilder::from_pth(model_dir.join("eyebrow_decomposer.pt"), dtype, device)?;
    let vb_emc = VarBuilder::from_pth(
        model_dir.join("eyebrow_morphing_combiner.pt"),
        dtype,
        device,
    )?;
    let vb_fm = VarBuilder::from_pth(model_dir.join("face_morpher.pt"), dtype, device)?;
    let vb_rot = VarBuilder::from_pth(
        model_dir.join("two_algo_face_body_rotator.pt"),
        dtype,
        device,
    )?;
    let vb_edit = VarBuilder::from_pth(model_dir.join("editor.pt"), dtype, device)?;

    let eyebrow_decomposer = EyebrowDecomposer03::new(
        PoserEncoderDecoder00Args {
            image_size: 128,
            input_image_channels: 4,
            output_image_channels: 4,
            num_pose_params: 0,
            start_channels: 64,
            bottleneck_image_size: 16,
            num_bottleneck_blocks: 6,
            max_channels: 512,
        },
        vb_ed,
    )?;

    let eyebrow_morphing_combiner = EyebrowMorphingCombiner03::new(
        PoserEncoderDecoder00Args {
            image_size: 128,
            input_image_channels: 8,
            output_image_channels: 4,
            num_pose_params: 12,
            start_channels: 64,
            bottleneck_image_size: 16,
            num_bottleneck_blocks: 6,
            max_channels: 512,
        },
        vb_emc,
    )?;

    let face_morpher = FaceMorpher09::new(
        FaceMorpher08Args {
            image_size: 192,
            image_channels: 4,
            num_expression_params: 27,
            start_channels: 64,
            bottleneck_image_size: 24,
            num_bottleneck_blocks: 6,
            max_channels: 512,
        },
        vb_fm,
        Act::Relu,
    )?;

    let rotator = TwoAlgoFaceBodyRotator05Separable::new(
        ResizeConvEncoderDecoderArgs {
            image_size: 256,
            input_channels: 4 + 6,
            start_channels: 64,
            bottleneck_image_size: 32,
            num_bottleneck_blocks: 6,
            max_channels: 512,
            upsample_mode: UpsampleMode::Nearest,
        },
        vb_rot,
        Act::LeakyRelu(0.1),
    )?;

    let editor = Editor07Separable::new(
        ResizeConvUNetArgs {
            image_size: 512,
            input_channels: 2 * 4 + 6 + 2,
            start_channels: 32,
            bottleneck_image_size: 64,
            num_bottleneck_blocks: 6,
            max_channels: 512,
            upsample_mode: UpsampleMode::Nearest,
        },
        vb_edit,
        Act::LeakyRelu(0.1),
    )?;

    Ok(SeparableModules {
        eyebrow_decomposer,
        eyebrow_morphing_combiner,
        face_morpher,
        rotator,
        editor,
    })
}

pub struct StandardFloatModules {
    pub eyebrow_decomposer: EyebrowDecomposer00,
    pub eyebrow_morphing_combiner: EyebrowMorphingCombiner00,
    pub face_morpher: FaceMorpher08,
    pub rotator: TwoAlgoFaceBodyRotator05,
    pub editor: Editor07,
}

pub struct SeparableModules {
    pub eyebrow_decomposer: EyebrowDecomposer03,
    pub eyebrow_morphing_combiner: EyebrowMorphingCombiner03,
    pub face_morpher: FaceMorpher09,
    pub rotator: TwoAlgoFaceBodyRotator05Separable,
    pub editor: Editor07Separable,
}
