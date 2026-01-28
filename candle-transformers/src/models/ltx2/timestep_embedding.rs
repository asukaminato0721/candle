use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

pub fn get_timestep_embedding(
    timesteps: &Tensor,
    embedding_dim: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
    scale: f64,
    max_period: f64,
) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let device = timesteps.device();
    let exponent = Tensor::arange(0u32, half_dim as u32, device)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&Tensor::from_vec(
            vec![-max_period.ln() as f32],
            (1,),
            device,
        )?)?;
    let denom = (half_dim as f64 - downscale_freq_shift) as f32;
    let exponent = exponent.broadcast_div(&Tensor::from_vec(vec![denom], (1,), device)?)?;
    let emb = exponent.exp()?; // [half_dim]
    let emb = timesteps
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&emb.unsqueeze(0)?)?;
    let emb = (emb * scale)?;
    let sin = emb.sin()?;
    let cos = emb.cos()?;
    let emb = Tensor::cat(&[&sin, &cos], D::Minus1)?;
    let emb = if flip_sin_to_cos {
        let (b, d) = emb.dims2()?;
        let half = d / 2;
        let right = emb.narrow(1, half, half)?;
        let left = emb.narrow(1, 0, half)?;
        Tensor::cat(&[&right, &left], D::Minus1)?
    } else {
        emb
    };
    if embedding_dim % 2 == 1 {
        let pad = Tensor::zeros((emb.dim(0)?, 1), emb.dtype(), device)?;
        Tensor::cat(&[&emb, &pad], D::Minus1)
    } else {
        Ok(emb)
    }
}

pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(
        in_channels: usize,
        time_embed_dim: usize,
        out_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear_1 = linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let out = out_dim.unwrap_or(time_embed_dim);
        let linear_2 = linear(time_embed_dim, out, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, sample: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        let mut x = sample.clone();
        if let Some(cond) = condition {
            x = (&x + cond)?;
        }
        let x = self.linear_1.forward(&x)?;
        let x = candle_nn::Activation::Silu.forward(&x)?;
        self.linear_2.forward(&x)
    }
}

pub struct Timesteps {
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
    scale: f64,
}

impl Timesteps {
    pub fn new(
        num_channels: usize,
        flip_sin_to_cos: bool,
        downscale_freq_shift: f64,
        scale: f64,
    ) -> Self {
        Self {
            num_channels,
            flip_sin_to_cos,
            downscale_freq_shift,
            scale,
        }
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        get_timestep_embedding(
            timesteps,
            self.num_channels,
            self.flip_sin_to_cos,
            self.downscale_freq_shift,
            self.scale,
            10000.0,
        )
    }
}

pub struct PixArtAlphaCombinedTimestepSizeEmbeddings {
    time_proj: Timesteps,
    timestep_embedder: TimestepEmbedding,
}

impl PixArtAlphaCombinedTimestepSizeEmbeddings {
    pub fn new(embedding_dim: usize, _size_emb_dim: usize, vb: VarBuilder) -> Result<Self> {
        let time_proj = Timesteps::new(256, true, 0.0, 1.0);
        let timestep_embedder = TimestepEmbedding::new(
            256,
            embedding_dim,
            Some(embedding_dim),
            vb.pp("timestep_embedder"),
        )?;
        Ok(Self {
            time_proj,
            timestep_embedder,
        })
    }

    pub fn forward(&self, timestep: &Tensor, hidden_dtype: DType) -> Result<Tensor> {
        let timesteps_proj = self.time_proj.forward(timestep)?;
        let timesteps_proj = timesteps_proj.to_dtype(hidden_dtype)?;
        self.timestep_embedder.forward(&timesteps_proj, None)
    }
}
