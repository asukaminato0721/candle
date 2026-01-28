use candle::{DType, Module, Result, Tensor};
use candle_nn::{linear, Activation, Linear, VarBuilder};

use super::timestep_embedding::PixArtAlphaCombinedTimestepSizeEmbeddings;

pub struct AdaLayerNormSingle {
    emb: PixArtAlphaCombinedTimestepSizeEmbeddings,
    linear: Linear,
}

impl AdaLayerNormSingle {
    pub fn new(embedding_dim: usize, embedding_coefficient: usize, vb: VarBuilder) -> Result<Self> {
        let emb = PixArtAlphaCombinedTimestepSizeEmbeddings::new(
            embedding_dim,
            embedding_dim / 3,
            vb.pp("emb"),
        )?;
        let linear = linear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            vb.pp("linear"),
        )?;
        Ok(Self { emb, linear })
    }

    pub fn forward(&self, timestep: &Tensor, hidden_dtype: DType) -> Result<(Tensor, Tensor)> {
        let embedded = self.emb.forward(timestep, hidden_dtype)?;
        let silu = Activation::Silu.forward(&embedded)?;
        let out = self.linear.forward(&silu)?;
        Ok((out, embedded))
    }
}
