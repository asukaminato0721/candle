use candle::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

pub struct GemmaFeaturesExtractorProjLinear {
    aggregate_embed: Linear,
}

impl GemmaFeaturesExtractorProjLinear {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let aggregate_embed = linear(3840 * 49, 3840, vb.pp("aggregate_embed"))?;
        Ok(Self { aggregate_embed })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.aggregate_embed.forward(x)
    }
}
