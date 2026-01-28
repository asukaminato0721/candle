use candle::{Module, Result, Tensor};
use candle_nn::{linear, Activation, Linear, VarBuilder};

pub struct PixArtAlphaTextProjection {
    linear_1: Linear,
    linear_2: Linear,
    act: Activation,
}

impl PixArtAlphaTextProjection {
    pub fn new(
        in_features: usize,
        hidden_size: usize,
        out_features: Option<usize>,
        act_fn: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_features = out_features.unwrap_or(hidden_size);
        let linear_1 = linear(in_features, hidden_size, vb.pp("linear_1"))?;
        let linear_2 = linear(hidden_size, out_features, vb.pp("linear_2"))?;
        let act = match act_fn {
            "silu" => Activation::Silu,
            _ => Activation::GeluPytorchTanh,
        };
        Ok(Self {
            linear_1,
            linear_2,
            act,
        })
    }

    pub fn forward(&self, caption: &Tensor) -> Result<Tensor> {
        let hidden = self.linear_1.forward(caption)?;
        let hidden = self.act.forward(&hidden)?;
        self.linear_2.forward(&hidden)
    }
}
