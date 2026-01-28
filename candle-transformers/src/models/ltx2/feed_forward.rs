use candle::{Module, Result, Tensor};
use candle_nn::{linear, Activation, Linear, VarBuilder};

pub struct GeluApprox {
    proj: Linear,
}

impl GeluApprox {
    pub fn new(dim_in: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear(dim_in, dim_out, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(x)?;
        Activation::GeluPytorchTanh.forward(&x)
    }
}

pub struct FeedForward {
    proj: GeluApprox,
    out: Linear,
}

impl FeedForward {
    pub fn new(dim: usize, dim_out: usize, mult: usize, vb: VarBuilder) -> Result<Self> {
        let inner_dim = dim * mult;
        let proj = GeluApprox::new(dim, inner_dim, vb.pp("net").pp("0"))?;
        let out = linear(inner_dim, dim_out, vb.pp("net").pp("2"))?;
        Ok(Self { proj, out })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(x)?;
        self.out.forward(&x)
    }
}
