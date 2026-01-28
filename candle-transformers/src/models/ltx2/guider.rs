use candle::{Result, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct CfgGuider {
    pub scale: f64,
}

impl CfgGuider {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }

    pub fn enabled(&self) -> bool {
        self.scale != 1.0
    }

    pub fn delta(&self, cond: &Tensor, uncond: &Tensor) -> Result<Tensor> {
        (cond - uncond)? * (self.scale - 1.0)
    }
}
