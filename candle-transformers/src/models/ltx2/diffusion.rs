use candle::{DType, IndexOp, Result, Tensor};

use super::utils::{to_denoised, to_velocity};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PerturbationType {
    SkipVideoSelfAttn,
    SkipAudioSelfAttn,
    SkipA2vCrossAttn,
    SkipV2aCrossAttn,
}

#[derive(Debug, Clone)]
pub struct BatchedPerturbationConfig;

impl BatchedPerturbationConfig {
    pub fn empty() -> Self {
        Self
    }

    pub fn all_in_batch(&self, _pt: PerturbationType, _idx: usize) -> bool {
        false
    }

    pub fn mask_like(&self, _pt: PerturbationType, _idx: usize, x: &Tensor) -> Result<Tensor> {
        Tensor::ones_like(x)
    }
}

pub struct EulerDiffusionStep;

impl EulerDiffusionStep {
    pub fn step(
        &self,
        sample: &Tensor,
        denoised: &Tensor,
        sigmas: &Tensor,
        idx: usize,
    ) -> Result<Tensor> {
        let sigma = sigmas.i(idx)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        let sigma_next = sigmas
            .i(idx + 1)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()? as f64;
        let dt = sigma_next - sigma;
        let velocity = to_velocity(sample, sigma, denoised)?;
        let out = (sample.to_dtype(DType::F32)? + (velocity.to_dtype(DType::F32)? * dt)?)?;
        out.to_dtype(sample.dtype())
    }
}

pub struct LatentDenoiser;

impl LatentDenoiser {
    pub fn denoise(sample: &Tensor, velocity: &Tensor, sigma: f64) -> Result<Tensor> {
        to_denoised(sample, velocity, sigma)
    }
}
