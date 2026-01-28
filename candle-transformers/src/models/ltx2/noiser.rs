use candle::{Result, Tensor};

use super::types::LatentState;

pub struct GaussianNoiser;

impl GaussianNoiser {
    pub fn new() -> Self {
        Self
    }

    pub fn apply(&self, latent_state: &LatentState, noise_scale: f64) -> Result<LatentState> {
        let noise = latent_state.latent.randn_like(0.0, 1.0)?;
        let scaled_mask = (latent_state.denoise_mask.clone() * noise_scale)?;
        let one = Tensor::ones_like(&scaled_mask)?;
        let inv = (&one - &scaled_mask)?;
        let latent =
            (noise.broadcast_mul(&scaled_mask)? + latent_state.latent.broadcast_mul(&inv)?)?;
        Ok(LatentState {
            latent,
            denoise_mask: latent_state.denoise_mask.clone(),
            positions: latent_state.positions.clone(),
            clean_latent: latent_state.clean_latent.clone(),
        })
    }
}
