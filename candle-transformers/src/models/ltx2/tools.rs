use candle::{DType, Device, IndexOp, Result, Tensor};

use super::patchifiers::{
    cast_positions_dtype, get_pixel_coords, AudioPatchifier, VideoLatentPatchifier,
};
use super::types::{
    AudioLatentShape, LatentState, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape,
};

pub struct VideoLatentTools {
    pub patchifier: VideoLatentPatchifier,
    pub target_shape: VideoLatentShape,
    pub fps: f64,
    pub scale_factors: SpatioTemporalScaleFactors,
    pub causal_fix: bool,
}

impl VideoLatentTools {
    pub fn new(patch_size: usize, target_shape: VideoLatentShape, fps: f64) -> Self {
        Self {
            patchifier: VideoLatentPatchifier::new(patch_size),
            target_shape,
            fps,
            scale_factors: SpatioTemporalScaleFactors::default(),
            causal_fix: true,
        }
    }

    pub fn from_pixel_shape(
        patch_size: usize,
        pixel_shape: VideoPixelShape,
        latent_channels: usize,
    ) -> Self {
        let target_shape = VideoLatentShape::from_pixel_shape(
            pixel_shape,
            latent_channels,
            SpatioTemporalScaleFactors::default(),
        );
        Self::new(patch_size, target_shape, pixel_shape.fps)
    }

    pub fn create_initial_state(
        &self,
        device: &Device,
        dtype: DType,
        initial_latent: Option<Tensor>,
    ) -> Result<LatentState> {
        let latent = if let Some(latent) = initial_latent {
            latent
        } else {
            Tensor::zeros(self.target_shape.to_dims(), dtype, device)?
        };
        let clean_latent = latent.clone();
        let denoise_mask =
            Tensor::ones(self.target_shape.mask_shape().to_dims(), DType::F32, device)?;

        let coords = self
            .patchifier
            .get_patch_grid_bounds(self.target_shape, device)?;
        let coords = get_pixel_coords(&coords, self.scale_factors, self.causal_fix)?;
        let time = (coords.i((.., 0, .., ..))? / self.fps)?;
        let height = coords.i((.., 1, .., ..))?;
        let width = coords.i((.., 2, .., ..))?;
        let positions = Tensor::stack(&[time, height, width], 1)?;
        let positions = cast_positions_dtype(&positions, dtype)?;

        self.patchify(&LatentState {
            latent,
            denoise_mask,
            positions,
            clean_latent,
        })
    }

    pub fn patchify(&self, state: &LatentState) -> Result<LatentState> {
        let latent = self.patchifier.patchify(&state.latent)?;
        let clean_latent = self.patchifier.patchify(&state.clean_latent)?;
        let denoise_mask = self.patchifier.patchify(&state.denoise_mask)?;
        Ok(LatentState {
            latent,
            denoise_mask,
            positions: state.positions.clone(),
            clean_latent,
        })
    }

    pub fn unpatchify(&self, state: &LatentState) -> Result<LatentState> {
        let latent = self
            .patchifier
            .unpatchify(&state.latent, self.target_shape)?;
        let clean_latent = self
            .patchifier
            .unpatchify(&state.clean_latent, self.target_shape)?;
        let denoise_mask = self
            .patchifier
            .unpatchify(&state.denoise_mask, self.target_shape.mask_shape())?;
        Ok(LatentState {
            latent,
            denoise_mask,
            positions: state.positions.clone(),
            clean_latent,
        })
    }

    pub fn clear_conditioning(&self, state: &LatentState) -> Result<LatentState> {
        let tokens = self.patchifier.get_token_count(self.target_shape);
        let latent = state.latent.narrow(1, 0, tokens)?;
        let clean_latent = state.clean_latent.narrow(1, 0, tokens)?;
        let denoise_mask = state.denoise_mask.narrow(1, 0, tokens)?;
        let positions = state.positions.narrow(2, 0, tokens)?;
        Ok(LatentState {
            latent,
            denoise_mask,
            positions,
            clean_latent,
        })
    }
}

pub struct AudioLatentTools {
    pub patchifier: AudioPatchifier,
    pub target_shape: AudioLatentShape,
}

impl AudioLatentTools {
    pub fn new(
        patch_size: usize,
        target_shape: AudioLatentShape,
        sample_rate: usize,
        hop_length: usize,
        audio_latent_downsample_factor: usize,
        is_causal: bool,
    ) -> Self {
        Self {
            patchifier: AudioPatchifier::new(
                patch_size,
                sample_rate,
                hop_length,
                audio_latent_downsample_factor,
                is_causal,
                0,
            ),
            target_shape,
        }
    }

    pub fn create_initial_state(
        &self,
        device: &Device,
        dtype: DType,
        initial_latent: Option<Tensor>,
    ) -> Result<LatentState> {
        let latent = if let Some(latent) = initial_latent {
            latent
        } else {
            Tensor::zeros(self.target_shape.to_dims(), dtype, device)?
        };
        let clean_latent = latent.clone();
        let denoise_mask =
            Tensor::ones(self.target_shape.mask_shape().to_dims(), DType::F32, device)?;
        let positions = self
            .patchifier
            .get_patch_grid_bounds(self.target_shape, device)?;
        let positions = cast_positions_dtype(&positions, dtype)?;
        self.patchify(&LatentState {
            latent,
            denoise_mask,
            positions,
            clean_latent,
        })
    }

    pub fn patchify(&self, state: &LatentState) -> Result<LatentState> {
        let latent = self.patchifier.patchify(&state.latent)?;
        let clean_latent = self.patchifier.patchify(&state.clean_latent)?;
        let denoise_mask = self.patchifier.patchify(&state.denoise_mask)?;
        Ok(LatentState {
            latent,
            denoise_mask,
            positions: state.positions.clone(),
            clean_latent,
        })
    }

    pub fn unpatchify(&self, state: &LatentState) -> Result<LatentState> {
        let latent = self
            .patchifier
            .unpatchify(&state.latent, self.target_shape)?;
        let clean_latent = self
            .patchifier
            .unpatchify(&state.clean_latent, self.target_shape)?;
        let denoise_mask = self
            .patchifier
            .unpatchify(&state.denoise_mask, self.target_shape.mask_shape())?;
        Ok(LatentState {
            latent,
            denoise_mask,
            positions: state.positions.clone(),
            clean_latent,
        })
    }

    pub fn clear_conditioning(&self, state: &LatentState) -> Result<LatentState> {
        let tokens = self.patchifier.get_token_count(self.target_shape);
        let latent = state.latent.narrow(1, 0, tokens)?;
        let clean_latent = state.clean_latent.narrow(1, 0, tokens)?;
        let denoise_mask = state.denoise_mask.narrow(1, 0, tokens)?;
        let positions = state.positions.narrow(2, 0, tokens)?;
        Ok(LatentState {
            latent,
            denoise_mask,
            positions,
            clean_latent,
        })
    }
}
