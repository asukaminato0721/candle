use candle::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct VideoPixelShape {
    pub batch: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub fps: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SpatioTemporalScaleFactors {
    pub time: usize,
    pub width: usize,
    pub height: usize,
}

impl SpatioTemporalScaleFactors {
    pub const fn default() -> Self {
        Self {
            time: 8,
            width: 32,
            height: 32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VideoLatentShape {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl VideoLatentShape {
    pub fn to_dims(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.batch,
            self.channels,
            self.frames,
            self.height,
            self.width,
        )
    }

    pub fn mask_shape(&self) -> Self {
        Self {
            channels: 1,
            ..*self
        }
    }

    pub fn from_pixel_shape(
        shape: VideoPixelShape,
        latent_channels: usize,
        scale_factors: SpatioTemporalScaleFactors,
    ) -> Self {
        let frames = (shape.frames.saturating_sub(1)) / scale_factors.time + 1;
        let height = shape.height / scale_factors.height;
        let width = shape.width / scale_factors.width;
        Self {
            batch: shape.batch,
            channels: latent_channels,
            frames,
            height,
            width,
        }
    }

    pub fn upscale(&self, scale_factors: SpatioTemporalScaleFactors) -> Self {
        Self {
            batch: self.batch,
            channels: 3,
            frames: (self.frames.saturating_sub(1)) * scale_factors.time + 1,
            height: self.height * scale_factors.height,
            width: self.width * scale_factors.width,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioLatentShape {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub mel_bins: usize,
}

impl AudioLatentShape {
    pub fn to_dims(&self) -> (usize, usize, usize, usize) {
        (self.batch, self.channels, self.frames, self.mel_bins)
    }

    pub fn mask_shape(&self) -> Self {
        Self {
            channels: 1,
            mel_bins: 1,
            ..*self
        }
    }

    pub fn from_duration(
        batch: usize,
        duration: f64,
        channels: usize,
        mel_bins: usize,
        sample_rate: usize,
        hop_length: usize,
        audio_latent_downsample_factor: usize,
    ) -> Self {
        let latents_per_second =
            sample_rate as f64 / hop_length as f64 / audio_latent_downsample_factor as f64;
        Self {
            batch,
            channels,
            frames: (duration * latents_per_second).round() as usize,
            mel_bins,
        }
    }

    pub fn from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: usize,
        mel_bins: usize,
        sample_rate: usize,
        hop_length: usize,
        audio_latent_downsample_factor: usize,
    ) -> Self {
        let duration = shape.frames as f64 / shape.fps;
        Self::from_duration(
            shape.batch,
            duration,
            channels,
            mel_bins,
            sample_rate,
            hop_length,
            audio_latent_downsample_factor,
        )
    }
}

#[derive(Debug, Clone)]
pub struct LatentState {
    pub latent: Tensor,
    pub denoise_mask: Tensor,
    pub positions: Tensor,
    pub clean_latent: Tensor,
}

impl LatentState {
    pub fn clone_state(&self) -> candle::Result<Self> {
        Ok(Self {
            latent: self.latent.clone(),
            denoise_mask: self.denoise_mask.clone(),
            positions: self.positions.clone(),
            clean_latent: self.clean_latent.clone(),
        })
    }
}
