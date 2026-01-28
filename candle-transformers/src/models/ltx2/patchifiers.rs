use candle::{DType, Device, IndexOp, Result, Tensor, D};

use super::types::{AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape};

pub struct VideoLatentPatchifier {
    patch_size: (usize, usize, usize),
}

impl VideoLatentPatchifier {
    pub fn new(patch_size: usize) -> Self {
        Self {
            patch_size: (1, patch_size, patch_size),
        }
    }

    pub fn patch_size(&self) -> (usize, usize, usize) {
        self.patch_size
    }

    pub fn get_token_count(&self, tgt_shape: VideoLatentShape) -> usize {
        let (p_t, p_h, p_w) = self.patch_size;
        let frames = tgt_shape.frames / p_t;
        let height = tgt_shape.height / p_h;
        let width = tgt_shape.width / p_w;
        frames * height * width
    }

    pub fn patchify(&self, latents: &Tensor) -> Result<Tensor> {
        let (b, c, f, h, w) = latents.dims5()?;
        let (p_t, p_h, p_w) = self.patch_size;
        let f2 = f / p_t;
        let h2 = h / p_h;
        let w2 = w / p_w;
        let latents = latents.reshape(vec![b, c, f2, p_t, h2, p_h, w2, p_w])?;
        let latents = latents.permute(vec![0, 2, 4, 6, 1, 3, 5, 7])?;
        latents.reshape((b, f2 * h2 * w2, c * p_t * p_h * p_w))
    }

    pub fn unpatchify(&self, latents: &Tensor, output_shape: VideoLatentShape) -> Result<Tensor> {
        let (p_t, p_h, p_w) = self.patch_size;
        let frames = output_shape.frames / p_t;
        let height = output_shape.height / p_h;
        let width = output_shape.width / p_w;
        let latents = latents.reshape(vec![
            output_shape.batch,
            frames,
            height,
            width,
            output_shape.channels,
            p_t,
            p_h,
            p_w,
        ])?;
        let latents = latents.permute(vec![0, 4, 1, 5, 2, 6, 3, 7])?;
        latents.reshape((
            output_shape.batch,
            output_shape.channels,
            frames * p_t,
            height * p_h,
            width * p_w,
        ))
    }

    pub fn get_patch_grid_bounds(
        &self,
        output_shape: VideoLatentShape,
        device: &Device,
    ) -> Result<Tensor> {
        let (p_t, p_h, p_w) = self.patch_size;
        let frames = output_shape.frames;
        let height = output_shape.height;
        let width = output_shape.width;
        let batch = output_shape.batch;

        let frames = Tensor::arange_step(0f32, frames as f32, p_t as f32, device)?;
        let height = Tensor::arange_step(0f32, height as f32, p_h as f32, device)?;
        let width = Tensor::arange_step(0f32, width as f32, p_w as f32, device)?;
        let grids = Tensor::meshgrid(&[&frames, &height, &width], false)?;
        let patch_starts = Tensor::stack(&grids, 0)?;

        let patch_size = Tensor::new(&[p_t as f32, p_h as f32, p_w as f32], device)?;
        let patch_size = patch_size.reshape((3, 1, 1, 1))?;
        let patch_ends = patch_starts.broadcast_add(&patch_size)?;
        let coords = Tensor::stack(&[patch_starts, patch_ends], D::Minus1)?;
        let coords = coords.unsqueeze(0)?;
        let (grid_f, grid_h, grid_w) = grids[0].dims3()?;
        let coords = coords.broadcast_as((batch, 3, grid_f, grid_h, grid_w, 2))?;
        coords.reshape((batch, 3, grid_f * grid_h * grid_w, 2))
    }
}

pub fn get_pixel_coords(
    latent_coords: &Tensor,
    scale_factors: SpatioTemporalScaleFactors,
    causal_fix: bool,
) -> Result<Tensor> {
    let scale = Tensor::new(
        &[
            scale_factors.time as f32,
            scale_factors.height as f32,
            scale_factors.width as f32,
        ],
        latent_coords.device(),
    )?
    .reshape((1, 3, 1, 1))?;
    let coords = latent_coords.broadcast_mul(&scale)?;
    if !causal_fix {
        return Ok(coords);
    }
    let time = coords.i((.., 0, .., ..))?;
    let offset = 1.0 - scale_factors.time as f64;
    let time = (time + offset)?.clamp(0f64, f64::MAX)?;
    let height = coords.i((.., 1, .., ..))?;
    let width = coords.i((.., 2, .., ..))?;
    Tensor::stack(&[time, height, width], 1)
}

pub struct AudioPatchifier {
    patch_size: (usize, usize, usize),
    sample_rate: usize,
    hop_length: usize,
    audio_latent_downsample_factor: usize,
    is_causal: bool,
    shift: usize,
}

impl AudioPatchifier {
    pub fn new(
        patch_size: usize,
        sample_rate: usize,
        hop_length: usize,
        audio_latent_downsample_factor: usize,
        is_causal: bool,
        shift: usize,
    ) -> Self {
        Self {
            patch_size: (1, patch_size, patch_size),
            sample_rate,
            hop_length,
            audio_latent_downsample_factor,
            is_causal,
            shift,
        }
    }

    pub fn patch_size(&self) -> (usize, usize, usize) {
        self.patch_size
    }

    pub fn get_token_count(&self, tgt_shape: AudioLatentShape) -> usize {
        tgt_shape.frames
    }

    fn get_audio_latent_time_in_sec(
        &self,
        start: usize,
        end: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let frames = Tensor::arange(start as f32, end as f32, device)?;
        let audio_mel_frame = (frames * self.audio_latent_downsample_factor as f64)?;
        let audio_mel_frame = if self.is_causal {
            let offset = 1.0 - self.audio_latent_downsample_factor as f64;
            (audio_mel_frame + offset)?.clamp(0f64, f64::MAX)?
        } else {
            audio_mel_frame
        };
        let scale = self.hop_length as f64 / self.sample_rate as f64;
        audio_mel_frame * scale
    }

    fn compute_audio_timings(&self, batch: usize, steps: usize, device: &Device) -> Result<Tensor> {
        let start_timings =
            self.get_audio_latent_time_in_sec(self.shift, steps + self.shift, device)?;
        let start_timings = start_timings
            .unsqueeze(0)?
            .broadcast_as((batch, steps))?
            .unsqueeze(1)?;
        let end_timings =
            self.get_audio_latent_time_in_sec(self.shift + 1, steps + self.shift + 1, device)?;
        let end_timings = end_timings
            .unsqueeze(0)?
            .broadcast_as((batch, steps))?
            .unsqueeze(1)?;
        Tensor::stack(&[start_timings, end_timings], D::Minus1)
    }

    pub fn patchify(&self, audio_latents: &Tensor) -> Result<Tensor> {
        let (b, c, t, f) = audio_latents.dims4()?;
        let audio_latents = audio_latents.permute((0, 2, 1, 3))?;
        audio_latents.reshape((b, t, c * f))
    }

    pub fn unpatchify(
        &self,
        audio_latents: &Tensor,
        output_shape: AudioLatentShape,
    ) -> Result<Tensor> {
        let audio_latents = audio_latents.reshape((
            output_shape.batch,
            output_shape.frames,
            output_shape.channels,
            output_shape.mel_bins,
        ))?;
        audio_latents.permute((0, 2, 1, 3))
    }

    pub fn get_patch_grid_bounds(
        &self,
        output_shape: AudioLatentShape,
        device: &Device,
    ) -> Result<Tensor> {
        self.compute_audio_timings(output_shape.batch, output_shape.frames, device)
    }
}

pub fn cast_positions_dtype(positions: &Tensor, dtype: DType) -> Result<Tensor> {
    if positions.dtype() == dtype {
        Ok(positions.clone())
    } else {
        positions.to_dtype(dtype)
    }
}
