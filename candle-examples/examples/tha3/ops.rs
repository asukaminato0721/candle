use candle::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor, WithDType};

#[cfg(feature = "cuda")]
mod cuda_kernels {
    include!(concat!(env!("OUT_DIR"), "/tha3_cuda_kernels.rs"));
}

pub fn apply_color_change(alpha: &Tensor, color_change: &Tensor, image: &Tensor) -> Result<Tensor> {
    let one = alpha.ones_like()?;
    let inv_alpha = one.broadcast_sub(alpha)?;
    let color = color_change.broadcast_mul(alpha)?;
    let base = image.broadcast_mul(&inv_alpha)?;
    color.broadcast_add(&base)
}

pub fn apply_rgb_change(alpha: &Tensor, color_change: &Tensor, image: &Tensor) -> Result<Tensor> {
    let image_rgb = image.narrow(1, 0, 3)?;
    let image_a = image.narrow(1, 3, 1)?;
    let color_rgb = color_change.narrow(1, 0, 3)?;
    let one = alpha.ones_like()?;
    let inv_alpha = one.broadcast_sub(alpha)?;
    let out_rgb = color_rgb
        .broadcast_mul(alpha)?
        .broadcast_add(&image_rgb.broadcast_mul(&inv_alpha)?)?;
    Tensor::cat(&[&out_rgb, &image_a], 1)
}

pub fn grid_sample(image: &Tensor, grid: &Tensor, align_corners: bool) -> Result<Tensor> {
    if image.dtype() == DType::F32 && grid.dtype() == DType::F32 {
        image.apply_op2(grid, GridSampleOp { align_corners })
    } else {
        let image_f32 = image.to_dtype(DType::F32)?;
        let grid_f32 = grid.to_dtype(DType::F32)?;
        let out = image_f32.apply_op2(&grid_f32, GridSampleOp { align_corners })?;
        out.to_dtype(image.dtype())
    }
}

#[derive(Clone, Copy)]
struct GridSampleOp {
    align_corners: bool,
}

impl CustomOp2 for GridSampleOp {
    fn name(&self) -> &'static str {
        "grid-sample"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // image: (n, c, h, w), grid: (n, h_out, w_out, 2)
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle::bail!("grid_sample expects contiguous tensors");
        }
        let (n, c, h, w) = l1.shape().dims4()?;
        let (n2, h_out, w_out, two) = l2.shape().dims4()?;
        if n != n2 || two != 2 {
            candle::bail!(
                "grid_sample unexpected shapes image {:?} grid {:?}",
                l1.shape(),
                l2.shape()
            );
        }
        let img = match s1 {
            CpuStorage::F32(v) => v,
            _ => candle::bail!("grid_sample only supports f32 on cpu"),
        };
        let grid = match s2 {
            CpuStorage::F32(v) => v,
            _ => candle::bail!("grid_sample only supports f32 on cpu"),
        };
        let (img_o1, img_o2) = l1
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("non-contiguous image".to_string()))?;
        let (grid_o1, grid_o2) = l2
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("non-contiguous grid".to_string()))?;
        let img = &img[img_o1..img_o2];
        let grid = &grid[grid_o1..grid_o2];

        let mut dst = vec![0f32; n * c * h_out * w_out];
        let w_f = w as f32;
        let h_f = h as f32;

        for bn in 0..n {
            for oy in 0..h_out {
                for ox in 0..w_out {
                    let grid_idx = (((bn * h_out + oy) * w_out + ox) * 2) as usize;
                    let gx = grid[grid_idx];
                    let gy = grid[grid_idx + 1];

                    let (sx, sy) = if self.align_corners {
                        (
                            (gx + 1.0) * 0.5 * (w_f - 1.0),
                            (gy + 1.0) * 0.5 * (h_f - 1.0),
                        )
                    } else {
                        (
                            ((gx + 1.0) * w_f - 1.0) * 0.5,
                            ((gy + 1.0) * h_f - 1.0) * 0.5,
                        )
                    };

                    let sx = sx.clamp(0.0, w_f - 1.0);
                    let sy = sy.clamp(0.0, h_f - 1.0);

                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = (x0 + 1).min(w - 1);
                    let y1 = (y0 + 1).min(h - 1);
                    let dx = sx - x0 as f32;
                    let dy = sy - y0 as f32;

                    let w00 = (1.0 - dx) * (1.0 - dy);
                    let w01 = dx * (1.0 - dy);
                    let w10 = (1.0 - dx) * dy;
                    let w11 = dx * dy;

                    let base = ((bn * c) * h) * w;
                    let idx00 = base + (y0 * w + x0);
                    let idx01 = base + (y0 * w + x1);
                    let idx10 = base + (y1 * w + x0);
                    let idx11 = base + (y1 * w + x1);

                    for ch in 0..c {
                        let off = ch * h * w;
                        let v00 = img[idx00 + off];
                        let v01 = img[idx01 + off];
                        let v10 = img[idx10 + off];
                        let v11 = img[idx11 + off];
                        let val = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
                        let out_idx = (((bn * c + ch) * h_out + oy) * w_out + ox) as usize;
                        dst[out_idx] = val;
                    }
                }
            }
        }
        let storage = WithDType::to_cpu_storage_owned(dst);
        Ok((storage, Shape::from((n, c, h_out, w_out))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        if self.align_corners {
            candle::bail!("grid_sample cuda only supports align_corners=false");
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle::bail!("grid_sample expects contiguous tensors");
        }
        let (n, c, h, w) = l1.shape().dims4()?;
        let (n2, h_out, w_out, two) = l2.shape().dims4()?;
        if n != n2 || two != 2 {
            candle::bail!(
                "grid_sample unexpected shapes image {:?} grid {:?}",
                l1.shape(),
                l2.shape()
            );
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        let dev = s1.device().clone();
        let img = s1.as_cuda_slice::<f32>()?;
        let grid = s2.as_cuda_slice::<f32>()?;
        let (img_o1, img_o2) = l1
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("non-contiguous image".to_string()))?;
        let (grid_o1, grid_o2) = l2
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("non-contiguous grid".to_string()))?;
        let img = img.slice(img_o1..img_o2);
        let grid = grid.slice(grid_o1..grid_o2);

        let out_elems = n * c * h_out * w_out;
        let out = unsafe { dev.alloc::<f32>(out_elems) }?;
        let func = dev.get_or_load_custom_func(
            "grid_sample_f32",
            "tha3",
            cuda_kernels::GRID_SAMPLE_KERNELS,
        )?;
        let block = 256u32;
        let grid_dim = ((out_elems as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&out);
        builder.arg(&img);
        builder.arg(&grid);
        candle::builder_arg!(
            builder,
            n as u32,
            c as u32,
            h as u32,
            w as u32,
            h_out as u32,
            w_out as u32
        );
        unsafe { builder.launch(cfg) }.w()?;

        let out = candle::CudaStorage::wrap_cuda_slice(out, dev);
        Ok((out, Shape::from((n, c, h_out, w_out))))
    }
}

pub struct GridChangeApplier {
    cached_shape: Option<(usize, usize, usize)>,
    cached_grid: Option<Tensor>,
}

impl GridChangeApplier {
    pub fn new() -> Self {
        Self {
            cached_shape: None,
            cached_grid: None,
        }
    }

    pub fn apply(&mut self, grid_change: &Tensor, image: &Tensor) -> Result<Tensor> {
        let (n, _c, h, w) = image.dims4()?;
        let device = image.device();
        let dtype = image.dtype();
        let grid_change = grid_change.to_dtype(DType::F32)?;
        let image = image.to_dtype(DType::F32)?;

        let base_grid = match self.cached_shape {
            Some((cn, ch, cw)) if (cn, ch, cw) == (n, h, w) => self
                .cached_grid
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing cached grid".to_string()))?
                .clone(),
            _ => {
                let grid = make_base_grid(n, h, w, device)?;
                self.cached_shape = Some((n, h, w));
                self.cached_grid = Some(grid.clone());
                grid
            }
        };

        let grid_change = grid_change.permute((0, 2, 3, 1))?.contiguous()?;
        let grid = base_grid.broadcast_add(&grid_change)?;
        let out = grid_sample(&image.contiguous()?, &grid.contiguous()?, false)?;
        if dtype != DType::F32 {
            out.to_dtype(dtype)
        } else {
            Ok(out)
        }
    }
}

fn make_base_grid(n: usize, h: usize, w: usize, device: &candle::Device) -> Result<Tensor> {
    let mut data = Vec::with_capacity(n * h * w * 2);
    let w_f = w as f32;
    let h_f = h as f32;
    for _ in 0..n {
        for y in 0..h {
            let gy = (2.0 * y as f32 + 1.0) / h_f - 1.0;
            for x in 0..w {
                let gx = (2.0 * x as f32 + 1.0) / w_f - 1.0;
                data.push(gx);
                data.push(gy);
            }
        }
    }
    Tensor::from_vec(data, (n, h, w, 2), device)
}
