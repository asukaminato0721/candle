use candle::{DType, Device, Result, Tensor};

pub struct Ltx2Scheduler;

impl Ltx2Scheduler {
    pub fn execute(
        &self,
        steps: usize,
        latent: Option<&Tensor>,
        max_shift: f64,
        base_shift: f64,
        stretch: bool,
        terminal: f64,
    ) -> Result<Tensor> {
        let tokens = if let Some(latent) = latent {
            let dims = latent.dims();
            let mut prod = 1usize;
            for d in dims.iter().skip(2) {
                prod *= *d;
            }
            prod as f64
        } else {
            4096.0
        };
        let dev = latent.map(|t| t.device().clone()).unwrap_or(Device::Cpu);
        let arange = Tensor::arange(0f32, (steps + 1) as f32, &dev)?;
        let sigmas = (1.0 - (arange / (steps as f64))?)?.to_dtype(DType::F32)?;

        let x1 = 1024.0;
        let x2 = 4096.0;
        let mm = (max_shift - base_shift) / (x2 - x1);
        let b = base_shift - mm * x1;
        let sigma_shift = tokens * mm + b;

        let mut sigmas_vec = sigmas.to_vec1::<f32>()?;
        for s in sigmas_vec.iter_mut() {
            if *s == 0.0 {
                continue;
            }
            let inv = 1.0f64 / (*s as f64) - 1.0;
            let v = (sigma_shift.exp() / (sigma_shift.exp() + inv.powf(1.0))) as f32;
            *s = v;
        }

        if stretch {
            let non_zero: Vec<f32> = sigmas_vec.iter().cloned().filter(|v| *v != 0.0).collect();
            if let Some(last) = non_zero.last() {
                let one_minus_z = 1.0 - *last as f64;
                let scale = one_minus_z / (1.0 - terminal);
                for s in sigmas_vec.iter_mut() {
                    if *s == 0.0 {
                        continue;
                    }
                    let one_minus = 1.0 - *s as f64;
                    *s = (1.0 - one_minus / scale) as f32;
                }
            }
        }

        Tensor::from_vec(sigmas_vec, (steps + 1,), sigmas.device())
    }
}
