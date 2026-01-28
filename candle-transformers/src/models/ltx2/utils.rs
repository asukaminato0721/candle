use candle::{DType, Result, Tensor, D};

pub fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let hidden = x.dim(D::Minus1)? as f64;
    let x_f = x.to_dtype(DType::F32)?;
    let norm = (x_f.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
    let x_norm = x_f.broadcast_div(&(norm + eps)?.sqrt()?)?;
    let x_norm = x_norm.to_dtype(x.dtype())?;
    match weight {
        Some(w) => x_norm.broadcast_mul(w),
        None => Ok(x_norm),
    }
}

pub fn to_velocity(sample: &Tensor, sigma: f64, denoised: &Tensor) -> Result<Tensor> {
    if sigma == 0.0 {
        candle::bail!("sigma cannot be 0.0")
    }
    let sample_f = sample.to_dtype(DType::F32)?;
    let denoised_f = denoised.to_dtype(DType::F32)?;
    ((sample_f - denoised_f)? / sigma)?.to_dtype(sample.dtype())
}

pub fn to_denoised(sample: &Tensor, velocity: &Tensor, sigma: f64) -> Result<Tensor> {
    let sample_f = sample.to_dtype(DType::F32)?;
    let velocity_f = velocity.to_dtype(DType::F32)?;
    (sample_f - (velocity_f * sigma)?)?.to_dtype(sample.dtype())
}

pub fn check_config_value<T: PartialEq + std::fmt::Debug>(
    config: &serde_json::Value,
    key: &str,
    expected: T,
) -> Result<()> {
    let actual = config.get(key);
    let actual_dbg = actual.map(|v| format!("{v}"));
    if actual_dbg.as_deref() != Some(&format!("{expected:?}")) {
        // soft check: if the key is missing, don't fail.
        if actual.is_some() {
            candle::bail!("config value {key} is {actual:?}, expected {expected:?}")
        }
    }
    Ok(())
}
