use candle::{DType, IndexOp, Result, Tensor, D};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtxRopeType {
    Interleaved,
    Split,
}

impl LtxRopeType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "split" => Self::Split,
            _ => Self::Interleaved,
        }
    }
}

pub fn apply_rotary_emb(
    x: &Tensor,
    freqs: (&Tensor, &Tensor),
    rope_type: LtxRopeType,
) -> Result<Tensor> {
    match rope_type {
        LtxRopeType::Interleaved => apply_interleaved_rotary_emb(x, freqs.0, freqs.1),
        LtxRopeType::Split => apply_split_rotary_emb(x, freqs.0, freqs.1),
    }
}

fn apply_interleaved_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let last = *dims
        .last()
        .ok_or_else(|| candle::Error::msg("empty dims"))?;
    if last % 2 != 0 {
        candle::bail!("rope expects even last dimension, got {last}")
    }
    let mut new_shape = dims[..dims.len() - 1].to_vec();
    new_shape.push(last / 2);
    new_shape.push(2);
    let x_r = x.reshape(new_shape)?;
    let parts = x_r.chunk(2, D::Minus1)?;
    let t1 = parts[0].squeeze(D::Minus1)?;
    let t2 = parts[1].squeeze(D::Minus1)?;
    let t2n = t2.neg()?;
    let t_dup = Tensor::stack(&[&t2n, &t1], D::Minus1)?;
    let x_rot = t_dup.reshape(dims.to_vec())?;
    let a = x.broadcast_mul(cos)?;
    let b = x_rot.broadcast_mul(sin)?;
    let out = (a + b)?;
    Ok(out)
}

fn apply_split_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let mut x = x.clone();
    let mut needs_reshape = false;
    if x.rank() != 4 && cos.rank() == 4 {
        let (b, h, t, _) = cos.dims4()?;
        let last = x.dim(D::Minus1)?;
        x = x.reshape((b, t, h, last / h))?.transpose(1, 2)?;
        needs_reshape = true;
    }

    let last = x.dim(D::Minus1)?;
    if last % 2 != 0 {
        candle::bail!("split rope expects even last dimension, got {last}")
    }

    let mut new_shape = x.dims().to_vec();
    new_shape.pop();
    new_shape.push(2);
    new_shape.push(last / 2);
    let split = x.reshape(new_shape)?;
    let parts = split.chunk(2, D::Minus2)?;
    let first_half = parts[0].squeeze(D::Minus2)?;
    let second_half = parts[1].squeeze(D::Minus2)?;

    let cos_u = cos.unsqueeze(D::Minus2)?;
    let sin_u = sin.unsqueeze(D::Minus2)?;

    let mut output = split.broadcast_mul(&cos_u)?;
    let fh = output.i((.., .., .., 0, ..))?;
    let sh = output.i((.., .., .., 1, ..))?;
    let fh2 = (&fh - sin_u.broadcast_mul(&second_half)?)?;
    let sh2 = (&sh + sin_u.broadcast_mul(&first_half)?)?;
    let output = Tensor::stack(&[&fh2, &sh2], D::Minus2)?;
    let output = output.reshape(x.dims())?;

    if needs_reshape {
        let (b, h, t, _) = cos.dims4()?;
        let output = output
            .transpose(1, 2)?
            .reshape((b, t, h * (last / 2 * 2)))?;
        Ok(output)
    } else {
        Ok(output)
    }
}

pub fn precompute_freqs_cis(
    indices_grid: &Tensor,
    dim: usize,
    out_dtype: DType,
    theta: f64,
    max_pos: &[usize],
    use_middle_indices_grid: bool,
    num_attention_heads: usize,
    rope_type: LtxRopeType,
    double_precision: bool,
) -> Result<(Tensor, Tensor)> {
    let indices = generate_freq_grid(
        theta,
        indices_grid.dim(1)?,
        dim,
        indices_grid.device(),
        double_precision,
    )?;
    let freqs = generate_freqs(&indices, indices_grid, max_pos, use_middle_indices_grid)?;
    if rope_type == LtxRopeType::Split {
        let expected = dim / 2;
        let current = freqs.dim(D::Minus1)?;
        let pad = expected.saturating_sub(current);
        split_freqs_cis(&freqs, pad, num_attention_heads, out_dtype)
    } else {
        let n_elem = 2 * indices_grid.dim(1)?;
        interleaved_freqs_cis(&freqs, dim % n_elem, out_dtype)
    }
}

fn generate_freq_grid(
    theta: f64,
    positional_embedding_max_pos_count: usize,
    inner_dim: usize,
    device: &candle::Device,
    double_precision: bool,
) -> Result<Tensor> {
    let n_elem = 2 * positional_embedding_max_pos_count;
    let len = inner_dim / n_elem;
    let mut vals = Vec::with_capacity(len);
    let start = 1.0f64;
    let end = theta;
    for i in 0..len {
        let t = if len <= 1 {
            0.0
        } else {
            i as f64 / (len - 1) as f64
        };
        let exp = (start.ln() + t * (end.ln() - start.ln())) / theta.ln();
        let v = theta.powf(exp) * std::f64::consts::FRAC_PI_2;
        vals.push(v as f32);
    }
    let dtype = if double_precision {
        DType::F64
    } else {
        DType::F32
    };
    Tensor::from_vec(vals, (len,), device)?.to_dtype(dtype)
}

fn get_fractional_positions(indices_grid: &Tensor, max_pos: &[usize]) -> Result<Tensor> {
    let n_pos_dims = indices_grid.dim(1)?;
    if n_pos_dims != max_pos.len() {
        candle::bail!("position dims mismatch")
    }
    let mut cols = Vec::with_capacity(n_pos_dims);
    for i in 0..n_pos_dims {
        let col = indices_grid.i((.., i))?;
        cols.push((&col / max_pos[i] as f64)?);
    }
    Tensor::stack(&cols.iter().collect::<Vec<_>>(), D::Minus1)
}

fn generate_freqs(
    indices: &Tensor,
    indices_grid: &Tensor,
    max_pos: &[usize],
    use_middle_indices_grid: bool,
) -> Result<Tensor> {
    let mut indices_grid = indices_grid.clone();
    if use_middle_indices_grid {
        let start = indices_grid.i((.., .., .., 0))?;
        let end = indices_grid.i((.., .., .., 1))?;
        let mid = (&start + &end)?;
        indices_grid = (mid * 0.5)?;
    } else if indices_grid.rank() == 4 {
        indices_grid = indices_grid.i((.., .., .., 0))?;
    }
    let fractional = get_fractional_positions(&indices_grid, max_pos)?;
    let indices = indices.to_device(fractional.device())?;
    let fractional = fractional.unsqueeze(D::Minus1)?;
    let fractional = (fractional * 2.0)?;
    let fractional = (fractional - 1.0)?;
    let freqs = indices
        .broadcast_mul(&fractional)?
        .transpose(D::Minus1, D::Minus2)?
        .flatten_from(D::Minus2)?;
    Ok(freqs)
}

fn split_freqs_cis(
    freqs: &Tensor,
    pad_size: usize,
    num_attention_heads: usize,
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let mut cos = freqs.cos()?;
    let mut sin = freqs.sin()?;
    if pad_size != 0 {
        let (b, t, _d) = cos.dims3()?;
        let cos_pad = Tensor::ones((b, t, pad_size), cos.dtype(), cos.device())?;
        let sin_pad = Tensor::zeros((b, t, pad_size), sin.dtype(), sin.device())?;
        cos = Tensor::cat(&[&cos_pad, &cos], D::Minus1)?;
        sin = Tensor::cat(&[&sin_pad, &sin], D::Minus1)?;
    }
    let (b, t, d) = cos.dims3()?;
    let cos = cos
        .reshape((b, t, num_attention_heads, d / num_attention_heads))?
        .transpose(1, 2)?;
    let sin = sin
        .reshape((b, t, num_attention_heads, d / num_attention_heads))?
        .transpose(1, 2)?;
    Ok((cos.to_dtype(out_dtype)?, sin.to_dtype(out_dtype)?))
}

fn interleaved_freqs_cis(
    freqs: &Tensor,
    pad_size: usize,
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let cos = repeat_interleave_last(&cos, 2)?;
    let sin = repeat_interleave_last(&sin, 2)?;
    let mut cos = cos;
    let mut sin = sin;
    if pad_size != 0 {
        let (b, t, _d) = cos.dims3()?;
        let cos_pad = Tensor::ones((b, t, pad_size), cos.dtype(), cos.device())?;
        let sin_pad = Tensor::zeros((b, t, pad_size), sin.dtype(), sin.device())?;
        cos = Tensor::cat(&[&cos_pad, &cos], D::Minus1)?;
        sin = Tensor::cat(&[&sin_pad, &sin], D::Minus1)?;
    }
    Ok((cos.to_dtype(out_dtype)?, sin.to_dtype(out_dtype)?))
}

fn repeat_interleave_last(x: &Tensor, repeats: usize) -> Result<Tensor> {
    let mut shape = x.dims().to_vec();
    let last = *shape.last().unwrap_or(&0);
    shape.push(repeats);
    let x = x.unsqueeze(D::Minus1)?;
    let x = x.broadcast_as(shape.as_slice())?;
    let mut out_shape = x.dims().to_vec();
    let last_dim = out_shape.pop().unwrap_or(1);
    let prev = out_shape.pop().unwrap_or(1);
    out_shape.push(prev * last_dim);
    x.reshape(out_shape)
}
