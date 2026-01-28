use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{linear, rms_norm, Linear, RmsNorm, VarBuilder};

use super::rope::{apply_rotary_emb, LtxRopeType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionFunction {
    Pytorch,
    Default,
}

impl AttentionFunction {
    pub fn from_str(s: &str) -> Self {
        match s {
            "pytorch" => Self::Pytorch,
            _ => Self::Default,
        }
    }
}

pub struct Attention {
    heads: usize,
    dim_head: usize,
    rope_type: LtxRopeType,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

impl Attention {
    pub fn new(
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        norm_eps: f64,
        rope_type: LtxRopeType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = heads * dim_head;
        let context_dim = context_dim.unwrap_or(query_dim);
        let q_norm = rms_norm(inner_dim, norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(inner_dim, norm_eps, vb.pp("k_norm"))?;
        let to_q = linear(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(context_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(context_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, query_dim, vb.pp("to_out").pp("0"))?;
        Ok(Self {
            heads,
            dim_head,
            rope_type,
            q_norm,
            k_norm,
            to_q,
            to_k,
            to_v,
            to_out,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
        k_pe: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let context = context.unwrap_or(x);
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(context)?;
        let v = self.to_v.forward(context)?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = match pe {
            Some(freqs) => {
                let q = apply_rotary_emb(&q, freqs, self.rope_type)?;
                let k_freqs = k_pe.unwrap_or(freqs);
                let k = apply_rotary_emb(&k, k_freqs, self.rope_type)?;
                (q, k)
            }
            None => (q, k),
        };
        let out = scaled_dot_product_attention(&q, &k, &v, self.heads, mask)?;
        self.to_out.forward(&out)
    }
}

fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, tq, _) = q.dims3()?;
    let (_, tk, _) = k.dims3()?;
    let dim_head = q.dim(D::Minus1)? / heads;
    let q = q.reshape((b, tq, heads, dim_head))?.transpose(1, 2)?; // b,h,t,dh
    let k = k.reshape((b, tk, heads, dim_head))?.transpose(1, 2)?;
    let v = v.reshape((b, tk, heads, dim_head))?.transpose(1, 2)?;

    let scale = (dim_head as f64).sqrt();
    let k_t = k.transpose(2, 3)?; // b,h,dh,tk
    let mut scores = (q.matmul(&k_t)? / scale)?;
    if let Some(mask) = mask {
        let mut mask = mask.clone();
        let rank = mask.rank();
        if rank == 2 {
            mask = mask.unsqueeze(0)?;
        }
        if mask.rank() == 3 {
            mask = mask.unsqueeze(1)?;
        }
        scores = scores.broadcast_add(&mask)?;
    }
    let attn = candle_nn::ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?;
    let attn = attn.to_dtype(scores.dtype())?;
    let out = attn.matmul(&v)?; // b,h,t,dh
    let out = out.transpose(1, 2)?.reshape((b, tq, heads * dim_head))?;
    Ok(out)
}
