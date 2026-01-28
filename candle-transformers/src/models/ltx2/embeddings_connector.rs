use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::Attention;
use super::feed_forward::FeedForward;
use super::rope::{precompute_freqs_cis, LtxRopeType};
use super::utils::rms_norm;

struct BasicTransformerBlock1D {
    attn1: Attention,
    ff: FeedForward,
}

impl BasicTransformerBlock1D {
    fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        rope_type: LtxRopeType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn1 = Attention::new(dim, None, heads, dim_head, 1e-6, rope_type, vb.pp("attn1"))?;
        let ff = FeedForward::new(dim, dim, 4, vb.pp("ff"))?;
        Ok(Self { attn1, ff })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let norm = rms_norm(hidden_states, None, 1e-6)?;
        let attn = self.attn1.forward(&norm, None, attention_mask, pe, None)?;
        let hidden_states = (hidden_states + &attn)?;
        let norm = rms_norm(&hidden_states, None, 1e-6)?;
        let ff = self.ff.forward(&norm)?;
        hidden_states + ff
    }
}

pub struct Embeddings1DConnector {
    num_attention_heads: usize,
    inner_dim: usize,
    positional_embedding_theta: f64,
    positional_embedding_max_pos: Vec<usize>,
    rope_type: LtxRopeType,
    double_precision_rope: bool,
    blocks: Vec<BasicTransformerBlock1D>,
}

impl Embeddings1DConnector {
    pub fn new(
        attention_head_dim: usize,
        num_attention_heads: usize,
        num_layers: usize,
        positional_embedding_theta: f64,
        positional_embedding_max_pos: Vec<usize>,
        rope_type: LtxRopeType,
        double_precision_rope: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = num_attention_heads * attention_head_dim;
        let mut blocks = Vec::with_capacity(num_layers);
        let vb_blocks = vb.pp("transformer_1d_blocks");
        for idx in 0..num_layers {
            blocks.push(BasicTransformerBlock1D::new(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                rope_type,
                vb_blocks.pp(idx),
            )?);
        }
        Ok(Self {
            num_attention_heads,
            inner_dim,
            positional_embedding_theta,
            positional_embedding_max_pos,
            rope_type,
            double_precision_rope,
            blocks,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let seq_len = hidden_states.dim(1)?;
        let indices = Tensor::arange(0f32, seq_len as f32, hidden_states.device())?;
        let indices = indices.unsqueeze(0)?.unsqueeze(0)?;
        let freqs = precompute_freqs_cis(
            &indices,
            self.inner_dim,
            hidden_states.dtype(),
            self.positional_embedding_theta,
            &self.positional_embedding_max_pos,
            false,
            self.num_attention_heads,
            self.rope_type,
            self.double_precision_rope,
        )?;

        let mut hs = hidden_states.clone();
        for block in &self.blocks {
            hs = block.forward(&hs, attention_mask, Some((&freqs.0, &freqs.1)))?;
        }
        let hs = rms_norm(&hs, None, 1e-6)?;
        Ok((hs, attention_mask.map(|m| m.clone())))
    }
}
