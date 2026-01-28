use tokenizers::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy};
use tokenizers::{Tokenizer, TruncationParams};

pub struct LtxvGemmaTokenizer {
    tokenizer: Tokenizer,
    pub max_length: usize,
    pad_id: u32,
}

impl LtxvGemmaTokenizer {
    pub fn new(tokenizer_path: &str, max_length: usize) -> candle::Result<Self> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(candle::Error::msg)?;
        let pad_id = tokenizer
            .get_vocab(true)
            .get("<pad>")
            .copied()
            .or_else(|| tokenizer.token_to_id("<|pad|>"))
            .or_else(|| tokenizer.token_to_id("<eos>"))
            .or_else(|| tokenizer.token_to_id("<bos>"))
            .unwrap_or(0);
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(max_length),
                direction: PaddingDirection::Left,
                pad_id,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(candle::Error::msg)?;
        Ok(Self {
            tokenizer,
            max_length,
            pad_id,
        })
    }

    pub fn tokenize_with_weights(&self, text: &str) -> candle::Result<Vec<(u32, u32)>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(candle::Error::msg)?;
        let ids = encoding.get_ids();
        let attention = encoding.get_attention_mask();
        let out: Vec<(u32, u32)> = ids
            .iter()
            .zip(attention.iter())
            .map(|(id, mask)| (*id as u32, *mask as u32))
            .collect();
        if out.is_empty() {
            return Ok(vec![(self.pad_id, 0)]);
        }
        Ok(out)
    }
}
