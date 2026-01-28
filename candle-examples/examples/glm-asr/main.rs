use anyhow::{Context, Result};
use clap::Parser;
use model::GlmAsrModel;

mod download;
mod model;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Path to the input audio file.
    #[arg(long)]
    audio: String,

    /// Model ID on Hugging Face Hub.
    #[arg(long, default_value = "zai-org/GLM-ASR-Nano-2512")]
    model_id: String,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut model =
        GlmAsrModel::new(&args.model_id, args.cpu).context("Failed to load GLM-ASR model")?;

    println!("Model loaded successfully on device: {:?}", model.device());

    let (audio_data, sample_rate) =
        candle_examples::audio::pcm_decode(&args.audio).context("Failed to decode audio file")?;

    let result = model
        .transcribe_audio(&audio_data, sample_rate, args.max_new_tokens)
        .context("Failed to transcribe audio")?;

    println!("\n===================================================\n");
    println!("{}", result.text);

    Ok(())
}
