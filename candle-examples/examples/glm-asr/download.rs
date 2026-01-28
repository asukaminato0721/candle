use std::path::PathBuf;

use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};

/// # Errors
///
/// Returns an error if model files cannot be downloaded.
pub fn model_files(model_id: &str) -> Result<((PathBuf, Vec<PathBuf>), PathBuf)> {
    let revision = "main";
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let config = repo.get("config.json")?;

    let tokenizer_file = repo
        .get("tokenizer.json")
        .or_else(|_| repo.get("tokenizer/tokenizer.json"))?;

    let weights = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")
        .or_else(|_| {
            let mut files = Vec::new();
            for filename in [
                "model.safetensors",
                "pytorch_model.safetensors",
                "model-00001-of-00001.safetensors",
            ] {
                if let Ok(file) = repo.get(filename) {
                    files.push(file);
                }
            }
            if files.is_empty() {
                anyhow::bail!("No safetensors files found in model repository {model_id}");
            }
            Ok(files)
        })?;

    Ok(((config, weights), tokenizer_file))
}
