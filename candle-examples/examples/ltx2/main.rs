#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Result};
use clap::Parser;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs;
use std::path::{Path, PathBuf};

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma3::{Config as GemmaConfig, Model as GemmaModel};
use candle_transformers::models::ltx2::audio_vae::{
    decode_audio, AudioDecoder, AudioVaeConfig, Vocoder, VocoderConfig,
};
use candle_transformers::models::ltx2::embeddings_connector::Embeddings1DConnector;
use candle_transformers::models::ltx2::feature_extractor::GemmaFeaturesExtractorProjLinear;
use candle_transformers::models::ltx2::model::{LtxModel, LtxModelType, X0Model};
use candle_transformers::models::ltx2::noiser::GaussianNoiser;
use candle_transformers::models::ltx2::rope::LtxRopeType;
use candle_transformers::models::ltx2::scheduler::Ltx2Scheduler;
use candle_transformers::models::ltx2::text_encoder::{
    AVGemmaEncoderOutput, AVGemmaTextEncoderModel,
};
use candle_transformers::models::ltx2::tokenizer::LtxvGemmaTokenizer;
use candle_transformers::models::ltx2::transformer_args::Modality;
use candle_transformers::models::ltx2::types::{
    AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape,
};
use candle_transformers::models::ltx2::upsampler::LatentUpsampler;
use candle_transformers::models::ltx2::video_vae::{
    decode_video, upsample_latent, VideoDecoder, VideoDecoderConfig, VideoEncoder,
    VideoEncoderConfig,
};
use candle_transformers::models::ltx2::{
    AudioLatentTools, CfgGuider, EulerDiffusionStep, VideoLatentTools,
};

#[derive(Parser)]
#[command(author, version, about = "LTX-2 text-to-video+audio (inference only)")]
struct Args {
    /// Path to the LTX-2 checkpoint (.safetensors file or directory containing shards)
    #[arg(long)]
    checkpoint: PathBuf,

    /// Path to Gemma 3 directory (config.json + model*.safetensors)
    #[arg(long)]
    gemma: PathBuf,

    /// Path to tokenizer.json (defaults to <gemma>/tokenizer.json)
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Prompt text
    #[arg(
        long,
        default_value = "A cinematic shot of a surfer riding a glowing bioluminescent wave at night"
    )]
    prompt: String,

    /// Negative prompt text
    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// Video height (must be divisible by 32)
    #[arg(long, default_value_t = 512)]
    height: usize,

    /// Video width (must be divisible by 32)
    #[arg(long, default_value_t = 768)]
    width: usize,

    /// Number of frames (must satisfy 1 + 8*k)
    #[arg(long, default_value_t = 121)]
    num_frames: usize,

    /// Frames per second
    #[arg(long, default_value_t = 24.0)]
    fps: f64,

    /// Diffusion steps
    #[arg(long, default_value_t = 40)]
    steps: usize,

    /// CFG guidance scale
    #[arg(long, default_value_t = 4.0)]
    cfg_scale: f64,

    /// Random seed
    #[arg(long, default_value_t = 10)]
    seed: u64,

    /// Max token length for Gemma tokenizer
    #[arg(long, default_value_t = 1024)]
    max_length: usize,

    /// Output directory (frames + audio)
    #[arg(long, default_value = "ltx2_out")]
    output_dir: PathBuf,

    /// Mux frames + audio into an MP4 using ffmpeg
    #[arg(long)]
    mux: bool,

    /// Path to ffmpeg binary (defaults to `ffmpeg` in PATH)
    #[arg(long)]
    ffmpeg_path: Option<PathBuf>,

    /// Enable two-stage pipeline (requires spatial upsampler)
    #[arg(long)]
    two_stage: bool,

    /// Spatial upsampler checkpoint (.safetensors file or directory)
    #[arg(long)]
    spatial_upsampler: Option<PathBuf>,

    /// Optional stage-2 checkpoint (defaults to --checkpoint)
    #[arg(long)]
    stage2_checkpoint: Option<PathBuf>,

    /// Stage-2 steps when not using the distilled schedule
    #[arg(long, default_value_t = 4)]
    stage2_steps: usize,

    /// Use the distilled schedule for stage-2 (default: true)
    #[arg(long, default_value_t = true)]
    stage2_distilled_schedule: bool,

    /// Force CPU
    #[arg(long)]
    cpu: bool,

    /// Use f16 instead of bf16
    #[arg(long)]
    use_f16: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.height % 32 != 0 || args.width % 32 != 0 {
        bail!("height/width must be divisible by 32");
    }
    if (args.num_frames - 1) % 8 != 0 {
        bail!("num_frames must satisfy 1 + 8*k (e.g. 1, 9, 17, 25, ...)");
    }
    if args.two_stage && (args.height % 64 != 0 || args.width % 64 != 0) {
        bail!("height/width must be divisible by 64 for the two-stage pipeline");
    }
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    device.set_seed(args.seed)?;

    let dtype = if args.use_f16 {
        DType::F16
    } else {
        DType::BF16
    };

    let checkpoint_files = collect_safetensors(&args.checkpoint)?;
    let config = load_config_from_safetensors(&checkpoint_files[0])?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&checkpoint_files, dtype, &device)? };

    let transformer = build_transformer(vb.clone(), &config)?;

    let video_encoder_cfg = VideoEncoderConfig::from_config_value(&config)?;
    let video_decoder_cfg = VideoDecoderConfig::from_config_value(&config)?;
    let video_decoder_vb = vb
        .clone()
        .rename_f(|k| map_vae_key(k, "vae.decoder.", "vae.per_channel_statistics."));
    let video_latent_channels = video_encoder_cfg.out_channels;
    let video_decoder = VideoDecoder::new(video_decoder_vb, video_decoder_cfg)?;

    let audio_cfg = AudioVaeConfig::from_config_value(&config)?;
    let audio_z_channels = audio_cfg.z_channels;
    let audio_mel_bins = audio_cfg.mel_bins;
    let vocoder_cfg = VocoderConfig::from_config_value(&config)?;
    let audio_decoder_vb = vb
        .clone()
        .rename_f(|k| map_vae_key(k, "audio_vae.decoder.", "audio_vae.per_channel_statistics."));
    let vocoder_vb = vb.clone().rename_f(|k| format!("vocoder.{k}"));
    let audio_decoder = AudioDecoder::new(audio_decoder_vb, audio_cfg)?;
    let vocoder = Vocoder::new(vocoder_vb, vocoder_cfg)?;

    let mut text_encoder = build_text_encoder(&config, &args, &device, dtype, vb.clone())?;

    let AVGemmaEncoderOutput {
        video_encoding: v_context_p,
        audio_encoding: a_context_p,
        attention_mask: attn_mask_p,
    } = text_encoder.forward(&args.prompt)?;
    let AVGemmaEncoderOutput {
        video_encoding: v_context_n,
        audio_encoding: a_context_n,
        attention_mask: attn_mask_n,
    } = text_encoder.forward(&args.negative_prompt)?;

    let video_shape = VideoPixelShape {
        batch: 1,
        frames: args.num_frames,
        height: if args.two_stage {
            args.height / 2
        } else {
            args.height
        },
        width: if args.two_stage {
            args.width / 2
        } else {
            args.width
        },
        fps: args.fps,
    };

    let video_latent_shape = VideoLatentShape::from_pixel_shape(
        video_shape,
        video_latent_channels,
        candle_transformers::models::ltx2::types::SpatioTemporalScaleFactors::default(),
    );

    let (sample_rate, hop_length, audio_downsample) = audio_timing_params(&config);
    let audio_latent_shape = AudioLatentShape::from_video_pixel_shape(
        video_shape,
        audio_z_channels,
        audio_mel_bins,
        sample_rate,
        hop_length,
        audio_downsample,
    );

    let video_tools = VideoLatentTools::new(1, video_latent_shape, args.fps);
    let audio_tools = AudioLatentTools::new(
        1,
        audio_latent_shape,
        sample_rate,
        hop_length,
        audio_downsample,
        true,
    );

    let noiser = GaussianNoiser::new();
    let mut video_state = noiser.apply(
        &video_tools.create_initial_state(&device, dtype, None)?,
        1.0,
    )?;
    let mut audio_state = noiser.apply(
        &audio_tools.create_initial_state(&device, dtype, None)?,
        1.0,
    )?;

    let scheduler = Ltx2Scheduler;
    let sigmas = scheduler.execute(args.steps, None, 2.05, 0.95, true, 0.1)?;
    let sigmas = sigmas.to_device(&device)?;

    let stepper = EulerDiffusionStep;
    let guider = CfgGuider::new(args.cfg_scale);

    let (mut video_state, mut audio_state) = euler_denoising_loop(
        &sigmas,
        video_state,
        audio_state,
        &stepper,
        |video_state, audio_state, sigmas, idx| {
            let sigma = sigmas.i(idx)?;
            let pos_video =
                modality_from_state(video_state, &v_context_p, Some(&attn_mask_p), &sigma)?;
            let pos_audio =
                modality_from_state(audio_state, &a_context_p, Some(&attn_mask_p), &sigma)?;
            let (mut denoised_v, mut denoised_a) = transformer.forward(
                Some(pos_video),
                Some(pos_audio),
                candle_transformers::models::ltx2::diffusion::BatchedPerturbationConfig::empty(),
            )?;

            if guider.enabled() {
                let neg_video =
                    modality_from_state(video_state, &v_context_n, Some(&attn_mask_n), &sigma)?;
                let neg_audio =
                    modality_from_state(audio_state, &a_context_n, Some(&attn_mask_n), &sigma)?;
                let (neg_v, neg_a) = transformer.forward(
                    Some(neg_video),
                    Some(neg_audio),
                    candle_transformers::models::ltx2::diffusion::BatchedPerturbationConfig::empty(
                    ),
                )?;
                if let (Some(dv), Some(nv)) = (denoised_v.as_ref(), neg_v.as_ref()) {
                    denoised_v = Some((dv + guider.delta(dv, nv)?)?);
                }
                if let (Some(da), Some(na)) = (denoised_a.as_ref(), neg_a.as_ref()) {
                    denoised_a = Some((da + guider.delta(da, na)?)?);
                }
            }

            let denoised_v = denoised_v.ok_or_else(|| anyhow::anyhow!("video denoise missing"))?;
            let denoised_a = denoised_a.ok_or_else(|| anyhow::anyhow!("audio denoise missing"))?;
            Ok((denoised_v, denoised_a))
        },
    )?;

    let video_state = video_tools.unpatchify(&video_tools.clear_conditioning(&video_state)?)?;
    let audio_state = audio_tools.unpatchify(&audio_tools.clear_conditioning(&audio_state)?)?;

    let (decoded_video, decoded_audio) = if args.two_stage {
        let upsampler_path = args
            .spatial_upsampler
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--spatial-upscaler is required for --two-stage"))?;

        let video_encoder_vb = vb
            .clone()
            .rename_f(|k| map_vae_key(k, "vae.encoder.", "vae.per_channel_statistics."));
        let video_encoder = VideoEncoder::new(video_encoder_vb, video_encoder_cfg)?;
        let upsampler = load_upsampler(upsampler_path, dtype, &device)?;
        let upscaled_video_latent =
            upsample_latent(&video_state.latent, &video_encoder, &upsampler)?;

        let stage2_checkpoint = args.stage2_checkpoint.as_ref().unwrap_or(&args.checkpoint);
        let stage2_files = collect_safetensors(stage2_checkpoint)?;
        let stage2_config = load_config_from_safetensors(&stage2_files[0])?;
        let stage2_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&stage2_files, dtype, &device)? };
        let stage2_transformer = build_transformer(stage2_vb, &stage2_config)?;

        let stage2_video_shape = VideoPixelShape {
            batch: 1,
            frames: args.num_frames,
            height: args.height,
            width: args.width,
            fps: args.fps,
        };

        let stage2_video_latent_shape = VideoLatentShape::from_pixel_shape(
            stage2_video_shape,
            video_latent_channels,
            candle_transformers::models::ltx2::types::SpatioTemporalScaleFactors::default(),
        );

        let stage2_audio_latent_shape = AudioLatentShape::from_video_pixel_shape(
            stage2_video_shape,
            audio_z_channels,
            audio_mel_bins,
            sample_rate,
            hop_length,
            audio_downsample,
        );

        let stage2_video_tools = VideoLatentTools::new(1, stage2_video_latent_shape, args.fps);
        let stage2_audio_tools = AudioLatentTools::new(
            1,
            stage2_audio_latent_shape,
            sample_rate,
            hop_length,
            audio_downsample,
            true,
        );

        let stage2_sigmas = if args.stage2_distilled_schedule {
            Tensor::from_vec(vec![0.909375f32, 0.725, 0.421875, 0.0], (4,), &device)?
        } else {
            scheduler
                .execute(args.stage2_steps, None, 2.05, 0.95, true, 0.1)?
                .to_device(&device)?
        };
        let noise_scale = stage2_sigmas
            .i(0)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()? as f64;

        let mut stage2_video_state = noiser.apply(
            &stage2_video_tools.create_initial_state(
                &device,
                dtype,
                Some(upscaled_video_latent),
            )?,
            noise_scale,
        )?;
        let mut stage2_audio_state = noiser.apply(
            &stage2_audio_tools.create_initial_state(
                &device,
                dtype,
                Some(audio_state.latent.clone()),
            )?,
            noise_scale,
        )?;

        let (stage2_video_state, stage2_audio_state) = euler_denoising_loop(
            &stage2_sigmas,
            stage2_video_state,
            stage2_audio_state,
            &stepper,
            |video_state, audio_state, sigmas, idx| {
                let sigma = sigmas.i(idx)?;
                let pos_video =
                    modality_from_state(video_state, &v_context_p, Some(&attn_mask_p), &sigma)?;
                let pos_audio =
                    modality_from_state(audio_state, &a_context_p, Some(&attn_mask_p), &sigma)?;
                let (denoised_v, denoised_a) = stage2_transformer.forward(
                    Some(pos_video),
                    Some(pos_audio),
                    candle_transformers::models::ltx2::diffusion::BatchedPerturbationConfig::empty(
                    ),
                )?;
                Ok((
                    denoised_v.ok_or_else(|| anyhow::anyhow!("video denoise missing"))?,
                    denoised_a.ok_or_else(|| anyhow::anyhow!("audio denoise missing"))?,
                ))
            },
        )?;

        let stage2_video_state = stage2_video_tools
            .unpatchify(&stage2_video_tools.clear_conditioning(&stage2_video_state)?)?;
        let stage2_audio_state = stage2_audio_tools
            .unpatchify(&stage2_audio_tools.clear_conditioning(&stage2_audio_state)?)?;

        let decoded_video = decode_video(&stage2_video_state.latent, &video_decoder, None)?;
        let decoded_audio = decode_audio(&stage2_audio_state.latent, &audio_decoder, &vocoder)?;
        (decoded_video, decoded_audio)
    } else {
        let decoded_video = decode_video(&video_state.latent, &video_decoder, None)?;
        let decoded_audio = decode_audio(&audio_state.latent, &audio_decoder, &vocoder)?;
        (decoded_video, decoded_audio)
    };

    let output_sample_rate = config
        .get("vocoder")
        .and_then(|v| v.get("output_sample_rate"))
        .and_then(|v| v.as_u64())
        .unwrap_or(24000) as u32;
    save_outputs(
        &args.output_dir,
        &decoded_video,
        &decoded_audio,
        output_sample_rate,
        args.mux,
        args.ffmpeg_path.as_ref(),
        args.fps,
    )?;

    Ok(())
}

fn map_vae_key(key: &str, prefix: &str, stats_prefix: &str) -> String {
    if let Some(rest) = key.strip_prefix("per_channel_statistics.") {
        format!("{stats_prefix}{rest}")
    } else {
        format!("{prefix}{key}")
    }
}

fn collect_safetensors(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = fs::read_dir(path)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    if files.is_empty() {
        bail!("no .safetensors files found in {}", path.display());
    }
    files.sort();
    Ok(files)
}

fn load_config_from_safetensors(path: &Path) -> Result<serde_json::Value> {
    let file = fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let (_header_len, metadata) = SafeTensors::read_metadata(&mmap)?;
    let metadata_value = serde_json::to_value(&metadata)?;
    let Some(config_json) = metadata_value
        .get("__metadata__")
        .and_then(|v| v.get("config"))
        .and_then(|v| v.as_str())
    else {
        bail!("safetensors metadata is missing 'config'");
    };
    Ok(serde_json::from_str(config_json)?)
}

fn build_transformer(vb: VarBuilder, config: &serde_json::Value) -> Result<X0Model> {
    let transformer_cfg =
        candle_transformers::models::ltx2::model::LtxTransformerConfig::from_config_value(config);
    let transformer_vb = vb.rename_f(|k| format!("model.diffusion_model.{k}"));
    let transformer = LtxModel::new(transformer_cfg, transformer_vb, LtxModelType::AudioVideo)?;
    Ok(X0Model::new(transformer))
}

fn load_upsampler(path: &Path, dtype: DType, device: &Device) -> Result<LatentUpsampler> {
    let files = collect_safetensors(path)?;
    let config = load_config_from_safetensors(&files[0])?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
    let in_channels = config
        .get("in_channels")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let mid_channels = config
        .get("mid_channels")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;
    let num_blocks_per_stage = config
        .get("num_blocks_per_stage")
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;
    let dims = config.get("dims").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let spatial_upsample = config
        .get("spatial_upsample")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let temporal_upsample = config
        .get("temporal_upsample")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let spatial_scale = config
        .get("spatial_scale")
        .and_then(|v| v.as_f64())
        .unwrap_or(2.0);
    let rational_resampler = config
        .get("rational_resampler")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(LatentUpsampler::new(
        vb,
        in_channels,
        mid_channels,
        num_blocks_per_stage,
        dims,
        spatial_upsample,
        temporal_upsample,
        spatial_scale,
        rational_resampler,
    )?)
}

fn euler_denoising_loop<F>(
    sigmas: &Tensor,
    mut video_state: LatentState,
    mut audio_state: LatentState,
    stepper: &EulerDiffusionStep,
    mut denoise_fn: F,
) -> Result<(LatentState, LatentState)>
where
    F: FnMut(&LatentState, &LatentState, &Tensor, usize) -> Result<(Tensor, Tensor)>,
{
    let steps = sigmas.dim(0)?.saturating_sub(1);
    for idx in 0..steps {
        let (denoised_v, denoised_a) = denoise_fn(&video_state, &audio_state, sigmas, idx)?;
        let denoised_v = post_process_latent(
            &denoised_v,
            &video_state.denoise_mask,
            &video_state.clean_latent,
        )?;
        let denoised_a = post_process_latent(
            &denoised_a,
            &audio_state.denoise_mask,
            &audio_state.clean_latent,
        )?;
        let new_v = stepper.step(&video_state.latent, &denoised_v, sigmas, idx)?;
        let new_a = stepper.step(&audio_state.latent, &denoised_a, sigmas, idx)?;
        video_state = replace_latent(&video_state, new_v);
        audio_state = replace_latent(&audio_state, new_a);
    }
    Ok((video_state, audio_state))
}

fn build_text_encoder(
    config: &serde_json::Value,
    args: &Args,
    device: &Device,
    dtype: DType,
    vb: VarBuilder,
) -> Result<AVGemmaTextEncoderModel> {
    let tokenizer_path = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| args.gemma.join("tokenizer.json"));
    let tokenizer = LtxvGemmaTokenizer::new(tokenizer_path.to_str().unwrap(), args.max_length)?;

    let gemma_cfg_path = args.gemma.join("config.json");
    let gemma_cfg: GemmaConfig = serde_json::from_reader(fs::File::open(gemma_cfg_path)?)?;
    let gemma_weights = collect_gemma_weights(&args.gemma)?;
    let gemma_vb = unsafe { VarBuilder::from_mmaped_safetensors(&gemma_weights, dtype, device)? };
    let gemma_model = GemmaModel::new(false, &gemma_cfg, gemma_vb)?;

    let rope_type = config
        .get("transformer")
        .and_then(|v| v.get("rope_type"))
        .and_then(|v| v.as_str())
        .map(LtxRopeType::from_str)
        .unwrap_or(LtxRopeType::Interleaved);
    let double_precision = config
        .get("transformer")
        .and_then(|v| v.get("frequencies_precision"))
        .and_then(|v| v.as_str())
        .map(|v| v == "float64")
        .unwrap_or(false);
    let connector_max_pos = config
        .get("transformer")
        .and_then(|v| v.get("connector_positional_embedding_max_pos"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect()
        })
        .unwrap_or_else(|| vec![1]);

    let feature_vb = vb
        .clone()
        .rename_f(|k| format!("text_embedding_projection.{k}"));
    let feature_extractor = GemmaFeaturesExtractorProjLinear::new(feature_vb)?;

    let video_conn_vb = vb
        .clone()
        .rename_f(|k| format!("model.diffusion_model.video_embeddings_connector.{k}"));
    let audio_conn_vb = vb
        .clone()
        .rename_f(|k| format!("model.diffusion_model.audio_embeddings_connector.{k}"));

    let video_connector = Embeddings1DConnector::new(
        128,
        30,
        2,
        10000.0,
        connector_max_pos.clone(),
        rope_type,
        double_precision,
        video_conn_vb,
    )?;
    let audio_connector = Embeddings1DConnector::new(
        128,
        30,
        2,
        10000.0,
        connector_max_pos,
        rope_type,
        double_precision,
        audio_conn_vb,
    )?;

    let mut text_encoder =
        AVGemmaTextEncoderModel::new(feature_extractor, video_connector, audio_connector);
    text_encoder.tokenizer = Some(tokenizer);
    text_encoder.model = Some(gemma_model);
    Ok(text_encoder)
}

fn collect_gemma_weights(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("model"))
                    .unwrap_or(false)
        })
        .collect();
    if files.is_empty() {
        bail!(
            "no gemma model*.safetensors files found in {}",
            dir.display()
        );
    }
    files.sort();
    Ok(files)
}

fn audio_timing_params(config: &serde_json::Value) -> (usize, usize, usize) {
    let audio_cfg = config.get("audio_vae").unwrap_or(config);
    let model_params = audio_cfg.get("model").and_then(|v| v.get("params"));
    let sample_rate = model_params
        .and_then(|v| v.get("sampling_rate"))
        .and_then(|v| v.as_u64())
        .unwrap_or(16000) as usize;
    let preprocessing = audio_cfg.get("preprocessing");
    let hop_length = preprocessing
        .and_then(|v| v.get("stft"))
        .and_then(|v| v.get("hop_length"))
        .and_then(|v| v.as_u64())
        .unwrap_or(160) as usize;
    let downsample = 4usize;
    (sample_rate, hop_length, downsample)
}

fn timesteps_from_mask(mask: &Tensor, sigma: &Tensor) -> candle::Result<Tensor> {
    mask.broadcast_mul(sigma)
}

fn modality_from_state(
    state: &LatentState,
    context: &Tensor,
    attention_mask: Option<&Tensor>,
    sigma: &Tensor,
) -> candle::Result<Modality> {
    Ok(Modality {
        latent: state.latent.clone(),
        timesteps: timesteps_from_mask(&state.denoise_mask, sigma)?,
        positions: state.positions.clone(),
        context: context.clone(),
        enabled: true,
        context_mask: attention_mask.map(|m| m.clone()),
    })
}

fn post_process_latent(denoised: &Tensor, mask: &Tensor, clean: &Tensor) -> candle::Result<Tensor> {
    let one = Tensor::ones_like(mask)?;
    let inv = (&one - mask)?;
    denoised
        .broadcast_mul(mask)?
        .broadcast_add(&clean.to_dtype(denoised.dtype())?.broadcast_mul(&inv)?)
}

fn replace_latent(state: &LatentState, latent: Tensor) -> LatentState {
    LatentState {
        latent,
        denoise_mask: state.denoise_mask.clone(),
        positions: state.positions.clone(),
        clean_latent: state.clean_latent.clone(),
    }
}

fn save_outputs(
    output_dir: &Path,
    video: &Tensor,
    audio: &Tensor,
    sample_rate: u32,
    mux: bool,
    ffmpeg_path: Option<&PathBuf>,
    fps: f64,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    save_video_frames(video, output_dir)?;
    save_audio_wav(audio, output_dir.join("audio.wav"), sample_rate)?;
    if mux {
        mux_with_ffmpeg(output_dir, ffmpeg_path, fps)?;
    }
    Ok(())
}

fn save_video_frames(video: &Tensor, output_dir: &Path) -> Result<()> {
    let video = video.to_device(&Device::Cpu)?;
    let (b, f, h, w, c) = video.dims5()?;
    if b != 1 {
        bail!("expected batch=1, got {b}");
    }
    if c != 3 {
        bail!("expected 3 channels, got {c}");
    }
    for idx in 0..f {
        let frame = video.i((0, idx, .., .., ..))?;
        let data = frame.flatten_all()?.to_vec1::<u8>()?;
        let img = image::RgbImage::from_raw(w as u32, h as u32, data)
            .ok_or_else(|| anyhow::anyhow!("failed to create image for frame {idx}"))?;
        let path = output_dir.join(format!("frame_{idx:04}.png"));
        img.save(path)?;
    }
    Ok(())
}

fn save_audio_wav(audio: &Tensor, path: PathBuf, sample_rate: u32) -> Result<()> {
    let audio = audio.to_device(&Device::Cpu)?;
    let audio = audio.to_dtype(DType::F32)?;
    let dims = audio.dims();
    let (channels, samples) = match dims {
        [s] => (1usize, *s),
        [c, s] => (*c, *s),
        _ => bail!("unexpected audio dims {dims:?}"),
    };

    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: channels as u16,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;

    if channels == 1 {
        let data = audio.flatten_all()?.to_vec1::<f32>()?;
        for s in data {
            let v = (s.max(-1.0).min(1.0) * i16::MAX as f32) as i16;
            writer.write_sample(v)?;
        }
    } else {
        let data = audio.to_vec2::<f32>()?;
        for i in 0..samples {
            for ch in 0..channels {
                let s = data[ch][i];
                let v = (s.max(-1.0).min(1.0) * i16::MAX as f32) as i16;
                writer.write_sample(v)?;
            }
        }
    }
    writer.finalize()?;
    Ok(())
}

fn mux_with_ffmpeg(output_dir: &Path, ffmpeg_path: Option<&PathBuf>, fps: f64) -> Result<()> {
    let ffmpeg = ffmpeg_path
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "ffmpeg".to_string());
    let frames = output_dir.join("frame_%04d.png");
    let audio = output_dir.join("audio.wav");
    let output = output_dir.join("output.mp4");

    let status = std::process::Command::new(ffmpeg)
        .arg("-y")
        .arg("-framerate")
        .arg(format!("{fps}"))
        .arg("-i")
        .arg(frames)
        .arg("-i")
        .arg(audio)
        .arg("-c:v")
        .arg("libx264")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-c:a")
        .arg("aac")
        .arg("-shortest")
        .arg(output)
        .status();

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => bail!("ffmpeg failed with status {s}"),
        Err(err) => bail!("failed to run ffmpeg: {err}"),
    }
}
