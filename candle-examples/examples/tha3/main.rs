//! Tha3 (Talking-Head Anime 3) inference port.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod image_util;
mod model;
mod ops;

use candle::{DType, Result, Tensor};
use clap::{Parser, ValueEnum};
use model::{
    build_separable_modules, build_standard_float_modules, EyebrowDecomposer00,
    EyebrowDecomposer03, EyebrowMorphingCombiner00, EyebrowMorphingCombiner03, SeparableModules,
    StandardFloatModules, TwoAlgoFaceBodyRotator05, TwoAlgoFaceBodyRotator05Separable,
};
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, ValueEnum)]
#[clap(rename_all = "kebab-case")]
enum Variant {
    StandardFloat,
    StandardHalf,
    SeparableFloat,
    SeparableHalf,
}

impl Variant {
    fn default_model_dir(self) -> &'static str {
        match self {
            Variant::StandardFloat => "data/models/standard_float",
            Variant::StandardHalf => "data/models/standard_half",
            Variant::SeparableFloat => "data/models/separable_float",
            Variant::SeparableHalf => "data/models/separable_half",
        }
    }

    fn dtype(self) -> DType {
        match self {
            Variant::StandardHalf | Variant::SeparableHalf => DType::F16,
            Variant::StandardFloat | Variant::SeparableFloat => DType::F32,
        }
    }

    fn is_separable(self) -> bool {
        matches!(self, Variant::SeparableFloat | Variant::SeparableHalf)
    }
}

#[derive(Clone, Debug)]
struct PoseOverride {
    name: String,
    value: f32,
}

impl FromStr for PoseOverride {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (name, value) = s
            .split_once('=')
            .ok_or_else(|| "expected NAME=VALUE".to_string())?;
        let value = value
            .parse::<f32>()
            .map_err(|err| format!("invalid value for {name}: {err}"))?;
        Ok(Self {
            name: name.to_string(),
            value,
        })
    }
}

#[derive(Parser)]
struct Args {
    /// Model variant to load.
    #[arg(long, value_enum, default_value = "standard-float")]
    variant: Variant,

    /// Path to model directory containing *.pt files (defaults to data/models/<variant>).
    #[arg(long)]
    model_dir: Option<String>,

    /// Input RGBA image.
    #[arg(long)]
    image: String,

    /// Output PNG path.
    #[arg(long, default_value = "tha3_output.png")]
    output: String,

    /// Pose overrides in the form name=value (repeatable).
    #[arg(long, value_name = "NAME=VALUE")]
    pose: Vec<PoseOverride>,

    /// Print pose parameter names and exit.
    #[arg(long)]
    list_pose_params: bool,

    /// Run on CPU rather than GPU.
    #[arg(long)]
    cpu: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.list_pose_params {
        print_pose_params();
        return Ok(());
    }
    let device = candle_examples::device(args.cpu)?;

    let model_dir = args
        .model_dir
        .unwrap_or_else(|| args.variant.default_model_dir().to_string());
    let model_dir = std::path::PathBuf::from(model_dir);
    let dtype = args.variant.dtype();

    let image = image_util::load_rgba_image(std::path::Path::new(&args.image), 512)?;
    let mut image = image.to_device(&device)?;
    if image.dtype() != dtype {
        image = image.to_dtype(dtype)?;
    }
    let pose_specs = pose_param_specs();
    let mut pose_values = default_pose_values(&pose_specs);
    apply_pose_overrides(&mut pose_values, &pose_specs, &args.pose)?;
    let mut pose = Tensor::from_vec(pose_values, (1, pose_specs.len()), &device)?;
    if pose.dtype() != dtype {
        pose = pose.to_dtype(dtype)?;
    }

    let output = if args.variant.is_separable() {
        let modules = build_separable_modules(&model_dir, &device, dtype)?;
        run_separable(&modules, &image, &pose)?
    } else {
        let modules = build_standard_float_modules(&model_dir, &device, dtype)?;
        run_standard(&modules, &image, &pose)?
    };
    image_util::save_rgba_tensor(&output, std::path::Path::new(&args.output))?;

    println!("saved output to {}", args.output);
    Ok(())
}

struct PoseGroup {
    name: &'static str,
    arity: usize,
    default_value: f32,
}

const POSE_GROUPS: &[PoseGroup] = &[
    PoseGroup {
        name: "eyebrow_troubled",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eyebrow_angry",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eyebrow_lowered",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eyebrow_raised",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eyebrow_happy",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eyebrow_serious",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eye_wink",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eye_happy_wink",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eye_surprised",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eye_relaxed",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eye_unimpressed",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "eye_raised_lower_eyelid",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "iris_small",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "mouth_aaa",
        arity: 1,
        default_value: 1.0,
    },
    PoseGroup {
        name: "mouth_iii",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "mouth_uuu",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "mouth_eee",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "mouth_ooo",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "mouth_delta",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "mouth_lowered_corner",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "mouth_raised_corner",
        arity: 2,
        default_value: 0.0,
    },
    PoseGroup {
        name: "mouth_smirk",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "iris_rotation_x",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "iris_rotation_y",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "head_x",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "head_y",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "neck_z",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "body_y",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "body_z",
        arity: 1,
        default_value: 0.0,
    },
    PoseGroup {
        name: "breathing",
        arity: 1,
        default_value: 0.0,
    },
];

struct PoseParamSpec {
    name: String,
    group: &'static str,
    default_value: f32,
}

fn pose_param_specs() -> Vec<PoseParamSpec> {
    let mut specs = Vec::new();
    for group in POSE_GROUPS {
        if group.arity == 1 {
            specs.push(PoseParamSpec {
                name: group.name.to_string(),
                group: group.name,
                default_value: group.default_value,
            });
        } else {
            specs.push(PoseParamSpec {
                name: format!("{}_left", group.name),
                group: group.name,
                default_value: group.default_value,
            });
            specs.push(PoseParamSpec {
                name: format!("{}_right", group.name),
                group: group.name,
                default_value: group.default_value,
            });
        }
    }
    specs
}

fn default_pose_values(specs: &[PoseParamSpec]) -> Vec<f32> {
    specs.iter().map(|spec| spec.default_value).collect()
}

fn apply_pose_overrides(
    pose: &mut [f32],
    specs: &[PoseParamSpec],
    overrides: &[PoseOverride],
) -> Result<()> {
    let mut name_to_index = HashMap::new();
    for (idx, spec) in specs.iter().enumerate() {
        name_to_index.insert(spec.name.as_str(), idx);
    }

    for override_item in overrides {
        if let Some(&idx) = name_to_index.get(override_item.name.as_str()) {
            pose[idx] = override_item.value;
            continue;
        }

        let group = POSE_GROUPS
            .iter()
            .find(|group| group.name == override_item.name);
        if let Some(group) = group {
            if group.arity == 2 {
                let left = format!("{}_left", group.name);
                let right = format!("{}_right", group.name);
                if let (Some(&left_idx), Some(&right_idx)) = (
                    name_to_index.get(left.as_str()),
                    name_to_index.get(right.as_str()),
                ) {
                    pose[left_idx] = override_item.value;
                    pose[right_idx] = override_item.value;
                    continue;
                }
            }
        }

        candle::bail!("unknown pose parameter: {}", override_item.name);
    }
    Ok(())
}

fn print_pose_params() {
    let specs = pose_param_specs();
    for (idx, spec) in specs.iter().enumerate() {
        println!("{idx:02} {}", spec.name);
    }
}

fn run_standard(modules: &StandardFloatModules, image: &Tensor, pose: &Tensor) -> Result<Tensor> {
    // Eyebrow decomposer input crop.
    let eyebrow_input = image.narrow(2, 64, 128)?.narrow(3, 192, 128)?;
    let eyebrow_decomp = modules.eyebrow_decomposer.forward(&eyebrow_input)?;

    let background_layer = &eyebrow_decomp[EyebrowDecomposer00::BACKGROUND_LAYER_INDEX];
    let eyebrow_layer = &eyebrow_decomp[EyebrowDecomposer00::EYEBROW_LAYER_INDEX];
    let eyebrow_pose = pose.narrow(1, 0, 12)?;
    let eyebrow_morph = modules.eyebrow_morphing_combiner.forward(
        background_layer,
        eyebrow_layer,
        &eyebrow_pose,
    )?;
    let eyebrow_morphed =
        &eyebrow_morph[EyebrowMorphingCombiner00::EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX];

    // Face morpher input crop and replace eyebrow region.
    let face_crop = image.narrow(2, 32, 192)?.narrow(3, 160, 192)?;
    let (n, c, _, _) = face_crop.dims4()?;
    let face_input = face_crop.slice_assign(&[0..n, 0..c, 32..160, 32..160], eyebrow_morphed)?;
    let face_pose = pose.narrow(1, 12, 27)?;
    let face_morph_out = modules.face_morpher.forward(&face_input, &face_pose)?;
    let face_morphed = &face_morph_out[0];

    // Compose into full image.
    let (n, c, _, _) = image.dims4()?;
    let face_morphed_full = image.slice_assign(&[0..n, 0..c, 32..224, 160..352], face_morphed)?;
    let face_morphed_half = face_morphed_full.upsample_bilinear2d(256, 256, false)?;

    // Rotator and editor.
    let rotation_pose = pose.narrow(1, 12 + 27, 6)?;
    let rotator_out = modules
        .rotator
        .forward(&face_morphed_half, &rotation_pose)?;
    let half_warped = &rotator_out[TwoAlgoFaceBodyRotator05::WARPED_IMAGE_INDEX];
    let half_grid_change = &rotator_out[TwoAlgoFaceBodyRotator05::GRID_CHANGE_INDEX];
    let full_warped = half_warped.upsample_bilinear2d(512, 512, false)?;
    let full_grid_change = half_grid_change.upsample_bilinear2d(512, 512, false)?;

    let editor_out = modules.editor.forward(
        &face_morphed_full,
        &full_warped,
        &full_grid_change,
        &rotation_pose,
    )?;
    Ok(editor_out[0].clone())
}

fn run_separable(modules: &SeparableModules, image: &Tensor, pose: &Tensor) -> Result<Tensor> {
    // Eyebrow decomposer input crop.
    let eyebrow_input = image.narrow(2, 64, 128)?.narrow(3, 192, 128)?;
    let eyebrow_decomp = modules.eyebrow_decomposer.forward(&eyebrow_input)?;

    let background_layer = &eyebrow_decomp[EyebrowDecomposer03::BACKGROUND_LAYER_INDEX];
    let eyebrow_layer = &eyebrow_decomp[EyebrowDecomposer03::EYEBROW_LAYER_INDEX];
    let eyebrow_pose = pose.narrow(1, 0, 12)?;
    let eyebrow_morph = modules.eyebrow_morphing_combiner.forward(
        background_layer,
        eyebrow_layer,
        &eyebrow_pose,
    )?;
    let eyebrow_morphed =
        &eyebrow_morph[EyebrowMorphingCombiner03::EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX];

    // Face morpher input crop and replace eyebrow region.
    let face_crop = image.narrow(2, 32, 192)?.narrow(3, 160, 192)?;
    let (n, c, _, _) = face_crop.dims4()?;
    let face_input = face_crop.slice_assign(&[0..n, 0..c, 32..160, 32..160], eyebrow_morphed)?;
    let face_pose = pose.narrow(1, 12, 27)?;
    let face_morph_out = modules.face_morpher.forward(&face_input, &face_pose)?;
    let face_morphed = &face_morph_out[0];

    // Compose into full image.
    let (n, c, _, _) = image.dims4()?;
    let face_morphed_full = image.slice_assign(&[0..n, 0..c, 32..224, 160..352], face_morphed)?;
    let face_morphed_half = face_morphed_full.upsample_bilinear2d(256, 256, false)?;

    // Rotator and editor.
    let rotation_pose = pose.narrow(1, 12 + 27, 6)?;
    let rotator_out = modules
        .rotator
        .forward(&face_morphed_half, &rotation_pose)?;
    let half_warped = &rotator_out[TwoAlgoFaceBodyRotator05Separable::WARPED_IMAGE_INDEX];
    let half_grid_change = &rotator_out[TwoAlgoFaceBodyRotator05Separable::GRID_CHANGE_INDEX];
    let full_warped = half_warped.upsample_bilinear2d(512, 512, false)?;
    let full_grid_change = half_grid_change.upsample_bilinear2d(512, 512, false)?;

    let editor_out = modules.editor.forward(
        &face_morphed_full,
        &full_warped,
        &full_grid_change,
        &rotation_pose,
    )?;
    Ok(editor_out[0].clone())
}
