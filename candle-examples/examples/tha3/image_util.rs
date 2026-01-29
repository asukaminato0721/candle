use candle::{DType, Result, Tensor};
use image::{imageops::FilterType, ImageBuffer, Rgba};

pub fn load_rgba_image(path: &std::path::Path, size: usize) -> Result<Tensor> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(candle::Error::wrap)?;
    let (w, h) = (img.width(), img.height());
    let d = w.min(h);
    let x0 = (w - d) / 2;
    let y0 = (h - d) / 2;
    let img = img.crop_imm(x0, y0, d, d);
    let img = img.resize_exact(size as u32, size as u32, FilterType::Lanczos3);
    let mut img = img.to_rgba8();

    // Zero rgb where alpha is zero to avoid color bleed.
    for px in img.pixels_mut() {
        if px[3] == 0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
        }
    }

    let mut data = vec![0f32; 4 * size * size];
    for y in 0..size {
        for x in 0..size {
            let px = img.get_pixel(x as u32, y as u32);
            let r = srgb_to_linear(px[0] as f32 / 255.0);
            let g = srgb_to_linear(px[1] as f32 / 255.0);
            let b = srgb_to_linear(px[2] as f32 / 255.0);
            let a = px[3] as f32 / 255.0;
            let idx = y * size + x;
            data[idx] = r * 2.0 - 1.0;
            data[size * size + idx] = g * 2.0 - 1.0;
            data[2 * size * size + idx] = b * 2.0 - 1.0;
            data[3 * size * size + idx] = a * 2.0 - 1.0;
        }
    }
    let t = Tensor::from_vec(data, (4, size, size), &candle::Device::Cpu)?;
    t.unsqueeze(0)?.to_dtype(DType::F32)
}

pub fn save_rgba_tensor(t: &Tensor, path: &std::path::Path) -> Result<()> {
    let t = match t.rank() {
        4 => t.squeeze(0)?,
        3 => t.clone(),
        _ => candle::bail!("expected tensor rank 3 or 4, got {}", t.rank()),
    };
    let (c, h, w) = t.dims3()?;
    if c != 4 {
        candle::bail!("expected 4 channels, got {c}");
    }
    let t = t.to_dtype(DType::F32)?;
    let t = t.permute((1, 2, 0))?.contiguous()?;
    let data = t.flatten_all()?.to_vec1::<f32>()?;

    let mut out = Vec::with_capacity(h * w * 4);
    for i in 0..(h * w) {
        let r = to_u8_linear(data[i * 4]);
        let g = to_u8_linear(data[i * 4 + 1]);
        let b = to_u8_linear(data[i * 4 + 2]);
        let a = to_u8_alpha(data[i * 4 + 3]);
        out.push(r);
        out.push(g);
        out.push(b);
        out.push(a);
    }
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(w as u32, h as u32, out)
        .ok_or_else(|| candle::Error::Msg("failed to create image".to_string()))?;
    img.save(path).map_err(candle::Error::wrap)?;
    Ok(())
}

fn to_u8_linear(v: f32) -> u8 {
    let v = (v + 1.0) * 0.5;
    let v = linear_to_srgb(v.clamp(0.0, 1.0));
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

fn to_u8_alpha(v: f32) -> u8 {
    let v = (v + 1.0) * 0.5;
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn srgb_to_linear(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.003130804953560372 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}
