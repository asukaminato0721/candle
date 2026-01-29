#include <stdint.h>
#include <math.h>

extern "C" __global__ void grid_sample_f32(
    float* __restrict__ out,
    const float* __restrict__ image,
    const float* __restrict__ grid,
    const uint32_t n,
    const uint32_t c,
    const uint32_t h,
    const uint32_t w,
    const uint32_t h_out,
    const uint32_t w_out) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = n * c * h_out * w_out;
  if (idx >= total) return;

  const uint32_t ox = idx % w_out;
  uint32_t tmp = idx / w_out;
  const uint32_t oy = tmp % h_out;
  tmp /= h_out;
  const uint32_t ch = tmp % c;
  const uint32_t bn = tmp / c;

  const uint32_t grid_idx = ((bn * h_out + oy) * w_out + ox) * 2;
  float gx = grid[grid_idx];
  float gy = grid[grid_idx + 1];

  float sx = ((gx + 1.0f) * (float)w - 1.0f) * 0.5f;
  float sy = ((gy + 1.0f) * (float)h - 1.0f) * 0.5f;

  if (sx < 0.0f) sx = 0.0f;
  if (sy < 0.0f) sy = 0.0f;
  const float w_max = (float)w - 1.0f;
  const float h_max = (float)h - 1.0f;
  if (sx > w_max) sx = w_max;
  if (sy > h_max) sy = h_max;

  int x0 = (int)floorf(sx);
  int y0 = (int)floorf(sy);
  int x1 = x0 + 1; if (x1 >= (int)w) x1 = (int)w - 1;
  int y1 = y0 + 1; if (y1 >= (int)h) y1 = (int)h - 1;

  float dx = sx - (float)x0;
  float dy = sy - (float)y0;

  float w00 = (1.0f - dx) * (1.0f - dy);
  float w01 = dx * (1.0f - dy);
  float w10 = (1.0f - dx) * dy;
  float w11 = dx * dy;

  const uint32_t base = ((bn * c + ch) * h) * w;
  const uint32_t idx00 = base + (uint32_t)(y0 * (int)w + x0);
  const uint32_t idx01 = base + (uint32_t)(y0 * (int)w + x1);
  const uint32_t idx10 = base + (uint32_t)(y1 * (int)w + x0);
  const uint32_t idx11 = base + (uint32_t)(y1 * (int)w + x1);

  float v00 = image[idx00];
  float v01 = image[idx01];
  float v10 = image[idx10];
  float v11 = image[idx11];

  out[idx] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
}
