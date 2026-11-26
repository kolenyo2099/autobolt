import type { RoiRect } from './types';

export const CLIP_SIZE = 336;
// Official CLIP / OSV-5M normalization constants
const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073] as const;
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711] as const;

function createWorkCanvas(width: number, height: number): {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
} {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Failed to acquire 2D canvas context for CLIP preprocessing');
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  return { canvas, ctx };
}

function rgbaToClipTensor(data: Uint8ClampedArray, size: number): Float32Array {
  const hw = size * size;
  const out = new Float32Array(3 * hw);
  const inv255 = 1 / 255;

  const meanR = CLIP_MEAN[0];
  const meanG = CLIP_MEAN[1];
  const meanB = CLIP_MEAN[2];
  const invStdR = 1 / CLIP_STD[0];
  const invStdG = 1 / CLIP_STD[1];
  const invStdB = 1 / CLIP_STD[2];

  for (let i = 0, p = 0; i < hw; i++, p += 4) {
    out[i] = (data[p] * inv255 - meanR) * invStdR;
    out[hw + i] = (data[p + 1] * inv255 - meanG) * invStdG;
    out[2 * hw + i] = (data[p + 2] * inv255 - meanB) * invStdB;
  }
  return out;
}

/**
 * Standard CLIP preprocessing:
 * 1. Resize shortest edge to `size` (336 for StreetCLIP, 224 for OSV-5M Baseline)
 * 2. Center crop size x size
 * 3. Rescale [0,1] and normalize with CLIP mean/std into NCHW Float32Array
 */
export function preprocessCenterCrop(bitmap: ImageBitmap, size = CLIP_SIZE): Float32Array {
  const { width, height } = bitmap;
  const scale = size / Math.min(width, height);
  const scaledW = Math.round(width * scale);
  const scaledH = Math.round(height * scale);
  const offsetX = Math.floor((scaledW - size) / 2);
  const offsetY = Math.floor((scaledH - size) / 2);

  const { ctx } = createWorkCanvas(size, size);
  ctx.drawImage(bitmap, -offsetX, -offsetY, scaledW, scaledH);
  const imageData = ctx.getImageData(0, 0, size, size);
  return rgbaToClipTensor(imageData.data, size);
}

/**
 * Multi-crop preprocessing (3 spatial crops along the long axis when aspect ratio > 1.15).
 */
export function preprocessMultiCrop(bitmap: ImageBitmap, size = CLIP_SIZE): Float32Array[] {
  const { width, height } = bitmap;
  const aspect = Math.max(width, height) / Math.min(width, height);
  if (aspect <= 1.15) {
    return [preprocessCenterCrop(bitmap, size)];
  }

  const scale = size / Math.min(width, height);
  const scaledW = Math.round(width * scale);
  const scaledH = Math.round(height * scale);
  const { ctx } = createWorkCanvas(size, size);

  const crops: Float32Array[] = [];
  const positions = [0, 0.5, 1];

  for (const t of positions) {
    ctx.clearRect(0, 0, size, size);
    const offsetX = scaledW > size ? Math.round((scaledW - size) * t) : 0;
    const offsetY = scaledH > size ? Math.round((scaledH - size) * t) : 0;
    ctx.drawImage(bitmap, -offsetX, -offsetY, scaledW, scaledH);
    const imageData = ctx.getImageData(0, 0, size, size);
    crops.push(rgbaToClipTensor(imageData.data, size));
  }

  return crops;
}

/**
 * Region-of-Interest preprocessing: extracts the normalized rectangle [x, y, w, h]
 * from the source image and fits/center-crops it to size x size.
 */
export function preprocessRoiCrop(
  bitmap: ImageBitmap,
  roi: RoiRect,
  size = CLIP_SIZE,
): Float32Array {
  const sx = Math.max(0, Math.min(bitmap.width - 1, Math.round(roi.x * bitmap.width)));
  const sy = Math.max(0, Math.min(bitmap.height - 1, Math.round(roi.y * bitmap.height)));
  const sw = Math.max(8, Math.min(bitmap.width - sx, Math.round(roi.w * bitmap.width)));
  const sh = Math.max(8, Math.min(bitmap.height - sy, Math.round(roi.h * bitmap.height)));

  const scale = size / Math.min(sw, sh);
  const scaledW = Math.round(sw * scale);
  const scaledH = Math.round(sh * scale);
  const offsetX = Math.floor((scaledW - size) / 2);
  const offsetY = Math.floor((scaledH - size) / 2);

  const { ctx } = createWorkCanvas(size, size);
  ctx.drawImage(bitmap, sx, sy, sw, sh, -offsetX, -offsetY, scaledW, scaledH);
  const imageData = ctx.getImageData(0, 0, size, size);
  return rgbaToClipTensor(imageData.data, size);
}

/**
 * Extracts a tightly packed RGB image scaled to at most `maxDim` on the longest side,
 * with dimensions rounded to multiples of 32 for bounded Qwen-VL visual token counts.
 */
export function extractVlmImageRgb(
  bitmap: ImageBitmap,
  roi?: RoiRect | null,
  maxDim = 448,
): { data: Uint8ClampedArray; width: number; height: number } {
  const sx = roi ? Math.max(0, Math.min(bitmap.width - 1, Math.round(roi.x * bitmap.width))) : 0;
  const sy = roi ? Math.max(0, Math.min(bitmap.height - 1, Math.round(roi.y * bitmap.height))) : 0;
  const sw = roi
    ? Math.max(16, Math.min(bitmap.width - sx, Math.round(roi.w * bitmap.width)))
    : bitmap.width;
  const sh = roi
    ? Math.max(16, Math.min(bitmap.height - sy, Math.round(roi.h * bitmap.height)))
    : bitmap.height;

  const scale = Math.min(1, maxDim / Math.max(sw, sh));
  const round32 = (v: number) => Math.max(64, Math.min(maxDim, Math.round((v * scale) / 32) * 32));
  const width = round32(sw);
  const height = round32(sh);

  const { ctx } = createWorkCanvas(width, height);
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, width, height);
  const rgba = ctx.getImageData(0, 0, width, height).data;

  const pixelCount = width * height;
  const data = new Uint8ClampedArray(pixelCount * 3);
  for (let i = 0, p = 0, q = 0; i < pixelCount; i++, p += 4, q += 3) {
    data[q] = rgba[p];
    data[q + 1] = rgba[p + 1];
    data[q + 2] = rgba[p + 2];
  }
  return { data, width, height };
}

export interface EdgeCornerCrop {
  id: string;
  label: string;
  roi: RoiRect;
  dataUrl: string;
}

/**
 * Produces a horizontally mirrored ImageBitmap (useful for flipped street/dashcam reposts).
 */
export async function flipBitmapHorizontal(bitmap: ImageBitmap): Promise<ImageBitmap> {
  const { canvas, ctx } = createWorkCanvas(bitmap.width, bitmap.height);
  ctx.translate(bitmap.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(bitmap, 0, 0);
  return createImageBitmap(canvas);
}

/**
 * Generates 8 zoomed edge & corner crops matching geo-sleuth's edge-scan habit
 * so analysts can inspect overlooked signs, plates, poles, and corners.
 */
export function generateEdgeCornerCrops(bitmap: ImageBitmap): EdgeCornerCrop[] {
  const defs: Array<{ id: string; label: string; roi: RoiRect }> = [
    { id: 'tl', label: 'Top-Left Corner', roi: { x: 0, y: 0, w: 0.34, h: 0.34 } },
    { id: 't', label: 'Top Edge', roi: { x: 0.33, y: 0, w: 0.34, h: 0.32 } },
    { id: 'tr', label: 'Top-Right Corner', roi: { x: 0.66, y: 0, w: 0.34, h: 0.34 } },
    { id: 'l', label: 'Left Edge', roi: { x: 0, y: 0.33, w: 0.34, h: 0.34 } },
    { id: 'r', label: 'Right Edge', roi: { x: 0.66, y: 0.33, w: 0.34, h: 0.34 } },
    { id: 'bl', label: 'Bottom-Left Corner', roi: { x: 0, y: 0.66, w: 0.34, h: 0.34 } },
    { id: 'b', label: 'Bottom Edge', roi: { x: 0.33, y: 0.68, w: 0.34, h: 0.32 } },
    { id: 'br', label: 'Bottom-Right Corner', roi: { x: 0.66, y: 0.66, w: 0.34, h: 0.34 } },
  ];

  const thumbSize = 180;
  const { canvas, ctx } = createWorkCanvas(thumbSize, thumbSize);

  return defs.map((def) => {
    const sx = Math.round(def.roi.x * bitmap.width);
    const sy = Math.round(def.roi.y * bitmap.height);
    const sw = Math.max(8, Math.round(def.roi.w * bitmap.width));
    const sh = Math.max(8, Math.round(def.roi.h * bitmap.height));
    ctx.clearRect(0, 0, thumbSize, thumbSize);
    ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, thumbSize, thumbSize);
    return {
      ...def,
      dataUrl: canvas.toDataURL('image/jpeg', 0.88),
    };
  });
}
