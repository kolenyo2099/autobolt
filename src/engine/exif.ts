import exifr from 'exifr';
import type { ExifMetadata } from './types';

export async function extractImageExif(file: Blob): Promise<ExifMetadata | null> {
  try {
    const parsed = await exifr.parse(file, {
      gps: true,
      exif: true,
      tiff: true,
      translateValues: true,
    });
    if (!parsed) return null;

    const meta: ExifMetadata = {};
    if (typeof parsed.Make === 'string') meta.make = parsed.Make.trim();
    if (typeof parsed.Model === 'string') meta.model = parsed.Model.trim();
    if (typeof parsed.LensModel === 'string') meta.lensModel = parsed.LensModel.trim();
    if (parsed.DateTimeOriginal instanceof Date) {
      meta.dateTimeOriginal = parsed.DateTimeOriginal.toISOString();
    } else if (typeof parsed.DateTimeOriginal === 'string') {
      meta.dateTimeOriginal = parsed.DateTimeOriginal;
    }
    if (typeof parsed.ExifImageWidth === 'number') meta.width = parsed.ExifImageWidth;
    if (typeof parsed.ExifImageHeight === 'number') meta.height = parsed.ExifImageHeight;
    if (typeof parsed.FocalLength === 'number' && parsed.FocalLength > 0) {
      meta.focalLengthMm = parsed.FocalLength;
    }
    if (typeof parsed.FocalLengthIn35mmFormat === 'number' && parsed.FocalLengthIn35mmFormat > 0) {
      meta.focalLength35mm = parsed.FocalLengthIn35mmFormat;
      meta.hfovDeg = (2 * Math.atan(36 / (2 * parsed.FocalLengthIn35mmFormat)) * 180) / Math.PI;
    }
    if (typeof parsed.Orientation === 'number') {
      meta.orientation = parsed.Orientation;
    }

    if (
      typeof parsed.latitude === 'number' &&
      typeof parsed.longitude === 'number' &&
      Number.isFinite(parsed.latitude) &&
      Number.isFinite(parsed.longitude)
    ) {
      meta.gps = {
        lat: parsed.latitude,
        lng: parsed.longitude,
      };
    }

    const hasAny =
      meta.make ||
      meta.model ||
      meta.lensModel ||
      meta.dateTimeOriginal ||
      meta.focalLength35mm ||
      meta.focalLengthMm ||
      meta.gps ||
      meta.width;
    return hasAny ? meta : null;
  } catch {
    return null;
  }
}

export function haversineDistanceKm(
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number,
): number {
  const R = 6371;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}
