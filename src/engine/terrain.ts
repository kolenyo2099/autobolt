import type { SkylineProfile, SkylineSample } from './types';

const TILE_Z = 11;
const TILE_SIZE = 256;
const EARTH_RADIUS_M = 6371000;
const REFRACTION_FACTOR = 0.87; // standard terrestrial refraction curvature coefficient

const tileCache = new Map<string, Float32Array>();

function latLngToTilePixel(lat: number, lng: number, zoom: number) {
  const latClamped = Math.max(-85.0511, Math.min(85.0511, lat));
  const n = Math.pow(2, zoom);
  const xFloat = ((lng + 180) / 360) * n;
  const latRad = (latClamped * Math.PI) / 180;
  const yFloat =
    ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n;

  const tx = ((Math.floor(xFloat) % n) + n) % n;
  const ty = Math.max(0, Math.min(n - 1, Math.floor(yFloat)));
  const px = Math.min(TILE_SIZE - 1, Math.max(0, Math.floor((xFloat - Math.floor(xFloat)) * TILE_SIZE)));
  const py = Math.min(TILE_SIZE - 1, Math.max(0, Math.floor((yFloat - Math.floor(yFloat)) * TILE_SIZE)));
  return { tx, ty, px, py };
}

async function fetchTerrariumTile(tx: number, ty: number): Promise<Float32Array> {
  const key = `${TILE_Z}/${tx}/${ty}`;
  const hit = tileCache.get(key);
  if (hit) return hit;

  const url = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${TILE_Z}/${tx}/${ty}.png`;
  const elev = await new Promise<Float32Array>((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = TILE_SIZE;
      canvas.height = TILE_SIZE;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Failed to create Canvas 2D context for DEM tile'));
        return;
      }
      ctx.drawImage(img, 0, 0, TILE_SIZE, TILE_SIZE);
      const rgba = ctx.getImageData(0, 0, TILE_SIZE, TILE_SIZE).data;
      const grid = new Float32Array(TILE_SIZE * TILE_SIZE);
      for (let i = 0, p = 0; i < grid.length; i++, p += 4) {
        // Mapzen/AWS Terrarium encoding: (R * 256 + G + B / 256) - 32768
        grid[i] = rgba[p] * 256 + rgba[p + 1] + rgba[p + 2] / 256 - 32768;
      }
      resolve(grid);
    };
    img.onerror = () => reject(new Error(`Failed loading AWS Terrarium tile ${key}`));
    img.src = url;
  });

  tileCache.set(key, elev);
  return elev;
}

function destPoint(latDeg: number, lngDeg: number, bearingDeg: number, distM: number): [number, number] {
  const b = (bearingDeg * Math.PI) / 180;
  const lat1 = (latDeg * Math.PI) / 180;
  const lng1 = (lngDeg * Math.PI) / 180;
  const ang = distM / EARTH_RADIUS_M;

  const sinLat2 =
    Math.sin(lat1) * Math.cos(ang) + Math.cos(lat1) * Math.sin(ang) * Math.cos(b);
  const lat2 = Math.asin(Math.max(-1, Math.min(1, sinLat2)));
  const lng2 =
    lng1 +
    Math.atan2(
      Math.sin(b) * Math.sin(ang) * Math.cos(lat1),
      Math.cos(ang) - Math.sin(lat1) * Math.sin(lat2),
    );
  return [(lat2 * 180) / Math.PI, ((lng2 * 180) / Math.PI + 540) % 360 - 180];
}

/**
 * Fetches AWS Terrarium DEM tiles covering the candidate viewing wedge and synthesizes
 * a horizon elevation skyline profile matching geo-sleuth terrain.py view physics.
 */
export async function computeSkylineProfile(params: {
  lat: number;
  lng: number;
  headingDeg: number;
  hfovDeg: number;
  pitchDeg?: number;
  observerHeightM?: number;
  maxRangeKm?: number;
}): Promise<SkylineProfile> {
  const numAz = 96;
  const numDist = 72;
  const maxRangeM = (params.maxRangeKm ?? 28) * 1000;
  const minRangeM = 300;
  const obsOffsetM = params.observerHeightM ?? 3;
  const pitchDeg = params.pitchDeg ?? 0;

  // Collect unique tiles across wedge
  const neededTiles = new Map<string, { tx: number; ty: number }>();
  const obsTile = latLngToTilePixel(params.lat, params.lng, TILE_Z);
  neededTiles.set(`${obsTile.tx},${obsTile.ty}`, { tx: obsTile.tx, ty: obsTile.ty });

  for (let a = 0; a <= 12; a++) {
    const xNorm = a / 12;
    const bearing = params.headingDeg + (xNorm - 0.5) * params.hfovDeg;
    for (let d = 1; d <= 6; d++) {
      const dist = (d / 6) * maxRangeM;
      const [pLat, pLng] = destPoint(params.lat, params.lng, bearing, dist);
      const tp = latLngToTilePixel(pLat, pLng, TILE_Z);
      const k = `${tp.tx},${tp.ty}`;
      if (!neededTiles.has(k) && neededTiles.size < 12) {
        neededTiles.set(k, { tx: tp.tx, ty: tp.ty });
      }
    }
  }

  const grids = new Map<string, Float32Array>();
  await Promise.all(
    Array.from(neededTiles.entries()).map(async ([k, { tx, ty }]) => {
      try {
        const g = await fetchTerrariumTile(tx, ty);
        grids.set(k, g);
      } catch {
        // Missing/failed corner tile degrades gracefully
      }
    }),
  );

  const sampleElev = (lat: number, lng: number): number | null => {
    const tp = latLngToTilePixel(lat, lng, TILE_Z);
    const grid = grids.get(`${tp.tx},${tp.ty}`);
    if (!grid) return null;
    return grid[tp.py * TILE_SIZE + tp.px];
  };

  const groundElev = sampleElev(params.lat, params.lng) ?? 0;
  const observerElevM = groundElev + obsOffsetM;

  const samples: SkylineSample[] = [];
  let maxElevDeg = -90;
  let minElevDeg = 90;

  for (let i = 0; i < numAz; i++) {
    const xNorm = i / (numAz - 1);
    const bearingDeg = (((params.headingDeg + (xNorm - 0.5) * params.hfovDeg) % 360) + 360) % 360;

    let rayMaxAngle = -90;
    let rayPeakDistKm = 0;

    for (let s = 1; s <= numDist; s++) {
      const t = s / numDist;
      const distM = minRangeM + (maxRangeM - minRangeM) * (t * t);
      const [pLat, pLng] = destPoint(params.lat, params.lng, bearingDeg, distM);
      const terrElev = sampleElev(pLat, pLng);
      if (terrElev === null) continue;

      // Earth curvature + atmospheric refraction drop
      const curvatureDropM = ((distM * distM) / (2 * EARTH_RADIUS_M)) * REFRACTION_FACTOR;
      const relHeightM = terrElev - observerElevM - curvatureDropM;
      const angleDeg = (Math.atan2(relHeightM, distM) * 180) / Math.PI;

      if (angleDeg > rayMaxAngle) {
        rayMaxAngle = angleDeg;
        rayPeakDistKm = distM / 1000;
      }
    }

    if (rayMaxAngle < -45) rayMaxAngle = 0;
    if (rayMaxAngle > maxElevDeg) maxElevDeg = rayMaxAngle;
    if (rayMaxAngle < minElevDeg) minElevDeg = rayMaxAngle;

    samples.push({
      xNorm,
      elevDeg: rayMaxAngle,
      distanceKm: rayPeakDistKm,
      bearingDeg,
    });
  }

  return {
    lat: params.lat,
    lng: params.lng,
    headingDeg: params.headingDeg,
    hfovDeg: params.hfovDeg,
    pitchDeg,
    observerElevM,
    maxElevDeg,
    minElevDeg,
    samples,
  };
}
