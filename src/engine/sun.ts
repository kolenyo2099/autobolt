import type { SolarBandPoint, SolarTimeWindow } from './types';

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

export interface SolarPosResult {
  elevationDeg: number;
  azimuthDeg: number; // clockwise from True North (0..360)
  shadowBearingDeg: number;
  shadowLengthPerM: number | null;
}

function julianDay(utcMs: number): number {
  return utcMs / 86400000 + 2440587.5;
}

/**
 * NOAA / Meeus solar position algorithm with atmospheric refraction correction.
 * Accurate to <= 0.02 deg for 1950-2050.
 */
export function solarPosition(lat: number, lng: number, utcDate: Date): SolarPosResult {
  const jd = julianDay(utcDate.getTime());
  const T = (jd - 2451545.0) / 36525.0;

  const L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360;
  const M = 357.52911 + T * (35999.05029 - 0.0001537 * T);
  const e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T);

  const mRad = M * DEG;
  const C =
    Math.sin(mRad) * (1.914602 - T * (0.004817 + 0.000014 * T)) +
    Math.sin(2 * mRad) * (0.019993 - 0.000101 * T) +
    Math.sin(3 * mRad) * 0.000289;

  const sunTrueLong = L0 + C;
  const omega = 125.04 - 1934.136 * T;
  const lambda = sunTrueLong - 0.00569 - 0.00478 * Math.sin(omega * DEG);

  const meanObliq =
    23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60;
  const obliqCorr = meanObliq + 0.00256 * Math.cos(omega * DEG);

  const decl = Math.asin(Math.sin(obliqCorr * DEG) * Math.sin(lambda * DEG));
  const y = Math.tan((obliqCorr * DEG) / 2) ** 2;

  const eqOfTime =
    4 *
    RAD *
    (y * Math.sin(2 * L0 * DEG) -
      2 * e * Math.sin(mRad) +
      4 * e * y * Math.sin(mRad) * Math.cos(2 * L0 * DEG) -
      0.5 * y * y * Math.sin(4 * L0 * DEG) -
      1.25 * e * e * Math.sin(2 * mRad));

  const utcMin =
    utcDate.getUTCHours() * 60 +
    utcDate.getUTCMinutes() +
    utcDate.getUTCSeconds() / 60;
  const trueSolarTime = (((utcMin + eqOfTime + 4 * lng) % 1440) + 1440) % 1440;
  const hourAngleDeg = trueSolarTime / 4 < 0 ? trueSolarTime / 4 + 180 : trueSolarTime / 4 - 180;
  const haRad = hourAngleDeg * DEG;
  const latRad = lat * DEG;

  const cosZenith = Math.max(
    -1,
    Math.min(1, Math.sin(latRad) * Math.sin(decl) + Math.cos(latRad) * Math.cos(decl) * Math.cos(haRad)),
  );
  const zenithDeg = Math.acos(cosZenith) * RAD;
  const rawElevDeg = 90 - zenithDeg;

  // Approximate atmospheric refraction
  let refraction = 0;
  if (rawElevDeg > 85) {
    refraction = 0;
  } else if (rawElevDeg > 5) {
    const te = Math.tan(rawElevDeg * DEG);
    refraction = (58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5) / 3600;
  } else if (rawElevDeg > -0.575) {
    refraction =
      (1735 + rawElevDeg * (-518.2 + rawElevDeg * (103.4 + rawElevDeg * (-12.79 + rawElevDeg * 0.711)))) /
      3600;
  } else {
    refraction = (-20.774 / Math.tan(rawElevDeg * DEG)) / 3600;
  }

  const elevationDeg = rawElevDeg + refraction;

  const sinZen = Math.sin(zenithDeg * DEG);
  let azimuthDeg = 0;
  if (Math.abs(sinZen) > 1e-6 && Math.abs(Math.cos(latRad)) > 1e-6) {
    const cosAz = Math.max(
      -1,
      Math.min(
        1,
        (Math.sin(latRad) * Math.cos(zenithDeg * DEG) - Math.sin(decl)) /
          (Math.cos(latRad) * sinZen),
      ),
    );
    azimuthDeg = hourAngleDeg > 0 ? (Math.acos(cosAz) * RAD + 180) % 360 : (540 - Math.acos(cosAz) * RAD) % 360;
  }

  const shadowBearingDeg = (azimuthDeg + 180) % 360;
  const shadowLengthPerM =
    elevationDeg > 0.5 ? 1 / Math.tan(elevationDeg * DEG) : null;

  return { elevationDeg, azimuthDeg, shadowBearingDeg, shadowLengthPerM };
}

export function shadowRatioToElevation(ratio: number): number {
  if (!Number.isFinite(ratio) || ratio <= 0) return 89;
  return Math.atan(1 / ratio) * RAD;
}

export function angularDiffDeg(a: number, b: number): number {
  const d = Math.abs((((a - b) % 360) + 540) % 360 - 180);
  return d;
}

/**
 * Scans populated Earth latitudes for all points matching observed sun elevation
 * (and optional shadow bearing) at a known UTC capture time.
 */
export function computeSolarBand(params: {
  utcDate: Date;
  elevationDeg: number;
  elevTolDeg?: number;
  shadowBearingDeg?: number | null;
  azTolDeg?: number;
}): SolarBandPoint[] {
  const elevTol = params.elevTolDeg ?? 4.5;
  const azTol = params.azTolDeg ?? 18;
  const hasAz =
    typeof params.shadowBearingDeg === 'number' && Number.isFinite(params.shadowBearingDeg);
  const pts: SolarBandPoint[] = [];

  for (let lat = -60; lat <= 70; lat += 4) {
    for (let lng = -180; lng <= 176; lng += 4) {
      const pos = solarPosition(lat, lng, params.utcDate);
      if (pos.elevationDeg <= 0) continue;
      const elevDiffDeg = Math.abs(pos.elevationDeg - params.elevationDeg);
      if (elevDiffDeg > elevTol) continue;

      let azDiffDeg: number | undefined;
      if (hasAz) {
        azDiffDeg = angularDiffDeg(pos.shadowBearingDeg, params.shadowBearingDeg!);
        if (azDiffDeg > azTol) continue;
      }

      const elevScore = 1 - elevDiffDeg / elevTol;
      const azScore = hasAz && azDiffDeg !== undefined ? 1 - azDiffDeg / azTol : 1;
      pts.push({
        lat,
        lng,
        score: Math.max(0.1, Math.min(1, 0.55 * elevScore + 0.45 * azScore)),
        elevDiffDeg,
        azDiffDeg,
      });
    }
  }

  return pts.sort((a, b) => b.score - a.score);
}

/**
 * Given a known candidate location and calendar date, finds all local times of day
 * matching the observed shadow elevation and/or shadow bearing.
 */
export function chronolocateAtPoint(params: {
  lat: number;
  lng: number;
  dateStr: string; // YYYY-MM-DD
  tzOffsetHours: number;
  elevationDeg: number;
  elevTolDeg?: number;
  shadowBearingDeg?: number | null;
  azTolDeg?: number;
}): SolarTimeWindow[] {
  const elevTol = params.elevTolDeg ?? 4.5;
  const azTol = params.azTolDeg ?? 18;
  const hasAz =
    typeof params.shadowBearingDeg === 'number' && Number.isFinite(params.shadowBearingDeg);

  const [y, m, d] = params.dateStr.split('-').map(Number);
  if (!y || !m || !d) return [];

  interface MatchSample {
    mins: number;
    score: number;
    elevDeg: number;
    azimuthDeg: number;
    shadowBearingDeg: number;
  }
  const matches: MatchSample[] = [];

  for (let mins = 0; mins < 1440; mins += 5) {
    const utcMs =
      Date.UTC(y, m - 1, d, 0, mins, 0) - Math.round(params.tzOffsetHours * 3600 * 1000);
    const pos = solarPosition(params.lat, params.lng, new Date(utcMs));
    if (pos.elevationDeg <= 0) continue;
    const eDiff = Math.abs(pos.elevationDeg - params.elevationDeg);
    if (eDiff > elevTol) continue;
    let aDiff = 0;
    if (hasAz) {
      aDiff = angularDiffDeg(pos.shadowBearingDeg, params.shadowBearingDeg!);
      if (aDiff > azTol) continue;
    }
    const score = (1 - eDiff / elevTol) + (hasAz ? 1 - aDiff / azTol : 0);
    matches.push({
      mins,
      score,
      elevDeg: pos.elevationDeg,
      azimuthDeg: pos.azimuthDeg,
      shadowBearingDeg: pos.shadowBearingDeg,
    });
  }

  if (matches.length === 0) return [];

  const fmt = (mins: number) => {
    const h = Math.floor(mins / 60);
    const mm = mins % 60;
    return `${String(h).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
  };

  const windows: SolarTimeWindow[] = [];
  let startIdx = 0;
  for (let i = 1; i <= matches.length; i++) {
    if (i === matches.length || matches[i].mins - matches[i - 1].mins > 10) {
      const slice = matches.slice(startIdx, i);
      let best = slice[0];
      for (const s of slice) {
        if (s.score > best.score) best = s;
      }
      windows.push({
        startTime: fmt(slice[0].mins),
        endTime: fmt(slice[slice.length - 1].mins),
        peakMatchTime: fmt(best.mins),
        elevDeg: best.elevDeg,
        azimuthDeg: best.azimuthDeg,
        shadowBearingDeg: best.shadowBearingDeg,
      });
      startIdx = i;
    }
  }
  return windows;
}

export function estimateHeadingFromLitFace(
  sunAzimuthDeg: number,
  litSide: 'left' | 'right' | 'front' | 'back',
): { headingDeg: number; cardinal: string } {
  // Approximate camera view heading based on which wall faces the sun
  const offset =
    litSide === 'left' ? 90 : litSide === 'right' ? -90 : litSide === 'front' ? 180 : 0;
  const headingDeg = (((sunAzimuthDeg + offset) % 360) + 360) % 360;
  const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const cardinal = dirs[Math.round(headingDeg / 45) % 8];
  return { headingDeg, cardinal };
}
