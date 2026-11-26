import type {
  ActiveBackend,
  AnalysisMode,
  AnalysisResult,
  ConfidenceSummary,
  EngineSource,
  GeocellHotspot,
  GeocellPoint,
  GeocellPrediction,
  LeadsCatalog,
  ScoredCity,
  ScoredCountry,
  ScoredRegion,
  ScoredVisualAxis,
} from './types';

function haversineKm(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 6371;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}

/**
 * IEEE-754 Float16 (binary16) to Float32 decoder.
 * Works uniformly across all browsers without requiring native Float16Array.
 */
export function decodeFloat16Array(buffer: ArrayBuffer): Float32Array {
  const u16 = new Uint16Array(buffer);
  const f32 = new Float32Array(u16.length);

  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;

    if (e === 0) {
      f32[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    } else if (e === 0x1f) {
      f32[i] = f ? NaN : (s ? -Infinity : Infinity);
    } else {
      f32[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
  }
  return f32;
}

export function l2Normalize(vec: Float32Array): Float32Array {
  let normSq = 0;
  for (let i = 0; i < vec.length; i++) {
    normSq += vec[i] * vec[i];
  }
  const invNorm = normSq > 1e-12 ? 1 / Math.sqrt(normSq) : 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] * invNorm;
  }
  return out;
}

export function softmax(logits: Float32Array): Float32Array {
  const n = logits.length;
  const probs = new Float32Array(n);
  if (n === 0) return probs;

  let maxLogit = -Infinity;
  for (let i = 0; i < n; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    probs[i] = e;
    sum += e;
  }

  const invSum = sum > 0 ? 1 / sum : 1 / n;
  for (let i = 0; i < n; i++) {
    probs[i] *= invSum;
  }
  return probs;
}

export function computeDotScores(
  imageEmbed: Float32Array,
  embeddings: Float32Array,
  startRow: number,
  count: number,
  dim: number,
): Float32Array {
  const scores = new Float32Array(count);
  for (let r = 0; r < count; r++) {
    const base = (startRow + r) * dim;
    let dot = 0;
    for (let d = 0; d < dim; d++) {
      dot += imageEmbed[d] * embeddings[base + d];
    }
    scores[r] = dot;
  }
  return scores;
}

export function scoreWithTemperature(scores: Float32Array, logitScale: number): Float32Array {
  const logits = new Float32Array(scores.length);
  for (let i = 0; i < scores.length; i++) {
    logits[i] = scores[i] * logitScale;
  }
  return softmax(logits);
}

function findNearestCatalogCity(
  lat: number,
  lng: number,
  catalog: LeadsCatalog,
  countryNameByCode: Map<string, string>,
): { nearestCity?: string; nearestCountry?: string; distanceToCityKm?: number } {
  let bestDist = Infinity;
  let bestCity: (typeof catalog.cities)[number] | null = null;
  for (const c of catalog.cities) {
    const d = haversineKm(lat, lng, c.lat, c.lng);
    if (d < bestDist) {
      bestDist = d;
      bestCity = c;
    }
  }
  if (!bestCity) return {};
  return {
    nearestCity: bestCity.name,
    nearestCountry: countryNameByCode.get(bestCity.countryCode) ?? bestCity.countryCode,
    distanceToCityKm: bestDist,
  };
}

/**
 * Computes a calibrated 68th-percentile probability-weighted radius (km) around (lat, lng)
 * over the top sorted cells, avoiding distortion from distant 0.1% global tail cells.
 */
function computeWeightedRadius68(
  estLat: number,
  estLng: number,
  sortedIndices: number[],
  probs: Float32Array,
  getCoords: (idx: number) => [number, number],
): number {
  const pairs: Array<{ d: number; p: number }> = [];
  let totalP = 0;
  for (let k = 0; k < Math.min(120, sortedIndices.length); k++) {
    const idx = sortedIndices[k];
    const p = probs[idx];
    const [lat, lng] = getCoords(idx);
    const d = haversineKm(estLat, estLng, lat, lng);
    pairs.push({ d, p });
    totalP += p;
  }
  if (totalP <= 0) return 0;
  pairs.sort((a, b) => a.d - b.d);
  let cum = 0;
  const target = totalP * 0.68;
  for (const item of pairs) {
    cum += item.p;
    if (cum >= target) return item.d;
  }
  return pairs[pairs.length - 1]?.d ?? 0;
}

/**
 * Scores the 5,000-cell OSV-5M geocell probe head on top of StreetCLIP image_embeds.
 */
export function scoreGeocellEmbedding(
  imageEmbed: Float32Array,
  geocellHead: Float32Array,
  geocellCoords: Float32Array,
  catalog: LeadsCatalog,
  countryNameByCode: Map<string, string>,
  logitScale = 100.0,
): GeocellPrediction {
  const dim = catalog.dim;
  const nCells = Math.floor(geocellCoords.length / 2);
  const scores = computeDotScores(imageEmbed, geocellHead, 0, nCells, dim);
  const probs = scoreWithTemperature(scores, logitScale);

  let topIdx = 0;
  let topProb = -1;
  for (let i = 0; i < nCells; i++) {
    if (probs[i] > topProb) {
      topProb = probs[i];
      topIdx = i;
    }
  }

  const topCellLat = geocellCoords[topIdx * 2];
  const topCellLng = geocellCoords[topIdx * 2 + 1];

  // Spherical probability-weighted mean within 300 km of top-1 cell
  let wx = 0;
  let wy = 0;
  let wz = 0;
  let wSum = 0;
  for (let i = 0; i < nCells; i++) {
    const p = probs[i];
    if (p < 1e-4) continue;
    const lat = geocellCoords[i * 2];
    const lng = geocellCoords[i * 2 + 1];
    if (haversineKm(topCellLat, topCellLng, lat, lng) <= 300) {
      const latR = (lat * Math.PI) / 180;
      const lngR = (lng * Math.PI) / 180;
      wx += p * Math.cos(latR) * Math.cos(lngR);
      wy += p * Math.cos(latR) * Math.sin(lngR);
      wz += p * Math.sin(latR);
      wSum += p;
    }
  }

  const estLat =
    wSum > 0 ? (Math.atan2(wz / wSum, Math.hypot(wx / wSum, wy / wSum)) * 180) / Math.PI : topCellLat;
  const estLng = wSum > 0 ? (Math.atan2(wy / wSum, wx / wSum) * 180) / Math.PI : topCellLng;

  const indices = Array.from({ length: nCells }, (_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);
  const top120 = indices.slice(0, 120);

  const spreadKm = computeWeightedRadius68(estLat, estLng, top120, probs, (idx) => [
    geocellCoords[idx * 2],
    geocellCoords[idx * 2 + 1],
  ]);

  const heatmap: GeocellPoint[] = [];
  for (let k = 0; k < top120.length; k++) {
    const idx = top120[k];
    const p = probs[idx];
    if (p < 0.0005 && heatmap.length >= 12) break;
    heatmap.push({
      cellId: idx,
      lat: geocellCoords[idx * 2],
      lng: geocellCoords[idx * 2 + 1],
      prob: p,
    });
  }

  const hotspots: GeocellHotspot[] = [];
  for (const idx of top120) {
    const lat = geocellCoords[idx * 2];
    const lng = geocellCoords[idx * 2 + 1];
    const tooClose = hotspots.some((h) => haversineKm(h.lat, h.lng, lat, lng) < 120);
    if (tooClose) continue;
    const nearest = findNearestCatalogCity(lat, lng, catalog, countryNameByCode);
    hotspots.push({
      cellId: idx,
      lat,
      lng,
      prob: probs[idx],
      ...nearest,
    });
    if (hotspots.length >= 5) break;
  }

  const nearestToEst = findNearestCatalogCity(estLat, estLng, catalog, countryNameByCode);

  return {
    source: 'streetclip-5k',
    numCells: nCells,
    lat: estLat,
    lng: estLng,
    topCellLat,
    topCellLng,
    topCellProb: topProb,
    spreadKm,
    hotspots,
    heatmap,
    ...nearestToEst,
  };
}

function getOsv5mSubcellLatLng(
  cellIdx: number,
  regOffsets: Float32Array,
  quadtreeF32: Float32Array,
): [number, number] {
  const base = cellIdx * 6;
  const meanLat = quadtreeF32[base];
  const meanLon = quadtreeF32[base + 1];
  const upLat = quadtreeF32[base + 2];
  const upLon = quadtreeF32[base + 3];
  const downLat = quadtreeF32[base + 4];
  const downLon = quadtreeF32[base + 5];

  const offLat = regOffsets[cellIdx * 2];
  const offLon = regOffsets[cellIdx * 2 + 1];
  const sLat = offLat > 0 ? upLat : downLat;
  const sLon = offLon > 0 ? upLon : downLon;

  const normLat = Math.max(-1, Math.min(1, meanLat + offLat * sLat));
  const normLon = Math.max(-1, Math.min(1, meanLon + offLon * sLon));
  return [normLat * 90, normLon * 180];
}

/**
 * Scores OSV-5M Baseline ONNX outputs (11,399 quadtree logits + sub-cell reg offsets)
 * and optionally derives country/region/city rankings for standalone OSV-5M mode.
 */
export function scoreOsv5mOutputs(
  classLogits: Float32Array,
  regOffsets: Float32Array,
  quadtreeF32: Float32Array,
  catalog: LeadsCatalog,
  meta: { latencyMs: number; cropCount: number; mode: AnalysisMode; backend: ActiveBackend },
): AnalysisResult {
  const nCells = classLogits.length;
  const probs = softmax(classLogits);
  const countryNameByCode = new Map(catalog.countries.map((c) => [c.code, c.name]));

  let topIdx = 0;
  let topProb = -1;
  for (let i = 0; i < nCells; i++) {
    if (probs[i] > topProb) {
      topProb = probs[i];
      topIdx = i;
    }
  }

  const [estLat, estLng] = getOsv5mSubcellLatLng(topIdx, regOffsets, quadtreeF32);
  const topCellLat = quadtreeF32[topIdx * 6] * 90;
  const topCellLng = quadtreeF32[topIdx * 6 + 1] * 180;

  const indices = Array.from({ length: nCells }, (_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);
  const top120 = indices.slice(0, 120);

  const spreadKm = computeWeightedRadius68(estLat, estLng, top120, probs, (idx) =>
    getOsv5mSubcellLatLng(idx, regOffsets, quadtreeF32),
  );

  const heatmap: GeocellPoint[] = [];
  for (let k = 0; k < top120.length; k++) {
    const idx = top120[k];
    const p = probs[idx];
    if (p < 0.0005 && heatmap.length >= 12) break;
    const [lat, lng] = getOsv5mSubcellLatLng(idx, regOffsets, quadtreeF32);
    heatmap.push({ cellId: idx, lat, lng, prob: p });
  }

  const hotspots: GeocellHotspot[] = [];
  for (const idx of top120) {
    const [lat, lng] = getOsv5mSubcellLatLng(idx, regOffsets, quadtreeF32);
    const tooClose = hotspots.some((h) => haversineKm(h.lat, h.lng, lat, lng) < 120);
    if (tooClose) continue;
    const nearest = findNearestCatalogCity(lat, lng, catalog, countryNameByCode);
    hotspots.push({
      cellId: idx,
      lat,
      lng,
      prob: probs[idx],
      ...nearest,
    });
    if (hotspots.length >= 5) break;
  }

  const nearestToEst = findNearestCatalogCity(estLat, estLng, catalog, countryNameByCode);
  const geocell: GeocellPrediction = {
    source: 'osv5m-11k',
    numCells: nCells,
    lat: estLat,
    lng: estLng,
    topCellLat,
    topCellLng,
    topCellProb: topProb,
    spreadKm,
    hotspots,
    heatmap,
    ...nearestToEst,
  };

  // Derive spatial country/region/city rankings from the top 120 OSV-5M cells
  const countryWeights = new Float32Array(catalog.countries.length);
  const regionWeights = new Float32Array(catalog.regions.length);
  const cityWeights = new Float32Array(catalog.cities.length);

  let top120Mass = 0;
  for (const idx of top120) {
    const p = probs[idx];
    top120Mass += p;
    const [lat, lng] = getOsv5mSubcellLatLng(idx, regOffsets, quadtreeF32);

    // Closest city gets weight (or split via soft inverse distance to top nearby city)
    let bestCityIdx = -1;
    let bestCityDist = Infinity;
    for (let c = 0; c < catalog.cities.length; c++) {
      const d = haversineKm(lat, lng, catalog.cities[c].lat, catalog.cities[c].lng);
      if (d < bestCityDist) {
        bestCityDist = d;
        bestCityIdx = c;
      }
    }
    if (bestCityIdx >= 0) {
      cityWeights[bestCityIdx] += p;
      const cc = catalog.cities[bestCityIdx].countryCode;
      const cIdx = catalog.countries.findIndex((x) => x.code === cc);
      if (cIdx >= 0) countryWeights[cIdx] += p;
    }

    let bestRegIdx = -1;
    let bestRegDist = Infinity;
    for (let r = 0; r < catalog.regions.length; r++) {
      const d = haversineKm(lat, lng, catalog.regions[r].lat, catalog.regions[r].lng);
      if (d < bestRegDist) {
        bestRegDist = d;
        bestRegIdx = r;
      }
    }
    if (bestRegIdx >= 0) {
      regionWeights[bestRegIdx] += p;
    }
  }

  const normMass = top120Mass > 0 ? top120Mass : 1;

  const countries: ScoredCountry[] = catalog.countries
    .map((c, idx) => ({
      ...c,
      score: countryWeights[idx] / normMass,
      prob: countryWeights[idx] / normMass,
      rank: 0,
    }))
    .sort((a, b) => b.prob - a.prob)
    .map((c, idx) => ({ ...c, rank: idx + 1 }));

  const top3Codes = new Set(countries.slice(0, 3).map((c) => c.code));

  const baseRegions = catalog.regions.map((r, idx) => ({
    ...r,
    countryName: countryNameByCode.get(r.countryCode) ?? r.countryCode,
    score: regionWeights[idx] / normMass,
    prob: regionWeights[idx] / normMass,
    hierarchicalProb: regionWeights[idx] / normMass,
    rank: 0,
  }));

  const regionsGlobal: ScoredRegion[] = [...baseRegions]
    .sort((a, b) => b.prob - a.prob)
    .map((r, idx) => ({ ...r, rank: idx + 1 }));
  const regionsHierarchical: ScoredRegion[] = [...baseRegions]
    .filter((r) => top3Codes.has(r.countryCode))
    .sort((a, b) => b.hierarchicalProb - a.hierarchicalProb)
    .map((r, idx) => ({ ...r, rank: idx + 1 }));

  const baseCities = catalog.cities.map((c, idx) => ({
    ...c,
    countryName: countryNameByCode.get(c.countryCode) ?? c.countryCode,
    score: cityWeights[idx] / normMass,
    prob: cityWeights[idx] / normMass,
    hierarchicalProb: cityWeights[idx] / normMass,
    rank: 0,
  }));

  const citiesGlobal: ScoredCity[] = [...baseCities]
    .sort((a, b) => b.prob - a.prob)
    .map((c, idx) => ({ ...c, rank: idx + 1 }));
  const citiesHierarchical: ScoredCity[] = [...baseCities]
    .filter((c) => top3Codes.has(c.countryCode))
    .sort((a, b) => b.hierarchicalProb - a.hierarchicalProb)
    .map((c, idx) => ({ ...c, rank: idx + 1 }));

  const topC = countries[0];
  const topCity = citiesHierarchical[0] ?? citiesGlobal[0];
  const top1Prob = topC?.prob ?? 0;
  const top2Prob = countries[1]?.prob ?? 0;
  const top1Margin = Math.max(0, top1Prob - top2Prob);

  let entropy = 0;
  for (const c of countries) {
    if (c.prob > 1e-12) entropy -= c.prob * Math.log(c.prob);
  }
  const entropyNormalized = Math.min(1, entropy / Math.log(Math.max(2, countries.length)));

  const tier: ConfidenceSummary['tier'] =
    top1Prob >= 0.45 && top1Margin >= 0.18
      ? 'High Signal'
      : top1Prob >= 0.22 || top1Margin >= 0.08
        ? 'Moderate Lead'
        : 'Ambiguous';

  const brief = topC && topCity
    ? `OSV-5M Baseline localizes this view near ${topCity.name}, ${topC.name} (${estLat.toFixed(2)}°, ${estLng.toFixed(2)}°, top cell ${(topProb * 100).toFixed(1)}%). Switch to Dual Mode to add StreetCLIP visual clues and custom text probes.`
    : `OSV-5M Baseline estimates coordinates (${estLat.toFixed(2)}°, ${estLng.toFixed(2)}°).`;

  return {
    imageEmbed: new Float32Array(0),
    engineSource: 'osv5m-q4',
    hasVisualClues: false,
    countries,
    regionsGlobal,
    regionsHierarchical,
    citiesGlobal,
    citiesHierarchical,
    visualAxes: [],
    confidence: {
      tier,
      levels: computeConfidenceLevels(countries, regionsHierarchical, citiesHierarchical),
      bimodalWarning: detectBimodalHotspots(geocell),
      entropyNormalized,
      top1Margin,
      top1Prob,
      brief,
    },
    geocell,
    latencyMs: meta.latencyMs,
    cropCount: meta.cropCount,
    mode: meta.mode,
    backend: meta.backend,
  };
}

export function computeConfidenceLevels(
  countries: ScoredCountry[],
  regionsHierarchical: ScoredRegion[],
  citiesHierarchical: ScoredCity[],
): ConfidenceSummary['levels'] {
  const c1 = countries[0]?.prob ?? 0;
  const c2 = countries[1]?.prob ?? 0;
  const cMargin = Math.max(0, c1 - c2);
  const country: ConfidenceSummary['levels']['country'] =
    c1 >= 0.72 && cMargin >= 0.35
      ? 'High'
      : c1 >= 0.4 || cMargin >= 0.15
        ? 'Medium'
        : 'Low';

  const r1 = regionsHierarchical[0]?.hierarchicalProb ?? 0;
  const r2 = regionsHierarchical[1]?.hierarchicalProb ?? 0;
  const rMargin = Math.max(0, r1 - r2);
  const region: ConfidenceSummary['levels']['region'] =
    r1 >= 0.45 && rMargin >= 0.18 ? 'High' : r1 >= 0.2 ? 'Medium' : 'Low';

  const ci1 = citiesHierarchical[0]?.hierarchicalProb ?? 0;
  const ci2 = citiesHierarchical[1]?.hierarchicalProb ?? 0;
  const ciMargin = Math.max(0, ci1 - ci2);
  const city: ConfidenceSummary['levels']['city'] =
    ci1 >= 0.35 && ciMargin >= 0.14 ? 'High' : ci1 >= 0.14 ? 'Medium' : 'Low';

  return { country, region, city };
}

export function detectBimodalHotspots(
  geocell?: GeocellPrediction,
): ConfidenceSummary['bimodalWarning'] | undefined {
  if (!geocell || geocell.hotspots.length < 2) return undefined;
  const [h0, h1] = geocell.hotspots;
  if (h1.prob < 0.35 * h0.prob) return undefined;
  const sep = haversineKm(h0.lat, h0.lng, h1.lat, h1.lng);
  if (sep < 250) return undefined;
  const label = (h: GeocellHotspot) =>
    h.nearestCity
      ? `${h.nearestCity}, ${h.nearestCountry ?? ''}`.trim().replace(/,$/, '')
      : `${h.lat.toFixed(1)}°, ${h.lng.toFixed(1)}°`;
  return {
    hotspotA: label(h0),
    hotspotB: label(h1),
    separationKm: Math.round(sep),
  };
}

export function buildConfidenceSummary(
  countries: ScoredCountry[],
  regionsHierarchical: ScoredRegion[],
  citiesHierarchical: ScoredCity[],
  visualAxes: ScoredVisualAxis[],
  geocell?: GeocellPrediction,
): ConfidenceSummary {
  const n = countries.length;
  let entropy = 0;
  for (let i = 0; i < n; i++) {
    const p = countries[i].prob;
    if (p > 1e-12) {
      entropy -= p * Math.log(p);
    }
  }
  const maxEntropy = Math.log(Math.max(2, n));
  const entropyNormalized = Math.min(1, Math.max(0, entropy / maxEntropy));
  const top1Prob = countries[0]?.prob ?? 0;
  const top2Prob = countries[1]?.prob ?? 0;
  const top1Margin = Math.max(0, top1Prob - top2Prob);

  let tier: ConfidenceSummary['tier'] = 'Ambiguous';
  if (top1Prob >= 0.45 && top1Margin >= 0.18) {
    tier = 'High Signal';
  } else if (top1Prob >= 0.22 || top1Margin >= 0.08) {
    tier = 'Moderate Lead';
  }

  const levels = computeConfidenceLevels(countries, regionsHierarchical, citiesHierarchical);
  const bimodalWarning = detectBimodalHotspots(geocell);

  const topCountry = countries[0];
  const topCity = citiesHierarchical[0];
  const envCue = visualAxes.find((a) => a.id === 'environment')?.topItem.label.toLowerCase();
  const archCue = visualAxes.find((a) => a.id === 'architecture')?.topItem.label.toLowerCase();
  const scriptCue = visualAxes.find((a) => a.id === 'signage')?.topItem.label.toLowerCase();

  const parts: string[] = [];
  if (topCountry) {
    const pct = (topCountry.prob * 100).toFixed(1);
    parts.push(`Primary country signal points to ${topCountry.name} (${pct}%), driving on the ${topCountry.driving}`);
  }
  if (topCity) {
    parts.push(`with strongest local convergence near ${topCity.name}, ${topCity.region}`);
  }
  const cueBits = [envCue, archCue, scriptCue].filter(Boolean);
  if (cueBits.length > 0) {
    parts.push(`corroborated by ${cueBits.slice(0, 2).join(' and ')}`);
  }

  return {
    tier,
    levels,
    bimodalWarning,
    entropyNormalized,
    top1Margin,
    top1Prob,
    brief: parts.length > 0 ? `${parts.join('; ')}.` : 'Insufficient visual evidence to rank geographic leads.',
  };
}

/**
 * Executes StreetCLIP hierarchical linear probing (Country -> Region -> City),
 * 6 independent visual-clue axes, and optional OSV-5M 5k/11k coordinate probing.
 */
export function scoreImageEmbedding(
  rawImageEmbed: Float32Array,
  catalog: LeadsCatalog,
  embeddings: Float32Array,
  meta: {
    engineSource: EngineSource;
    latencyMs: number;
    cropCount: number;
    mode: AnalysisMode;
    backend: ActiveBackend;
    geocellHead?: Float32Array | null;
    geocellCoords?: Float32Array | null;
    osv5mOverride?: GeocellPrediction | null;
  },
): AnalysisResult {
  const imageEmbed = l2Normalize(rawImageEmbed);
  const { dim, logitScale, offsets } = catalog;
  const countryNameByCode = new Map(catalog.countries.map((c) => [c.code, c.name]));

  // 1. Countries
  const [cStart, cCount] = offsets.countries;
  const countryScores = computeDotScores(imageEmbed, embeddings, cStart, cCount, dim);
  const countryProbs = scoreWithTemperature(countryScores, logitScale);

  const countries: ScoredCountry[] = catalog.countries
    .map((c, idx) => ({
      ...c,
      score: countryScores[idx],
      prob: countryProbs[idx],
      rank: 0,
    }))
    .sort((a, b) => b.prob - a.prob)
    .map((c, idx) => ({ ...c, rank: idx + 1 }));

  const countryProbMap = new Map(countries.map((c) => [c.code, c.prob]));
  const top3CountryCodes = new Set(countries.slice(0, 3).map((c) => c.code));

  // 2. Regions (Global & Hierarchical)
  const [rStart, rCount] = offsets.regions;
  const regionScores = computeDotScores(imageEmbed, embeddings, rStart, rCount, dim);
  const regionGlobalProbs = scoreWithTemperature(regionScores, logitScale);

  const regionHierProbMap = new Map<number, number>();
  for (const code of top3CountryCodes) {
    const indices: number[] = [];
    for (let i = 0; i < catalog.regions.length; i++) {
      if (catalog.regions[i].countryCode === code) indices.push(i);
    }
    if (indices.length === 0) continue;
    const subScores = new Float32Array(indices.length);
    for (let k = 0; k < indices.length; k++) subScores[k] = regionScores[indices[k]];
    const subProbs = scoreWithTemperature(subScores, logitScale);
    const parentProb = countryProbMap.get(code) ?? 0;
    for (let k = 0; k < indices.length; k++) {
      regionHierProbMap.set(indices[k], parentProb * subProbs[k]);
    }
  }

  const baseRegions = catalog.regions.map((r, idx) => ({
    ...r,
    countryName: countryNameByCode.get(r.countryCode) ?? r.countryCode,
    score: regionScores[idx],
    prob: regionGlobalProbs[idx],
    hierarchicalProb: regionHierProbMap.get(idx) ?? 0,
    rank: 0,
  }));

  const regionsGlobal: ScoredRegion[] = [...baseRegions]
    .sort((a, b) => b.prob - a.prob)
    .map((r, idx) => ({ ...r, rank: idx + 1 }));

  const regionsHierarchical: ScoredRegion[] = [...baseRegions]
    .filter((r) => top3CountryCodes.has(r.countryCode))
    .sort((a, b) => b.hierarchicalProb - a.hierarchicalProb)
    .map((r, idx) => ({ ...r, rank: idx + 1 }));

  // 3. Cities (Global & Hierarchical)
  const [cityStart, cityCount] = offsets.cities;
  const cityScores = computeDotScores(imageEmbed, embeddings, cityStart, cityCount, dim);
  const cityGlobalProbs = scoreWithTemperature(cityScores, logitScale);

  const cityHierProbMap = new Map<number, number>();
  for (const code of top3CountryCodes) {
    const indices: number[] = [];
    for (let i = 0; i < catalog.cities.length; i++) {
      if (catalog.cities[i].countryCode === code) indices.push(i);
    }
    if (indices.length === 0) continue;
    const subScores = new Float32Array(indices.length);
    for (let k = 0; k < indices.length; k++) subScores[k] = cityScores[indices[k]];
    const subProbs = scoreWithTemperature(subScores, logitScale);
    const parentProb = countryProbMap.get(code) ?? 0;
    for (let k = 0; k < indices.length; k++) {
      cityHierProbMap.set(indices[k], parentProb * subProbs[k]);
    }
  }

  const baseCities = catalog.cities.map((c, idx) => ({
    ...c,
    countryName: countryNameByCode.get(c.countryCode) ?? c.countryCode,
    score: cityScores[idx],
    prob: cityGlobalProbs[idx],
    hierarchicalProb: cityHierProbMap.get(idx) ?? 0,
    rank: 0,
  }));

  const citiesGlobal: ScoredCity[] = [...baseCities]
    .sort((a, b) => b.prob - a.prob)
    .map((c, idx) => ({ ...c, rank: idx + 1 }));

  const citiesHierarchical: ScoredCity[] = [...baseCities]
    .filter((c) => top3CountryCodes.has(c.countryCode))
    .sort((a, b) => b.hierarchicalProb - a.hierarchicalProb)
    .map((c, idx) => ({ ...c, rank: idx + 1 }));

  // 4. Visual Lead Axes
  const visualAxes: ScoredVisualAxis[] = catalog.visualAxes.map((axis) => {
    const [aStart, aCount] = offsets.visualAxes[axis.id];
    const axisScores = computeDotScores(imageEmbed, embeddings, aStart, aCount, dim);
    const axisProbs = scoreWithTemperature(axisScores, logitScale);

    const items = axis.items
      .map((item, idx) => ({
        ...item,
        score: axisScores[idx],
        prob: axisProbs[idx],
      }))
      .sort((a, b) => b.prob - a.prob);

    return {
      id: axis.id,
      title: axis.title,
      items,
      topItem: items[0],
    };
  });

  // 5. Coordinate Probe (StreetCLIP 5k, or OSV-5M 11k override in Dual Mode)
  const streetclipGeocell =
    meta.geocellHead && meta.geocellCoords
      ? scoreGeocellEmbedding(
          imageEmbed,
          meta.geocellHead,
          meta.geocellCoords,
          catalog,
          countryNameByCode,
          100.0,
        )
      : undefined;

  let geocell = streetclipGeocell;
  if (meta.osv5mOverride) {
    geocell = {
      ...meta.osv5mOverride,
      secondaryCoord: streetclipGeocell
        ? {
            lat: streetclipGeocell.lat,
            lng: streetclipGeocell.lng,
            label: 'StreetCLIP 5k',
            divergenceKm: haversineKm(
              meta.osv5mOverride.lat,
              meta.osv5mOverride.lng,
              streetclipGeocell.lat,
              streetclipGeocell.lng,
            ),
          }
        : undefined,
    };
  }

  const confidence = buildConfidenceSummary(
    countries,
    regionsHierarchical,
    citiesHierarchical,
    visualAxes,
    geocell,
  );

  return {
    imageEmbed,
    engineSource: meta.engineSource,
    hasVisualClues: true,
    countries,
    regionsGlobal,
    regionsHierarchical,
    citiesGlobal,
    citiesHierarchical,
    visualAxes,
    confidence,
    geocell,
    latencyMs: meta.latencyMs,
    cropCount: meta.cropCount,
    mode: meta.mode,
    backend: meta.backend,
  };
}
