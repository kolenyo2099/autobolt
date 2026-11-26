import type { OsmPin } from './types';

export type OverpassPresetId =
  | 'rail_bridge_water'
  | 'power_rail'
  | 'tram_street'
  | 'roundabout_water'
  | 'stadium_track'
  | 'tower_landmark';

export interface OverpassPresetSpec {
  id: OverpassPresetId;
  label: string;
  description: string;
}

export const OVERPASS_PRESETS: ReadonlyArray<OverpassPresetSpec> = [
  {
    id: 'rail_bridge_water',
    label: 'Rail / Highway Bridge over Water',
    description: 'Railway or highway bridges crossing rivers or canals',
  },
  {
    id: 'tram_street',
    label: 'Street-Running Tram / Light Rail',
    description: 'Urban streets sharing right-of-way with tram tracks',
  },
  {
    id: 'power_rail',
    label: 'High-Voltage Power Lines near Rail',
    description: 'Transmission towers within search radius',
  },
  {
    id: 'stadium_track',
    label: 'Sports Stadium / Running Track',
    description: 'Athletic ovals and school running tracks',
  },
  {
    id: 'roundabout_water',
    label: 'Waterfront Promenade / Pier',
    description: 'Piers, quays, or marinas along a coastline or river',
  },
  {
    id: 'tower_landmark',
    label: 'Observation Tower / Spire',
    description: 'High-visibility towers and monuments',
  },
];

const OVERPASS_ENDPOINTS = [
  'https://overpass-api.de/api/interpreter',
  'https://overpass.kumi.systems/api/interpreter',
  'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
];

// Serialize all Overpass calls to respect geo-sleuth's rate-limit rule (no parallel Overpass hits)
let overpassQueue: Promise<unknown> = Promise.resolve();

function enqueueOverpass<T>(task: () => Promise<T>): Promise<T> {
  const next = overpassQueue.then(task, task);
  overpassQueue = next.catch(() => {});
  return next;
}

export async function searchNominatimPoi(
  query: string,
  center?: { lat: number; lng: number } | null,
  radiusKm = 50,
): Promise<OsmPin[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];

  const params = new URLSearchParams({
    q: trimmed,
    format: 'jsonv2',
    limit: '10',
    addressdetails: '1',
  });

  if (center) {
    const dLat = radiusKm / 111;
    const dLng = radiusKm / (111 * Math.max(0.25, Math.cos((center.lat * Math.PI) / 180)));
    const west = (center.lng - dLng).toFixed(4);
    const north = (center.lat + dLat).toFixed(4);
    const east = (center.lng + dLng).toFixed(4);
    const south = (center.lat - dLat).toFixed(4);
    // Bias results toward candidate viewbox without hard-excluding wider matches
    params.set('viewbox', `${west},${north},${east},${south}`);
  }

  const resp = await fetch(`https://nominatim.openstreetmap.org/search?${params.toString()}`, {
    headers: { Accept: 'application/json' },
  });
  if (!resp.ok) {
    throw new Error(`Nominatim HTTP ${resp.status}`);
  }

  const rows = (await resp.json()) as Array<{
    place_id: number;
    display_name: string;
    name?: string;
    type?: string;
    lat: string;
    lon: string;
  }>;

  return rows
    .map((r) => ({
      id: `nom-${r.place_id}`,
      name: r.name || r.display_name.split(',')[0] || trimmed,
      category: r.type || 'POI',
      lat: Number(r.lat),
      lng: Number(r.lon),
    }))
    .filter((p) => Number.isFinite(p.lat) && Number.isFinite(p.lng));
}

function buildOverpassQuery(
  preset: OverpassPresetId,
  lat: number,
  lng: number,
  radiusM: number,
): string {
  const around = `(around:${Math.round(radiusM)},${lat.toFixed(5)},${lng.toFixed(5)})`;
  switch (preset) {
    case 'rail_bridge_water':
      return `[out:json][timeout:18];(way["bridge"="yes"]["railway"]${around};way["bridge"="yes"]["highway"~"motorway|trunk|primary"]${around};);out center 18;`;
    case 'tram_street':
      return `[out:json][timeout:18];(way["railway"="tram"]${around};node["railway"="tram_stop"]${around};);out center 18;`;
    case 'power_rail':
      return `[out:json][timeout:18];(node["power"="tower"]${around};);out center 20;`;
    case 'stadium_track':
      return `[out:json][timeout:18];(way["leisure"="track"]${around};way["leisure"="stadium"]${around};);out center 18;`;
    case 'roundabout_water':
      return `[out:json][timeout:18];(way["man_made"="pier"]${around};way["leisure"="marina"]${around};);out center 18;`;
    case 'tower_landmark':
      return `[out:json][timeout:18];(node["man_made"="tower"]${around};way["man_made"="tower"]${around};);out center 18;`;
  }
}

export function queryOverpassPreset(params: {
  preset: OverpassPresetId;
  lat: number;
  lng: number;
  radiusM?: number;
}): Promise<{ pins: OsmPin[]; mirrorUsed: string }> {
  const radiusM = Math.min(20000, Math.max(500, params.radiusM ?? 6000));
  const ql = buildOverpassQuery(params.preset, params.lat, params.lng, radiusM);

  return enqueueOverpass(async () => {
    let lastErr = 'No Overpass mirror responded';
    for (const endpoint of OVERPASS_ENDPOINTS) {
      try {
        const resp = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: `data=${encodeURIComponent(ql)}`,
        });
        if (!resp.ok) {
          lastErr = `${new URL(endpoint).host} HTTP ${resp.status}`;
          continue;
        }
        const json = (await resp.json()) as {
          elements?: Array<{
            id: number;
            type: string;
            lat?: number;
            lon?: number;
            center?: { lat: number; lon: number };
            tags?: Record<string, string>;
          }>;
        };
        const pins: OsmPin[] = [];
        for (const el of json.elements ?? []) {
          const pLat = el.lat ?? el.center?.lat;
          const pLng = el.lon ?? el.center?.lon;
          if (typeof pLat !== 'number' || typeof pLng !== 'number') continue;
          const name =
            el.tags?.name ||
            el.tags?.ref ||
            el.tags?.railway ||
            el.tags?.leisure ||
            el.tags?.man_made ||
            `${el.type} #${el.id}`;
          pins.push({
            id: `osm-${el.type}-${el.id}`,
            name,
            category: params.preset.replace(/_/g, ' '),
            lat: pLat,
            lng: pLng,
          });
          if (pins.length >= 20) break;
        }
        return { pins, mirrorUsed: new URL(endpoint).host };
      } catch (err) {
        lastErr = err instanceof Error ? err.message : String(err);
      }
    }
    throw new Error(`Overpass mirrors busy (${lastErr}). Try Nominatim POI search or a smaller radius.`);
  });
}
