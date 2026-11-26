import type { ClueMatch, EvidenceEntry, ScoredCountry } from './types';

interface CallingCodeItem {
  country: string;
  utc?: string;
}

interface DrivingSideItem {
  side: 'left' | 'right';
  switched?: string;
  note?: string;
}

interface TerritoryItem {
  name: string;
  region: string;
  subregion: string;
  status: string;
}

interface PrefixEntry {
  admin1?: string;
  admin2?: string;
  city?: string;
  note?: string;
}

interface RegionPackData {
  country: string;
  plates: Record<string, PrefixEntry[]>;
  areaCodes: Record<string, PrefixEntry[]>;
}

interface CluesBundle {
  callingCodes: Record<string, CallingCodeItem[]>;
  drivingSide: Record<string, DrivingSideItem>;
  territories: Record<string, TerritoryItem[]>;
  regionPacks: Record<string, RegionPackData>;
}

let bundleCache: CluesBundle | null = null;
let bundlePromise: Promise<CluesBundle> | null = null;

export async function loadCluesBundle(): Promise<CluesBundle> {
  if (bundleCache) return bundleCache;
  if (!bundlePromise) {
    const url = new URL('data/clues_tables.json', new URL('./', window.location.href)).href;
    bundlePromise = fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} fetching clues_tables.json`);
        return r.json() as Promise<CluesBundle>;
      })
      .then((data) => {
        bundleCache = data;
        return data;
      });
  }
  return bundlePromise;
}

const COUNTRY_ALIASES: Record<string, string> = {
  usa: 'united states',
  'united states of america': 'united states',
  uk: 'united kingdom',
  'great britain': 'united kingdom',
  'republic of korea': 'south korea',
  korea: 'south korea',
  turkiye: 'turkey',
  türkiye: 'turkey',
  czechia: 'czech republic',
  'russian federation': 'russia',
  'viet nam': 'vietnam',
  "cote d'ivoire": 'ivory coast',
  'uae': 'united arab emirates',
};

function normName(s: string): string {
  const low = s.toLowerCase().replace(/\(.*?\)/g, '').trim();
  return COUNTRY_ALIASES[low] ?? low;
}

export function matchCountryCodesByName(
  names: string[],
  catalogCountries: ScoredCountry[],
): string[] {
  const out = new Set<string>();
  const targets = names.map(normName).filter(Boolean);
  for (const c of catalogCountries) {
    const cn = normName(c.name);
    for (const t of targets) {
      if (cn === t || (t.length >= 4 && (cn.includes(t) || t.includes(cn)))) {
        out.add(c.code);
      }
    }
  }
  return Array.from(out);
}

export function lookupCallingCode(
  rawInput: string,
  bundle: CluesBundle,
  catalogCountries: ScoredCountry[],
): ClueMatch[] {
  const digits = rawInput.replace(/[^\d-]/g, '').replace(/^-+/, '');
  if (!digits) return [];

  const results: ClueMatch[] = [];
  const exact = bundle.callingCodes[digits];
  if (exact && exact.length > 0) {
    const names = exact.map((e) => e.country);
    const codes = matchCountryCodesByName(names, catalogCountries);
    results.push({
      id: `call-${digits}`,
      kind: 'calling-code',
      title: `Calling code +${digits} → ${names.join(', ')}`,
      subtitle: `Matches ${exact.map((e) => (e.utc ? `${e.country} (UTC${e.utc})` : e.country)).join(' · ')}`,
      countryCodes: codes,
      likelihoodRatio: 6.5,
      status: 'read',
    });
    return results;
  }

  // Try longest prefix match (up to 4 digits)
  for (let len = Math.min(4, digits.length); len >= 1; len--) {
    const prefix = digits.slice(0, len);
    const hit = bundle.callingCodes[prefix];
    if (hit && hit.length > 0) {
      const names = hit.map((e) => e.country);
      const codes = matchCountryCodesByName(names, catalogCountries);
      results.push({
        id: `call-pfx-${prefix}`,
        kind: 'calling-code',
        title: `Calling prefix +${prefix} → ${names.join(', ')}`,
        subtitle: `Longest international prefix match from "${rawInput.trim()}"`,
        countryCodes: codes,
        likelihoodRatio: 4.5,
        status: 'read',
      });
      break;
    }
  }
  return results;
}

export function lookupDrivingSide(
  side: 'left' | 'right',
  bundle: CluesBundle,
  catalogCountries: ScoredCountry[],
): ClueMatch {
  const names = Object.entries(bundle.drivingSide)
    .filter(([, v]) => v.side === side)
    .map(([k]) => k);
  const codes = matchCountryCodesByName(names, catalogCountries);
  // Fallback also checks catalogCountries driving attribute
  for (const c of catalogCountries) {
    if (c.driving === side && !codes.includes(c.code)) codes.push(c.code);
  }
  return {
    id: `drive-${side}`,
    kind: 'driving-side',
    title: `${side === 'left' ? 'Left-hand' : 'Right-hand'} traffic (${codes.length} catalog countries)`,
    subtitle:
      side === 'left'
        ? 'Favors UK, Ireland, Japan, Australia, New Zealand, India, Southern Africa, SE Asia'
        : 'Favors Continental Europe, Americas, China, Korea, Middle East',
    countryCodes: codes,
    likelihoodRatio: side === 'left' ? 3.5 : 2.2,
    status: 'observed',
  };
}

export function lookupPlateOrArea(
  rawQuery: string,
  bundle: CluesBundle,
  catalogCountries: ScoredCountry[],
  countryCodeHint?: string,
): ClueMatch[] {
  const q = rawQuery.trim();
  if (!q) return [];
  const qUp = q.toUpperCase();
  const matches: ClueMatch[] = [];

  for (const [cc, pack] of Object.entries(bundle.regionPacks)) {
    if (countryCodeHint && countryCodeHint !== cc) continue;
    const ccCodes = matchCountryCodesByName([pack.country], catalogCountries);

    const pHit = pack.plates[q] ?? pack.plates[qUp];
    if (pHit && pHit.length > 0) {
      const desc = pHit
        .map((e) => [e.admin1, e.admin2, e.city, e.note].filter(Boolean).join(' · '))
        .join(' | ');
      matches.push({
        id: `plate-${cc}-${qUp}`,
        kind: 'plate',
        title: `${pack.country} (${cc}) Plate "${q}" → ${pHit.map((e) => e.admin1 ?? e.city).filter(Boolean).join(', ')}`,
        subtitle: desc,
        countryCodes: ccCodes,
        likelihoodRatio: 5.0,
        status: 'read',
      });
    }

    const aHit = pack.areaCodes[q] ?? pack.areaCodes[q.replace(/^\+/, '')];
    if (aHit && aHit.length > 0) {
      const desc = aHit
        .map((e) => [e.admin1, e.admin2, e.city, e.note].filter(Boolean).join(' · '))
        .join(' | ');
      matches.push({
        id: `area-${cc}-${q}`,
        kind: 'area-code',
        title: `${pack.country} (${cc}) Area Code "${q}" → ${aHit.map((e) => e.admin1 ?? e.city).filter(Boolean).join(', ')}`,
        subtitle: desc,
        countryCodes: ccCodes,
        likelihoodRatio: 5.5,
        status: 'read',
      });
    }
  }
  return matches.slice(0, 8);
}

export function lookupTerritory(
  rawQuery: string,
  bundle: CluesBundle,
  catalogCountries: ScoredCountry[],
): ClueMatch[] {
  const q = rawQuery.trim().toLowerCase();
  if (q.length < 3) return [];
  const matches: ClueMatch[] = [];

  for (const [sov, items] of Object.entries(bundle.territories)) {
    for (const item of items) {
      if (
        item.name.toLowerCase().includes(q) ||
        sov.toLowerCase().includes(q) ||
        item.subregion.toLowerCase().includes(q)
      ) {
        const codes = matchCountryCodesByName([item.name, sov], catalogCountries);
        matches.push({
          id: `terr-${sov}-${item.name}`.replace(/\s+/g, '-'),
          kind: 'territory',
          title: `${item.name} (${sov})`,
          subtitle: `${item.status} · ${item.region} / ${item.subregion}`,
          countryCodes: codes,
          likelihoodRatio: 3.0,
          status: 'read',
        });
        if (matches.length >= 6) return matches;
      }
    }
  }
  return matches;
}

export async function searchAllClues(
  query: string,
  catalogCountries: ScoredCountry[],
  countryHint?: string,
): Promise<ClueMatch[]> {
  const bundle = await loadCluesBundle();
  const q = query.trim();
  if (!q) return [];

  const results: ClueMatch[] = [];
  const low = q.toLowerCase();

  if (low === 'left' || low === 'lht' || low.includes('left-hand')) {
    results.push(lookupDrivingSide('left', bundle, catalogCountries));
  } else if (low === 'right' || low === 'rht' || low.includes('right-hand')) {
    results.push(lookupDrivingSide('right', bundle, catalogCountries));
  }

  if (/^\+?\d[\d\s-]{0,7}$/.test(q)) {
    results.push(...lookupCallingCode(q, bundle, catalogCountries));
  }

  results.push(...lookupPlateOrArea(q, bundle, catalogCountries, countryHint));
  if (results.length < 6) {
    results.push(...lookupTerritory(q, bundle, catalogCountries));
  }

  return results;
}

/**
 * Extracts structured geo-sleuth evidence suggestions automatically from Qwen-VL output text.
 */
export async function extractAutoCluesFromVlmText(
  vlmText: string,
  catalogCountries: ScoredCountry[],
): Promise<EvidenceEntry[]> {
  if (!vlmText.trim()) return [];
  const bundle = await loadCluesBundle();
  const suggestions: EvidenceEntry[] = [];
  const seenIds = new Set<string>();

  const pushUnique = (entry: EvidenceEntry) => {
    if (seenIds.has(entry.id) || entry.countryCodes.length === 0) return;
    seenIds.add(entry.id);
    suggestions.push(entry);
  };

  // 1. Explicit +NNN calling codes in quoted VLM text
  const callRe = /(?:^|\s)\+(\d{1,3})(?=[\s\-)\].:,])/g;
  let m: RegExpExecArray | null;
  while ((m = callRe.exec(vlmText)) !== null) {
    for (const hit of lookupCallingCode(m[1], bundle, catalogCountries)) {
      pushUnique({
        id: `vlm-${hit.id}`,
        kind: 'calling-code',
        label: hit.title,
        detail: `Extracted +${m[1]} from Qwen-VL text readout`,
        status: 'read',
        countryCodes: hit.countryCodes,
        likelihoodRatio: 5.5,
        active: true,
        source: 'vlm-ocr',
      });
    }
  }

  // 2. Driving side checks
  const low = vlmText.toLowerCase();
  if (/left[- ]hand traffic|drive[sn] on the left|driving on the left/.test(low)) {
    const drive = lookupDrivingSide('left', bundle, catalogCountries);
    pushUnique({
      id: 'vlm-drive-left',
      kind: 'driving-side',
      label: 'Driving side: Left-hand traffic (VLM)',
      detail: 'Left-hand traffic noted by Qwen-VL clue check',
      status: 'observed',
      countryCodes: drive.countryCodes,
      likelihoodRatio: 3.2,
      active: true,
      source: 'vlm-ocr',
    });
  } else if (/right[- ]hand traffic|drive[sn] on the right|driving on the right/.test(low)) {
    const drive = lookupDrivingSide('right', bundle, catalogCountries);
    pushUnique({
      id: 'vlm-drive-right',
      kind: 'driving-side',
      label: 'Driving side: Right-hand traffic (VLM)',
      detail: 'Right-hand traffic noted by Qwen-VL clue check',
      status: 'observed',
      countryCodes: drive.countryCodes,
      likelihoodRatio: 2.0,
      active: true,
      source: 'vlm-ocr',
    });
  }

  // 3. Distinctive scripts
  if (/[\u3040-\u30ff]/.test(vlmText) || /\b(hiragana|katakana|kana|japanese)\b/i.test(vlmText)) {
    const codes = matchCountryCodesByName(['Japan'], catalogCountries);
    pushUnique({
      id: 'vlm-script-jp',
      kind: 'script',
      label: 'Japanese script / kana (VLM)',
      detail: 'Japanese characters or kana identified in frame',
      status: 'read',
      countryCodes: codes,
      likelihoodRatio: 6.0,
      active: true,
      source: 'vlm-ocr',
    });
  }
  if (/[\uac00-\ud7af]/.test(vlmText) || /\b(hangul|korean)\b/i.test(vlmText)) {
    const codes = matchCountryCodesByName(['South Korea'], catalogCountries);
    pushUnique({
      id: 'vlm-script-kr',
      kind: 'script',
      label: 'Korean Hangul script (VLM)',
      detail: 'Hangul script identified in frame',
      status: 'read',
      countryCodes: codes,
      likelihoodRatio: 6.0,
      active: true,
      source: 'vlm-ocr',
    });
  }
  if (/[\u0d80-\u0dff]/.test(vlmText) || /\bsinhala\b/i.test(vlmText)) {
    const codes = matchCountryCodesByName(['Sri Lanka'], catalogCountries);
    pushUnique({
      id: 'vlm-script-lk',
      kind: 'script',
      label: 'Sinhala script (VLM)',
      detail: 'Sinhala script strongly points to Sri Lanka',
      status: 'read',
      countryCodes: codes,
      likelihoodRatio: 6.0,
      active: true,
      source: 'vlm-ocr',
    });
  }
  if (/oʻ|gʻ|koʻchasi|o‘|g‘|\buzbek\b/i.test(vlmText)) {
    const codes = matchCountryCodesByName(['Uzbekistan'], catalogCountries);
    pushUnique({
      id: 'vlm-script-uz',
      kind: 'script',
      label: 'Uzbek script markers (VLM)',
      detail: 'Uzbek orthography detected in sign text',
      status: 'read',
      countryCodes: codes,
      likelihoodRatio: 5.0,
      active: true,
      source: 'vlm-ocr',
    });
  }

  return suggestions;
}
