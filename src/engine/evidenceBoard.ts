import { buildConfidenceSummary, detectBimodalHotspots } from './scoring';
import type { AnalysisResult, EvidenceEntry, ScoredCity, ScoredRegion } from './types';

/**
 * Applies Bayesian evidence likelihood ratios (from lookup tables, VLM OCR, or solar math)
 * to raw StreetCLIP / OSV-5M scores without silently erasing non-matching candidates.
 * Matches geo-sleuth's board.py rule: inferred evidence is clamped to [1/3, 3];
 * read/computed/observed evidence is clamped to [0.15, 8].
 */
export function applyEvidenceToResult(
  raw: AnalysisResult,
  entries: EvidenceEntry[],
): AnalysisResult {
  const active = entries.filter((e) => e.active && e.countryCodes.length > 0);
  if (active.length === 0) return raw;

  const multByCountry = new Map<string, number>();
  for (const c of raw.countries) {
    multByCountry.set(c.code, 1);
  }

  for (const entry of active) {
    const minLr = entry.status === 'inferred' ? 1 / 3 : 0.15;
    const maxLr = entry.status === 'inferred' ? 3 : 8;
    const lr = Math.max(minLr, Math.min(maxLr, entry.likelihoodRatio));
    const matchSet = new Set(entry.countryCodes);
    const nonMatchPenalty = lr > 1 ? 1 / Math.sqrt(lr) : Math.min(1.5, 1 / lr);

    for (const c of raw.countries) {
      const prev = multByCountry.get(c.code) ?? 1;
      multByCountry.set(c.code, prev * (matchSet.has(c.code) ? lr : nonMatchPenalty));
    }
  }

  let totalMass = 0;
  const weighted = raw.countries.map((c) => {
    const rawP = Math.max(1e-7, c.prob);
    const w = rawP * (multByCountry.get(c.code) ?? 1);
    totalMass += w;
    return { c, w };
  });

  const norm = totalMass > 0 ? 1 / totalMass : 1;
  const countries = weighted
    .map(({ c, w }) => ({
      ...c,
      prob: w * norm,
    }))
    .sort((a, b) => b.prob - a.prob)
    .map((c, idx) => ({ ...c, rank: idx + 1 }));

  const oldProbByCode = new Map(raw.countries.map((c) => [c.code, Math.max(1e-6, c.prob)]));
  const newProbByCode = new Map(countries.map((c) => [c.code, c.prob]));
  const top3Codes = new Set(countries.slice(0, 3).map((c) => c.code));

  const scaleFactor = (code: string) =>
    (newProbByCode.get(code) ?? 0) / (oldProbByCode.get(code) ?? 1e-6);

  const regionsHierarchical: ScoredRegion[] = raw.regionsGlobal
    .filter((r) => top3Codes.has(r.countryCode))
    .map((r) => {
      const origHier =
        raw.regionsHierarchical.find((hr) => hr.name === r.name && hr.countryCode === r.countryCode)
          ?.hierarchicalProb ?? r.prob;
      return {
        ...r,
        hierarchicalProb: origHier * scaleFactor(r.countryCode),
      };
    })
    .sort((a, b) => b.hierarchicalProb - a.hierarchicalProb)
    .map((r, idx) => ({ ...r, rank: idx + 1 }));

  const citiesHierarchical: ScoredCity[] = raw.citiesGlobal
    .filter((c) => top3Codes.has(c.countryCode))
    .map((c) => {
      const origHier =
        raw.citiesHierarchical.find((hc) => hc.name === c.name && hc.countryCode === c.countryCode)
          ?.hierarchicalProb ?? c.prob;
      return {
        ...c,
        hierarchicalProb: origHier * scaleFactor(c.countryCode),
      };
    })
    .sort((a, b) => b.hierarchicalProb - a.hierarchicalProb)
    .map((c, idx) => ({ ...c, rank: idx + 1 }));

  const confidence = buildConfidenceSummary(
    countries,
    regionsHierarchical,
    citiesHierarchical,
    raw.visualAxes,
    raw.geocell,
  );
  confidence.bimodalWarning = detectBimodalHotspots(raw.geocell);
  confidence.brief = `${confidence.brief} Updated with ${active.length} Bayesian evidence ${active.length === 1 ? 'entry' : 'entries'}.`;

  return {
    ...raw,
    countries,
    regionsHierarchical,
    citiesHierarchical,
    confidence,
  };
}
