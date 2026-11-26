import type { AnalysisResult, ExifMetadata } from '../engine/types';
import { haversineDistanceKm } from '../engine/exif';

export function buildMarkdownBrief(
  result: AnalysisResult,
  imageName: string,
  exif: ExifMetadata | null,
  gpsRevealed: boolean,
): string {
  const topCountries = result.countries.slice(0, 5);
  const topRegions = result.regionsHierarchical.slice(0, 5);
  const topCities = result.citiesHierarchical.slice(0, 5);

  const lines: string[] = [
    `# Investigations Lab's AutoBolt — Geolocation Lead Brief`,
    ``,
    `- **Image:** ${imageName}`,
    `- **Analysis Mode:** ${result.mode} (${result.cropCount} crop${result.cropCount > 1 ? 's' : ''})`,
    `- **Runtime:** ${result.backend.toUpperCase()} (${result.latencyMs.toFixed(0)} ms, 100% in-browser)`,
    `- **Signal Tier:** ${result.confidence.tier} (Top-1 ${(result.confidence.top1Prob * 100).toFixed(1)}%, Margin ${(result.confidence.top1Margin * 100).toFixed(1)}%)`,
    ``,
    `## Investigative Summary`,
    result.confidence.brief,
    ``,
    `## Top Country Leads`,
    ...topCountries.map(
      (c) => `${c.rank}. **${c.name}** (${c.code}) — ${(c.prob * 100).toFixed(1)}% · Drives ${c.driving} · ${c.scripts.join(', ')}`,
    ),
    ...(result.geocell
      ? [
          ``,
          `## Continuous Coordinate Estimate (OSV-5M 5,000-Cell Probe)`,
          `- **Estimated Position:** \`${result.geocell.lat.toFixed(4)}, ${result.geocell.lng.toFixed(4)}\` (±${Math.round(result.geocell.spreadKm)} km spread)`,
          ...result.geocell.hotspots.map(
            (h, i) =>
              `${i + 1}. \`${h.lat.toFixed(3)}, ${h.lng.toFixed(3)}\` (${(h.prob * 100).toFixed(1)}%)${h.nearestCity ? ` — near ${h.nearestCity}, ${h.nearestCountry}` : ''}`,
          ),
        ]
      : []),
    ``,
    `## Top Regional Leads (Hierarchical)`,
    ...topRegions.map(
      (r, idx) => `${idx + 1}. **${r.name}**, ${r.countryName} — ${(r.hierarchicalProb * 100).toFixed(1)}%`,
    ),
    ``,
    `## Top City / Town Leads (Hierarchical)`,
    ...topCities.map(
      (c, idx) =>
        `${idx + 1}. **${c.name}** (${c.region}, ${c.countryName}) — ${(c.hierarchicalProb * 100).toFixed(1)}% · \`${c.lat.toFixed(4)}, ${c.lng.toFixed(4)}\``,
    ),
    ``,
    `## Visual & Environmental Leads`,
    ...result.visualAxes.map(
      (axis) => `- **${axis.title}:** ${axis.topItem.label} (${(axis.topItem.prob * 100).toFixed(1)}%)`,
    ),
  ];

  if (gpsRevealed && exif?.gps && topCities[0]) {
    const errKm = haversineDistanceKm(
      exif.gps.lat,
      exif.gps.lng,
      topCities[0].lat,
      topCities[0].lng,
    );
    lines.push(
      ``,
      `## EXIF Ground-Truth Check`,
      `- **Actual Coordinates:** ${exif.gps.lat.toFixed(5)}, ${exif.gps.lng.toFixed(5)}`,
      `- **Distance to Top-1 City (${topCities[0].name}):** ${errKm.toFixed(1)} km`,
    );
  }

  return lines.join('\n');
}

export function downloadJsonBrief(result: AnalysisResult, imageName: string): void {
  const payload = {
    image: imageName,
    generatedAt: new Date().toISOString(),
    runtime: {
      backend: result.backend,
      latencyMs: result.latencyMs,
      mode: result.mode,
      cropCount: result.cropCount,
    },
    confidence: result.confidence,
    geocell: result.geocell
      ? {
          lat: result.geocell.lat,
          lng: result.geocell.lng,
          spreadKm: result.geocell.spreadKm,
          hotspots: result.geocell.hotspots,
        }
      : undefined,
    topCountries: result.countries.slice(0, 10),
    topRegionsHierarchical: result.regionsHierarchical.slice(0, 10),
    topCitiesHierarchical: result.citiesHierarchical.slice(0, 10),
    visualAxes: result.visualAxes.map((a) => ({
      id: a.id,
      title: a.title,
      topLeads: a.items.slice(0, 4),
    })),
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `autobolt-leads-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
