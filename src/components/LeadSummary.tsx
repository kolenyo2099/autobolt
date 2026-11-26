import {
  ShieldCheck,
  AlertTriangle,
  HelpCircle,
  ExternalLink,
  Crosshair,
} from 'lucide-react';
import type { AnalysisResult } from '../engine/types';

interface LeadSummaryProps {
  result: AnalysisResult | null;
  analyzing: boolean;
}

function formatLatLng(lat: number, lng: number): string {
  const ns = lat >= 0 ? 'N' : 'S';
  const ew = lng >= 0 ? 'E' : 'W';
  return `${Math.abs(lat).toFixed(2)}° ${ns}, ${Math.abs(lng).toFixed(2)}° ${ew}`;
}

export function LeadSummary({ result, analyzing }: LeadSummaryProps) {
  if (!result) {
    return (
      <section className="panel-card hero-lead-card" aria-label="Primary Geolocation Lead">
        <div className="hero-lead-top">
          <span className="hero-eyebrow">Primary Lead</span>
        </div>
        <h2 className="hero-location" style={{ color: 'var(--text-muted)' }}>
          {analyzing ? 'Analyzing scene…' : 'Waiting for photo analysis'}
        </h2>
        <p className="hero-brief">
          Select a sample image or drop your own street photo to infer the country, region, city, and coordinates locally in your browser.
        </p>
      </section>
    );
  }

  const topCountry = result.countries[0];
  const topRegion = result.regionsHierarchical[0] ?? result.regionsGlobal[0];
  const topCity = result.citiesHierarchical[0] ?? result.citiesGlobal[0];
  const geocell = result.geocell;

  const tierClass =
    result.confidence.tier === 'High Signal'
      ? 'high'
      : result.confidence.tier === 'Moderate Lead'
        ? 'moderate'
        : 'ambiguous';

  const TierIcon =
    result.confidence.tier === 'High Signal'
      ? ShieldCheck
      : result.confidence.tier === 'Moderate Lead'
        ? AlertTriangle
        : HelpCircle;

  const locationTitle = topCity
    ? `${topCity.name}, ${topCity.region}`
    : topRegion
      ? topRegion.name
      : topCountry?.name ?? 'Unknown location';

  const linkLat = geocell ? geocell.lat : topCity?.lat;
  const linkLng = geocell ? geocell.lng : topCity?.lng;
  const coordLabel = geocell?.source === 'osv5m-11k' ? 'OSV-5M 11k Est.' : 'OSV-5M 5k Est.';

  return (
    <section className="panel-card hero-lead-card" aria-label="Primary Geolocation Lead">
      <div className="hero-lead-top">
        <span className="hero-eyebrow">Primary Lead</span>
        <span className={`tier-badge ${tierClass}`}>
          <TierIcon size={14} aria-hidden="true" />
          <span>{result.confidence.tier}</span>
        </span>
      </div>

      <h2 className="hero-location">{locationTitle}</h2>

      {topCountry && (
        <div className="hero-country-row">
          <span className="hero-country-main">
            {topCountry.name} · <span className="mono">{(topCountry.prob * 100).toFixed(1)}%</span>
          </span>
          <span className="hero-country-meta">
            Drives {topCountry.driving} · {topCountry.scripts[0]}
          </span>
        </div>
      )}

      {geocell && (
        <div
          className="hero-coords-line"
          title={`${coordLabel} continuous coordinate estimate across 225 countries/territories`}
        >
          <Crosshair size={13} aria-hidden="true" />
          <span className="mono">
            {coordLabel} {formatLatLng(geocell.lat, geocell.lng)}
          </span>
          <span className="hero-country-meta mono">
            · ±{Math.round(geocell.spreadKm)} km radius
          </span>
          {geocell.secondaryCoord && (
            <span className="hero-country-meta">
              ·{' '}
              {geocell.secondaryCoord.divergenceKm <= 100
                ? `Dual match (${Math.round(geocell.secondaryCoord.divergenceKm)} km)`
                : `${geocell.secondaryCoord.label}: ${Math.round(geocell.secondaryCoord.divergenceKm)} km away`}
            </span>
          )}
        </div>
      )}

      <p className="hero-brief">{result.confidence.brief}</p>

      {linkLat !== undefined && linkLng !== undefined && (
        <div className="hero-actions">
          <a
            className="hero-link"
            href={`https://www.openstreetmap.org/?mlat=${linkLat}&mlon=${linkLng}#map=12/${linkLat}/${linkLng}`}
            target="_blank"
            rel="noreferrer"
          >
            <span>OpenStreetMap</span>
            <ExternalLink size={12} aria-hidden="true" />
          </a>
          <a
            className="hero-link"
            href={`https://www.google.com/maps/search/?api=1&query=${linkLat},${linkLng}`}
            target="_blank"
            rel="noreferrer"
          >
            <span>Google Maps</span>
            <ExternalLink size={12} aria-hidden="true" />
          </a>
        </div>
      )}
    </section>
  );
}
