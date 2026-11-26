import { useEffect, useState } from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  HelpCircle,
  ExternalLink,
  Crosshair,
  Loader2,
  ScanText,
  Square,
  Plus,
} from 'lucide-react';
import { extractAutoCluesFromVlmText } from '../engine/clues';
import { VLM_MODEL_SPECS } from '../engine/vlmClient';
import type {
  AnalysisResult,
  EvidenceEntry,
  VlmModelId,
  VlmProgress,
  VlmResult,
} from '../engine/types';

interface LeadSummaryProps {
  result: AnalysisResult | null;
  analyzing: boolean;
  hasImage: boolean;
  vlmModelId: VlmModelId;
  vlmGrounded: boolean;
  vlmLoading: boolean;
  vlmProgress: VlmProgress | null;
  vlmStreamText: string;
  vlmResult: VlmResult | null;
  vlmError: string | null;
  evidenceEntries: EvidenceEntry[];
  onChangeVlmModel: (id: VlmModelId) => void;
  onChangeVlmGrounded: (grounded: boolean) => void;
  onRunVlmCheck: (modelId: VlmModelId, grounded: boolean) => void;
  onStopVlm: () => void;
  onApplyEvidence: (entry: EvidenceEntry) => void;
}

function formatLatLng(lat: number, lng: number): string {
  const ns = lat >= 0 ? 'N' : 'S';
  const ew = lng >= 0 ? 'E' : 'W';
  return `${Math.abs(lat).toFixed(2)}° ${ns}, ${Math.abs(lng).toFixed(2)}° ${ew}`;
}

export function LeadSummary({
  result,
  analyzing,
  hasImage,
  vlmModelId,
  vlmGrounded,
  vlmLoading,
  vlmProgress,
  vlmStreamText,
  vlmResult,
  vlmError,
  evidenceEntries,
  onChangeVlmModel,
  onChangeVlmGrounded,
  onRunVlmCheck,
  onStopVlm,
  onApplyEvidence,
}: LeadSummaryProps) {
  const [suggestedClues, setSuggestedClues] = useState<EvidenceEntry[]>([]);

  useEffect(() => {
    let cancelled = false;
    if (!vlmResult?.text || !result) {
      setSuggestedClues([]);
      return;
    }
    extractAutoCluesFromVlmText(vlmResult.text, result.countries)
      .then((items) => {
        if (!cancelled) setSuggestedClues(items);
      })
      .catch(() => {
        if (!cancelled) setSuggestedClues([]);
      });
    return () => {
      cancelled = true;
    };
  }, [vlmResult, result]);
  const selectedVlmSpec =
    VLM_MODEL_SPECS.find((m) => m.id === vlmModelId) ?? VLM_MODEL_SPECS[0];
  const vlmPreview = vlmLoading
    ? vlmProgress?.stage === 'generating'
      ? 'Streaming clues…'
      : vlmProgress?.percent !== undefined
        ? `Downloading ${vlmProgress.percent}%…`
        : 'Loading Qwen-VL…'
    : vlmResult
      ? `${vlmResult.modelLabel} · ${(vlmResult.latencyMs / 1000).toFixed(1)}s`
      : `On-demand OCR & sign reasoning (${selectedVlmSpec.approxSizeLabel})`;
  const vlmActiveText = vlmResult?.text ?? vlmStreamText;

  const vlmDisclosure = hasImage ? (
    <details className="disclosure embedded hero-vlm-disclosure">
      <summary>
        <span className="summary-title">Deep VLM clue check (Qwen-VL Q4)</span>
        <span className="summary-preview">{vlmPreview}</span>
      </summary>
      <div className="disclosure-body">
        <div className="vlm-controls">
          <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Runs an on-demand 4-bit quantized Qwen vision-language model locally on WebGPU (cached in your browser after first download) to read street signs, storefront text, and road clues.
          </p>

          <div className="vlm-row">
            <div className="field-group" style={{ flex: '1 1 240px' }}>
              <label htmlFor="vlm-model-select" style={{ fontWeight: 600 }}>
                Quantized Qwen-VL Model
              </label>
              <select
                id="vlm-model-select"
                className="field-select"
                value={vlmModelId}
                disabled={vlmLoading}
                onChange={(e) => onChangeVlmModel(e.target.value as VlmModelId)}
              >
                {VLM_MODEL_SPECS.map((spec) => (
                  <option key={spec.id} value={spec.id}>
                    {spec.label} ({spec.approxSizeLabel}) — {spec.notes}
                  </option>
                ))}
              </select>
            </div>

            <div className="vlm-actions">
              {vlmLoading ? (
                <button type="button" className="btn btn-sm" onClick={onStopVlm}>
                  <Square size={13} aria-hidden="true" />
                  <span>Stop</span>
                </button>
              ) : null}
              <button
                type="button"
                className="btn btn-primary"
                disabled={!hasImage || vlmLoading}
                onClick={() => onRunVlmCheck(vlmModelId, vlmGrounded)}
              >
                {vlmLoading ? (
                  <Loader2 size={14} aria-hidden="true" />
                ) : (
                  <ScanText size={14} aria-hidden="true" />
                )}
                <span>{vlmLoading ? 'Running…' : 'Run VLM Clue Check'}</span>
              </button>
            </div>
          </div>

          <label
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.45rem',
              fontSize: '0.8rem',
            }}
          >
            <input
              type="checkbox"
              checked={vlmGrounded}
              disabled={vlmLoading}
              onChange={(e) => onChangeVlmGrounded(e.target.checked)}
            />
            <span>Ground prompt with AutoBolt&apos;s top country/city/coordinate leads</span>
          </label>

          {vlmLoading && vlmProgress && (
            <div>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: '0.5rem',
                  fontSize: '0.78rem',
                  color: 'var(--text-secondary)',
                  marginBottom: '0.3rem',
                }}
              >
                <span>{vlmProgress.message}</span>
                {vlmProgress.totalBytes > 1 && (
                  <span className="mono">
                    {(vlmProgress.loadedBytes / (1024 * 1024)).toFixed(0)} /{' '}
                    {(vlmProgress.totalBytes / (1024 * 1024)).toFixed(0)} MB
                    {vlmProgress.percent !== undefined ? ` (${vlmProgress.percent}%)` : ''}
                  </span>
                )}
              </div>
              <progress
                className="scl-progress"
                aria-label="Loading quantized Qwen-VL model"
                max={100}
                value={vlmProgress.percent}
              />
            </div>
          )}

          {vlmError && (
            <div className="vlm-error" role="alert">
              <strong>VLM error:</strong> {vlmError}
            </div>
          )}

          {vlmActiveText && (
            <div className="vlm-output-card">
              <div className="vlm-meta-row mono">
                <span>{vlmResult ? vlmResult.modelLabel : selectedVlmSpec.label}</span>
                {vlmResult ? (
                  <span>
                    {vlmResult.backend.toUpperCase()} · {vlmResult.dtype.toUpperCase()} ·{' '}
                    {(vlmResult.latencyMs / 1000).toFixed(1)}s ·{' '}
                    {vlmResult.tokensPerSec.toFixed(1)} tok/s
                  </span>
                ) : (
                  <span>Streaming…</span>
                )}
              </div>
              <div className="vlm-output-text">{vlmActiveText}</div>
              {vlmResult && suggestedClues.length > 0 && (
                <div className="preset-row" style={{ marginTop: '0.25rem' }}>
                  {suggestedClues.map((clue) => {
                    const already = evidenceEntries.some((e) => e.id === clue.id);
                    return (
                      <button
                        key={clue.id}
                        type="button"
                        className="btn btn-sm btn-pill"
                        disabled={already}
                        onClick={() => onApplyEvidence(clue)}
                        title={clue.detail}
                      >
                        <Plus size={12} aria-hidden="true" />
                        <span>{already ? 'Applied: ' : 'Apply: '}{clue.label}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </details>
  ) : null;

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
        {vlmDisclosure}
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
        <div className="tier-group">
          <div className="tier-levels mono" title="Independent geo-sleuth per-level confidence tiers">
            <span className={`level-chip level-${result.confidence.levels.country.toLowerCase()}`}>
              Country: {result.confidence.levels.country}
            </span>
            <span className={`level-chip level-${result.confidence.levels.region.toLowerCase()}`}>
              Region: {result.confidence.levels.region}
            </span>
            <span className={`level-chip level-${result.confidence.levels.city.toLowerCase()}`}>
              City: {result.confidence.levels.city}
            </span>
          </div>
          <span className={`tier-badge ${tierClass}`}>
            <TierIcon size={14} aria-hidden="true" />
            <span>{result.confidence.tier}</span>
          </span>
        </div>
      </div>

      {result.confidence.bimodalWarning && (
        <div className="bimodal-warning">
          <AlertTriangle size={13} aria-hidden="true" />
          <span>
            Competing hotspots ({result.confidence.bimodalWarning.separationKm} km apart):{' '}
            <strong>{result.confidence.bimodalWarning.hotspotA}</strong> vs{' '}
            <strong>{result.confidence.bimodalWarning.hotspotB}</strong> — check distinguishing clues rather than averaging.
          </span>
        </div>
      )}

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

      {vlmDisclosure}
    </section>
  );
}
