import { useRef, useState } from 'react';
import {
  ExternalLink,
  Loader2,
  Sparkles,
  SlidersHorizontal,
  Crosshair,
  Layers,
} from 'lucide-react';
import {
  ClueBoardPanel,
  OsmTerrainPanel,
  SunGeometryPanel,
} from './VerificationPanels';
import type {
  ActiveBackend,
  AnalysisResult,
  BackendMode,
  CustomProbeResult,
  EngineSource,
  EvidenceEntry,
  ExifMetadata,
  OsmPin,
  ScoredCity,
  ScoredCountry,
  SkylineProfile,
  SolarBandPoint,
} from '../engine/types';

interface LeadDisclosuresProps {
  result: AnalysisResult | null;
  exif: ExifMetadata | null;
  engineSource: EngineSource;
  backendMode: BackendMode;
  selectedCountryCode: string | null;
  selectedCity: ScoredCity | null;
  customProbeResult: CustomProbeResult | null;
  probeLoading: boolean;
  probeProgressMsg: string | null;
  probeProgressRatio: number | null;
  modelSizeLabel: string;
  osv5mSizeLabel: string;
  geocellsSizeLabel: string;
  backend: ActiveBackend;
  evidenceEntries: EvidenceEntry[];
  solarBandCount: number;
  osmPins: OsmPin[];
  skyline: SkylineProfile | null;
  onAddEvidence: (entry: EvidenceEntry) => void;
  onToggleEvidence: (id: string) => void;
  onRemoveEvidence: (id: string) => void;
  onClearEvidence: () => void;
  onPlotSolarBand: (points: SolarBandPoint[]) => void;
  onClearSolarBand: () => void;
  onSetOsmPins: (pins: OsmPin[]) => void;
  onSetSkyline: (profile: SkylineProfile | null) => void;
  onChangeEngine: (source: EngineSource, backend: BackendMode) => void;
  onSelectCountry: (code: string | null) => void;
  onSelectCity: (city: ScoredCity) => void;
  onFocusCoords: (lat: number, lng: number) => void;
  onRunCustomProbe: (labels: string[], useStreetTemplate: boolean) => void;
}

const PROBE_PRESETS = [
  {
    name: 'Compare Cities',
    labels: ['San Francisco', 'Seattle', 'Los Angeles', 'Vancouver', 'Lisbon', 'Nagasaki'],
  },
  {
    name: 'Mediterranean vs LatAm',
    labels: ['Spain', 'Portugal', 'Italy', 'Greece', 'Chile', 'Argentina', 'Mexico'],
  },
  {
    name: 'East Asian Cues',
    labels: ['Japan', 'South Korea', 'Taiwan', 'China', 'Thailand', 'Vietnam'],
  },
  {
    name: 'Road & Flora Check',
    labels: [
      'left-hand traffic',
      'right-hand traffic',
      'palm trees',
      'pine forest',
      'tram tracks',
      'cobblestone street',
    ],
  },
];

export function LeadDisclosures({
  result,
  exif,
  engineSource,
  backendMode,
  selectedCountryCode,
  selectedCity,
  customProbeResult,
  probeLoading,
  probeProgressMsg,
  probeProgressRatio,
  modelSizeLabel,
  osv5mSizeLabel,
  geocellsSizeLabel,
  backend,
  evidenceEntries,
  solarBandCount,
  osmPins,
  skyline,
  onAddEvidence,
  onToggleEvidence,
  onRemoveEvidence,
  onClearEvidence,
  onPlotSolarBand,
  onClearSolarBand,
  onSetOsmPins,
  onSetSkyline,
  onChangeEngine,
  onSelectCountry,
  onSelectCity,
  onFocusCoords,
  onRunCustomProbe,
}: LeadDisclosuresProps) {
  const stackRef = useRef<HTMLDivElement>(null);
  const [allExpanded, setAllExpanded] = useState(false);

  const [regionScope, setRegionScope] = useState<'hierarchical' | 'global'>('hierarchical');
  const [cityScope, setCityScope] = useState<'hierarchical' | 'global'>('hierarchical');
  const [showAllGeo, setShowAllGeo] = useState(false);

  const [probeInput, setProbeInput] = useState(
    'San Francisco\nSeattle\nLos Angeles\nNagasaki\nLisbon\nTokyo',
  );
  const [useStreetTemplate, setUseStreetTemplate] = useState(true);

  const syncAllExpanded = () => {
    const stack = stackRef.current;
    if (!stack) return;
    const items = stack.querySelectorAll<HTMLDetailsElement>(':scope > details');
    setAllExpanded(items.length > 0 && Array.from(items).every((d) => d.open));
  };

  const handleToggleAll = () => {
    const stack = stackRef.current;
    if (!stack) return;
    const nextOpen = !allExpanded;
    for (const d of stack.querySelectorAll<HTMLDetailsElement>(':scope > details')) {
      d.open = nextOpen;
    }
    setAllExpanded(nextOpen);
  };

  const geoLimit = showAllGeo ? 8 : 3;

  const displayedCountries: ScoredCountry[] = result
    ? result.countries.slice(0, geoLimit)
    : [];

  const displayedRegions = result
    ? (regionScope === 'hierarchical' ? result.regionsHierarchical : result.regionsGlobal)
        .filter((r) => !selectedCountryCode || r.countryCode === selectedCountryCode)
        .slice(0, geoLimit)
    : [];

  const displayedCities = result
    ? (cityScope === 'hierarchical' ? result.citiesHierarchical : result.citiesGlobal)
        .filter((c) => !selectedCountryCode || c.countryCode === selectedCountryCode)
        .slice(0, geoLimit)
    : [];

  const geoPreview = result
    ? `${result.countries[0]?.name ?? ''} (${((result.countries[0]?.prob ?? 0) * 100).toFixed(0)}%) · ${result.citiesHierarchical[0]?.name ?? ''}`
    : 'Awaiting analysis';

  const cluesPreview =
    result && !result.hasVisualClues
      ? 'Unavailable in OSV-5M standalone'
      : result
        ? result.visualAxes
            .slice(0, 3)
            .map((a) => a.topItem.label.split(' ').slice(0, 2).join(' '))
            .join(' · ')
        : '6 visual axes';

  const probePreview =
    result && !result.hasVisualClues
      ? 'Requires StreetCLIP or Dual Mode'
      : customProbeResult?.items[0]
        ? `#1 ${customProbeResult.items[0].label} (${(customProbeResult.items[0].prob * 100).toFixed(1)}%)`
        : 'Compare custom places or clues';

  const engineDisplay =
    engineSource === 'osv5m-q4'
      ? `OSV-5M Q4 (${osv5mSizeLabel})`
      : engineSource === 'dual-q4'
        ? 'Dual: StreetCLIP + OSV-5M'
        : `Vision Q4 (${modelSizeLabel}) + 5k Geocells`;

  const runtimePreview = result
    ? `${engineDisplay} · ${result.backend.toUpperCase()} · ${result.latencyMs.toFixed(0)} ms`
    : `${engineDisplay}${backend !== 'unknown' ? ` · ${backend.toUpperCase()}` : ''}`;

  const handleProbeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const lines = probeInput
      .split(/\n|,/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (lines.length === 0 || !result) return;
    onRunCustomProbe(lines, useStreetTemplate);
  };

  return (
    <section aria-labelledby="explore-details-heading">
      <div className="accordion-header">
        <h2 id="explore-details-heading" className="accordion-heading">
          Explore details
        </h2>
        <button
          type="button"
          className="btn btn-sm btn-ghost"
          {...({ command: '--toggle-all', commandfor: 'lead-disclosure-stack' } as Record<string, unknown>)}
          aria-expanded={allExpanded}
          onClick={handleToggleAll}
        >
          <span className="label-expand">Expand all</span>
          <span className="label-collapse">Collapse all</span>
        </button>
      </div>

      <div id="lead-disclosure-stack" ref={stackRef} className="disclosure-stack">
        {/* 1. Geographic breakdown & coordinate hotspots */}
        <details className="disclosure" onToggle={syncAllExpanded}>
          <summary>
            <span className="summary-title">Geographic breakdown &amp; alternatives</span>
            <span className="summary-preview">{geoPreview}</span>
          </summary>
          <div className="disclosure-body">
            {!result ? (
              <p style={{ color: 'var(--text-muted)', margin: '0.5rem 0 0', fontSize: '0.84rem' }}>
                Analyze a photo to inspect ranked countries, regions, candidate cities, and coordinate hotspots.
              </p>
            ) : (
              <>
                {result.geocell && result.geocell.hotspots.length > 0 && (
                  <div style={{ marginTop: '0.65rem' }}>
                    <div className="geo-col-head" style={{ marginBottom: '0.4rem' }}>
                      <h3 className="geo-col-title">
                        Coordinate Hotspots (
                        {result.geocell.source === 'osv5m-11k'
                          ? 'OSV-5M 11,399-cell + regression'
                          : 'OSV-5M 5,000-cell probe'}
                        )
                      </h3>
                      <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                        225 territories
                      </span>
                    </div>
                    <div className="hotspots-strip">
                      {result.geocell.hotspots.map((spot, idx) => (
                        <button
                          key={spot.cellId}
                          type="button"
                          className="hotspot-chip"
                          onClick={() => onFocusCoords(spot.lat, spot.lng)}
                          title="Focus map on this coordinate cluster"
                        >
                          <div className="hotspot-top">
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                              <Crosshair size={12} aria-hidden="true" />
                              <strong>#{idx + 1}</strong>
                            </span>
                            <span className="lead-pct mono">{(spot.prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="mono" style={{ fontSize: '0.74rem' }}>
                            {spot.lat.toFixed(2)}, {spot.lng.toFixed(2)}
                          </div>
                          {spot.nearestCity && (
                            <div className="lead-sub">
                              Near {spot.nearestCity}, {spot.nearestCountry}
                            </div>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '0.5rem',
                    marginTop: '0.75rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    {selectedCountryCode && (
                      <>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                          Filtered:{' '}
                          <strong>
                            {result.countries.find((c) => c.code === selectedCountryCode)?.name ??
                              selectedCountryCode}
                          </strong>
                        </span>
                        <button
                          type="button"
                          className="btn btn-sm btn-pill"
                          onClick={() => onSelectCountry(null)}
                        >
                          Reset
                        </button>
                      </>
                    )}
                  </div>

                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => setShowAllGeo((v) => !v)}
                  >
                    {showAllGeo ? 'Show top 3' : 'Show top 8'}
                  </button>
                </div>

                <div className="geo-columns">
                  {/* Countries */}
                  <div className="geo-col">
                    <div className="geo-col-head">
                      <h3 className="geo-col-title">Countries</h3>
                      <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                        {result.countries.length} total
                      </span>
                    </div>
                    {displayedCountries.map((country) => (
                      <button
                        key={country.code}
                        type="button"
                        className="lead-row"
                        aria-current={selectedCountryCode === country.code ? 'true' : undefined}
                        onClick={() =>
                          onSelectCountry(
                            selectedCountryCode === country.code ? null : country.code,
                          )
                        }
                      >
                        <div className="lead-row-main">
                          <span className="lead-name">
                            {country.rank}. {country.name}
                          </span>
                          <span className="lead-pct mono">{(country.prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${Math.max(2, country.prob * 100)}%` }}
                          />
                        </div>
                        <div className="lead-sub">
                          {country.continent} · Drives {country.driving} · {country.scripts[0]}
                        </div>
                      </button>
                    ))}
                  </div>

                  {/* Regions */}
                  <div className="geo-col">
                    <div className="geo-col-head">
                      <h3 className="geo-col-title">Regions / States</h3>
                      <button
                        type="button"
                        className="btn btn-sm"
                        style={{ padding: '2px 7px', fontSize: '0.7rem' }}
                        onClick={() =>
                          setRegionScope((s) => (s === 'hierarchical' ? 'global' : 'hierarchical'))
                        }
                      >
                        {regionScope === 'hierarchical' ? 'Top Countries' : 'Global'}
                      </button>
                    </div>
                    {displayedRegions.map((reg, idx) => {
                      const p = regionScope === 'hierarchical' ? reg.hierarchicalProb : reg.prob;
                      return (
                        <div key={`${reg.countryCode}-${reg.name}`} className="lead-row">
                          <div className="lead-row-main">
                            <span className="lead-name">
                              {idx + 1}. {reg.name}
                            </span>
                            <span className="lead-pct mono">{(p * 100).toFixed(1)}%</span>
                          </div>
                          <div className="bar-track">
                            <div
                              className="bar-fill"
                              style={{ width: `${Math.max(2, Math.min(100, p * 100))}%` }}
                            />
                          </div>
                          <div className="lead-sub">{reg.countryName}</div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Cities */}
                  <div className="geo-col">
                    <div className="geo-col-head">
                      <h3 className="geo-col-title">Cities / Towns</h3>
                      <button
                        type="button"
                        className="btn btn-sm"
                        style={{ padding: '2px 7px', fontSize: '0.7rem' }}
                        onClick={() =>
                          setCityScope((s) => (s === 'hierarchical' ? 'global' : 'hierarchical'))
                        }
                      >
                        {cityScope === 'hierarchical' ? 'Top Countries' : 'Global'}
                      </button>
                    </div>
                    {displayedCities.map((city, idx) => {
                      const p = cityScope === 'hierarchical' ? city.hierarchicalProb : city.prob;
                      const isSel =
                        selectedCity?.name === city.name &&
                        selectedCity?.countryCode === city.countryCode;
                      return (
                        <div
                          key={`${city.countryCode}-${city.name}`}
                          className="lead-row"
                          aria-current={isSel ? 'true' : undefined}
                        >
                          <button
                            type="button"
                            onClick={() => onSelectCity(city)}
                            style={{
                              border: 0,
                              background: 'transparent',
                              padding: 0,
                              textAlign: 'left',
                              width: '100%',
                            }}
                          >
                            <div className="lead-row-main">
                              <span className="lead-name">
                                {idx + 1}. {city.name}
                              </span>
                              <span className="lead-pct mono">{(p * 100).toFixed(1)}%</span>
                            </div>
                            <div className="bar-track" style={{ marginTop: 4 }}>
                              <div
                                className="bar-fill"
                                style={{ width: `${Math.max(2, Math.min(100, p * 100))}%` }}
                              />
                            </div>
                            <div className="lead-sub" style={{ marginTop: 3 }}>
                              {city.region}, {city.countryName}
                            </div>
                          </button>
                          <div className="lead-links">
                            <a
                              href={`https://www.openstreetmap.org/?mlat=${city.lat}&mlon=${city.lng}#map=12/${city.lat}/${city.lng}`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              OSM <ExternalLink size={10} aria-hidden="true" />
                            </a>
                            <a
                              href={`https://www.google.com/maps/search/?api=1&query=${city.lat},${city.lng}`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Maps <ExternalLink size={10} aria-hidden="true" />
                            </a>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="signal-footer">
                  <span>Signal diagnostics</span>
                  <span className="mono">
                    Top-1 {(result.confidence.top1Prob * 100).toFixed(1)}% · Margin +
                    {(result.confidence.top1Margin * 100).toFixed(1)}% · Entropy{' '}
                    {(result.confidence.entropyNormalized * 100).toFixed(0)}%
                  </span>
                </div>
              </>
            )}
          </div>
        </details>

        {/* 2. Visual & environmental clues */}
        <details className="disclosure" onToggle={syncAllExpanded}>
          <summary>
            <span className="summary-title">Visual &amp; environmental clues</span>
            <span className="summary-preview">{cluesPreview}</span>
          </summary>
          <div className="disclosure-body">
            {!result ? (
              <p style={{ color: 'var(--text-muted)', margin: '0.5rem 0 0', fontSize: '0.84rem' }}>
                Analyze a photo to extract environmental, architectural, road, and script leads.
              </p>
            ) : !result.hasVisualClues ? (
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '0.6rem', marginTop: '0.65rem' }}>
                <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  OSV-5M Baseline is a pure coordinate model without text-aligned visual clue axes.
                </p>
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={() => onChangeEngine('dual-q4', backendMode)}
                >
                  <Layers size={13} aria-hidden="true" />
                  <span>Switch to Dual Mode</span>
                </button>
              </div>
            ) : (
              <div className="clues-grid">
                {result.visualAxes.map((axis) => (
                  <div key={axis.id} className="clue-card">
                    <h3 className="clue-card-title">
                      <span>{axis.title}</span>
                      <span className="mono" style={{ fontSize: '0.74rem', color: 'var(--accent)' }}>
                        {(axis.topItem.prob * 100).toFixed(0)}%
                      </span>
                    </h3>
                    {axis.items.slice(0, 4).map((item) => (
                      <div key={item.id} className="clue-item">
                        <div className="clue-item-row">
                          <span>{item.label}</span>
                          <span className="mono">{(item.prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${Math.max(2, item.prob * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>
        </details>

        {/* 3. Test custom hypotheses (Text Q4) */}
        <details className="disclosure" onToggle={syncAllExpanded}>
          <summary>
            <span className="summary-title">Test custom hypotheses (Text Q4)</span>
            <span className="summary-preview">{probePreview}</span>
          </summary>
          <div className="disclosure-body">
            {result && !result.hasVisualClues ? (
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '0.6rem', marginTop: '0.65rem' }}>
                <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Custom text probing requires StreetCLIP embeddings alongside OSV-5M.
                </p>
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={() => onChangeEngine('dual-q4', backendMode)}
                >
                  <Layers size={13} aria-hidden="true" />
                  <span>Switch to Dual Mode</span>
                </button>
              </div>
            ) : (
              <form className="probe-form" onSubmit={handleProbeSubmit}>
                <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Compare any custom list of cities, landmarks, or visual clues against the active photo using the lazy-loaded quantized Text Q4 encoder.
                </p>

                <div className="preset-row">
                  {PROBE_PRESETS.map((p) => (
                    <button
                      key={p.name}
                      type="button"
                      className="btn btn-sm btn-pill"
                      onClick={() => setProbeInput(p.labels.join('\n'))}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>

                <label htmlFor="custom-probe-input" className="sr-only">
                  Candidate location or clue labels, one per line
                </label>
                <textarea
                  id="custom-probe-input"
                  className="probe-textarea"
                  value={probeInput}
                  onChange={(e) => setProbeInput(e.target.value)}
                  placeholder="Enter candidate places or hypotheses (one per line or comma-separated)..."
                />

                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '0.6rem',
                  }}
                >
                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.8rem' }}>
                    <input
                      type="checkbox"
                      checked={useStreetTemplate}
                      onChange={(e) => setUseStreetTemplate(e.target.checked)}
                    />
                    <span>Wrap with StreetCLIP template (&quot;A Street View photo in &#123;label&#125;.&quot;)</span>
                  </label>

                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={!result || !result.hasVisualClues || probeLoading}
                  >
                    {probeLoading ? <Loader2 size={14} aria-hidden="true" /> : <Sparkles size={14} aria-hidden="true" />}
                    <span>{probeLoading ? 'Scoring…' : 'Run Custom Probe'}</span>
                  </button>
                </div>
              </form>
            )}

            {probeLoading && probeProgressMsg && (
              <div style={{ marginTop: '0.85rem' }}>
                <div style={{ fontSize: '0.8rem', marginBottom: '0.3rem' }}>{probeProgressMsg}</div>
                <progress
                  className="scl-progress"
                  aria-label="Downloading quantized text encoder"
                  max={100}
                  value={probeProgressRatio !== null ? Math.round(probeProgressRatio * 100) : undefined}
                />
              </div>
            )}

            {customProbeResult && (
              <div style={{ marginTop: '0.95rem', display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  Ranked in <span className="mono">{customProbeResult.latencyMs.toFixed(0)} ms</span> on{' '}
                  <span className="mono">{customProbeResult.backend.toUpperCase()}</span>
                </div>
                {customProbeResult.items.map((item) => (
                  <div key={`${item.rank}-${item.label}`} className="lead-row">
                    <div className="lead-row-main">
                      <span className="lead-name">
                        {item.rank}. {item.label}
                      </span>
                      <span className="lead-pct mono">{(item.prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${Math.max(2, item.prob * 100)}%` }}
                      />
                    </div>
                    <div className="lead-sub mono">{item.prompt}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </details>

        {/* 4. Clue lookup & Bayesian evidence board */}
        <details className="disclosure" onToggle={syncAllExpanded}>
          <summary>
            <span className="summary-title">Clue lookup &amp; evidence board</span>
            <span className="summary-preview">
              {evidenceEntries.filter((e) => e.active).length > 0
                ? `${evidenceEntries.filter((e) => e.active).length} active clue${evidenceEntries.filter((e) => e.active).length > 1 ? 's' : ''}`
                : 'Calling codes, driving side, plates'}
            </span>
          </summary>
          <div className="disclosure-body">
            <ClueBoardPanel
              result={result}
              evidenceEntries={evidenceEntries}
              onAddEvidence={onAddEvidence}
              onToggleEvidence={onToggleEvidence}
              onRemoveEvidence={onRemoveEvidence}
              onClearEvidence={onClearEvidence}
            />
          </div>
        </details>

        {/* 5. Sun, shadow & chronolocation */}
        <details className="disclosure" onToggle={syncAllExpanded}>
          <summary>
            <span className="summary-title">Sun, shadow &amp; chronolocation</span>
            <span className="summary-preview">
              {solarBandCount > 0 ? `${solarBandCount} band pts plotted` : 'NOAA solar angle & time check'}
            </span>
          </summary>
          <div className="disclosure-body">
            <SunGeometryPanel
              result={result}
              exif={exif}
              solarBandCount={solarBandCount}
              onPlotSolarBand={onPlotSolarBand}
              onClearSolarBand={onClearSolarBand}
            />
          </div>
        </details>

        {/* 6. Verify with OSM & 3D terrain skyline */}
        <details className="disclosure" onToggle={syncAllExpanded}>
          <summary>
            <span className="summary-title">Verify with OSM &amp; 3D terrain skyline</span>
            <span className="summary-preview">
              {skyline
                ? `Skyline ${skyline.headingDeg}°`
                : osmPins.length > 0
                  ? `${osmPins.length} OSM pins`
                  : 'Nominatim, Overpass & DEM horizon'}
            </span>
          </summary>
          <div className="disclosure-body">
            <OsmTerrainPanel
              result={result}
              exif={exif}
              osmPins={osmPins}
              skyline={skyline}
              onSetOsmPins={onSetOsmPins}
              onFocusCoords={onFocusCoords}
              onSetSkyline={onSetSkyline}
            />
          </div>
        </details>

        {/* 7. Runtime & quantization details */}
        <details className="disclosure" onToggle={syncAllExpanded}>
          <summary>
            <span className="summary-title">Runtime &amp; quantization details</span>
            <span className="summary-preview mono">{runtimePreview}</span>
          </summary>
          <div className="disclosure-body">
            <div className="field-group" style={{ marginTop: '0.65rem' }}>
              <label htmlFor="inline-engine-select" style={{ fontWeight: 600 }}>
                Geolocation Engine
              </label>
              <select
                id="inline-engine-select"
                className="field-select"
                value={engineSource}
                onChange={(e) => onChangeEngine(e.target.value as EngineSource, backendMode)}
              >
                <option value="local-q4">
                  StreetCLIP Vision Q4 ({modelSizeLabel}) + 5k Geocells ({geocellsSizeLabel})
                </option>
                <option value="osv5m-q4">
                  OSV-5M Baseline Q4 ({osv5mSizeLabel}) — 224px · 11,399 cells + regression
                </option>
                <option value="dual-q4">
                  Dual Mode: OSV-5M 11k Coordinates + StreetCLIP Visual Clues
                </option>
                <option value="hf-q4f16">
                  Remote HF StreetCLIP-ONNX (q4f16 · 298 MB)
                </option>
              </select>
            </div>

            <div className="exif-grid" style={{ marginTop: '0.65rem' }}>
              <div className="exif-kv">
                <div className="exif-k">Active Models</div>
                <div className="exif-v mono">{engineDisplay}</div>
              </div>
              <div className="exif-kv">
                <div className="exif-k">Execution Backend</div>
                <div className="exif-v mono">
                  {result ? result.backend.toUpperCase() : backend.toUpperCase()}
                </div>
              </div>
              <div className="exif-kv">
                <div className="exif-k">Last Inference</div>
                <div className="exif-v mono">
                  {result ? `${result.latencyMs.toFixed(0)} ms (${result.cropCount} crop${result.cropCount > 1 ? 's' : ''})` : '—'}
                </div>
              </div>
            </div>

            <div style={{ marginTop: '0.75rem' }}>
              <button
                type="button"
                className="btn btn-sm"
                {...({ command: 'show-modal', commandfor: 'model-dialog' } as Record<string, unknown>)}
              >
                <SlidersHorizontal size={14} aria-hidden="true" />
                <span>Open Model &amp; Quantization Inspector…</span>
              </button>
            </div>
          </div>
        </details>
      </div>
    </section>
  );
}
