import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Crosshair,
  ExternalLink,
  Layers,
  Loader2,
  SlidersHorizontal,
  Sparkles,
} from 'lucide-react';
import { ClueBoardPanel } from './VerificationPanels';
import type {
  ActiveBackend,
  AnalysisResult,
  BackendMode,
  CustomProbeResult,
  EngineSource,
  EvidenceEntry,
  ScoredCity,
  ScoredCountry,
} from '../engine/types';

interface LeadDisclosuresProps {
  result: AnalysisResult | null;
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
  onAddEvidence: (entry: EvidenceEntry) => void;
  onToggleEvidence: (id: string) => void;
  onRemoveEvidence: (id: string) => void;
  onClearEvidence: () => void;
  onChangeEngine: (source: EngineSource, backend: BackendMode) => void;
  onSelectCountry: (code: string | null) => void;
  onSelectCity: (city: ScoredCity) => void;
  onFocusCoords: (lat: number, lng: number) => void;
  onRunCustomProbe: (labels: string[], useStreetTemplate: boolean) => void;
}

const SLIDE_META = [
  { id: 'geo', label: 'Geographic', title: 'Geographic breakdown & hotspots' },
  { id: 'clues', label: 'Visual Clues', title: 'Visual & environmental clues' },
  { id: 'board', label: 'Clue Board', title: 'Clue lookup & evidence board' },
  { id: 'probe', label: 'Custom Probe', title: 'Test custom hypotheses (Text Q4)' },
  { id: 'runtime', label: 'Runtime', title: 'Runtime & quantization details' },
] as const;

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
  onAddEvidence,
  onToggleEvidence,
  onRemoveEvidence,
  onClearEvidence,
  onChangeEngine,
  onSelectCountry,
  onSelectCity,
  onFocusCoords,
  onRunCustomProbe,
}: LeadDisclosuresProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [announcement, setAnnouncement] = useState('');

  const settledIndexRef = useRef(0);
  const currentParkRef = useRef(0);
  const pendingTargetRef = useRef<number | null>(null);
  const pendingStepRef = useRef<-1 | 1 | null>(null);
  const expectedTargetPosRef = useRef<number | null>(null);
  const hasInteractedRef = useRef(false);

  const [regionScope, setRegionScope] = useState<'hierarchical' | 'global'>('hierarchical');
  const [cityScope, setCityScope] = useState<'hierarchical' | 'global'>('hierarchical');
  const [showAllGeo, setShowAllGeo] = useState(false);

  const [probeInput, setProbeInput] = useState(
    'San Francisco\nSeattle\nLos Angeles\nNagasaki\nLisbon\nTokyo',
  );
  const [useStreetTemplate, setUseStreetTemplate] = useState(true);

  const getSlides = () => {
    const track = trackRef.current;
    if (!track) return [] as HTMLElement[];
    return Array.from(track.children) as HTMLElement[];
  };

  const getNearestSlideIndex = useCallback((): number => {
    const track = trackRef.current;
    if (!track) return 0;
    const slides = getSlides();
    if (slides.length === 0) return 0;
    const center = track.scrollLeft + track.clientWidth / 2;
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < slides.length; i++) {
      const slideCenter = slides[i].offsetLeft + slides[i].offsetWidth / 2;
      const d = Math.abs(center - slideCenter);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    return bestIdx;
  }, []);

  const syncParking = useCallback((idx: number, lockOverflow: boolean) => {
    const track = trackRef.current;
    if (!track || track.clientWidth === 0) return;
    const slides = getSlides();
    const targetSlide = slides[idx];
    if (!targetSlide) return;

    const prevSnap = track.style.scrollSnapType;
    track.style.scrollSnapType = 'none';
    if (lockOverflow) {
      track.style.overflowX = 'hidden';
    }

    if (idx < 0) {
      track.style.removeProperty('--current');
    } else {
      track.style.setProperty('--current', String(idx));
    }
    currentParkRef.current = idx;

    const afterLeft = targetSlide.offsetLeft;
    if (Math.abs(track.scrollLeft - afterLeft) > 1) {
      track.scrollTo({ left: afterLeft, behavior: 'instant' });
      void track.scrollLeft;
    }

    track.style.scrollSnapType = prevSnap;
    if (lockOverflow) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (trackRef.current) {
            trackRef.current.style.overflowX = '';
          }
        });
      });
    }
  }, []);

  const scrollToSlide = useCallback((targetIdx: number, resetOrderForMarker = false) => {
    const track = trackRef.current;
    if (!track) return;
    hasInteractedRef.current = true;

    const launchScroll = () => {
      const tr = trackRef.current;
      if (!tr) return;
      tr.style.overflowX = '';
      const slide = Array.from(tr.children)[targetIdx] as HTMLElement | undefined;
      if (!slide) return;
      pendingTargetRef.current = targetIdx;
      expectedTargetPosRef.current = slide.offsetLeft;
      setActiveIndex(targetIdx);

      const delta = slide.offsetLeft - tr.scrollLeft;
      if (Math.abs(delta) < 2) {
        pendingTargetRef.current = null;
        expectedTargetPosRef.current = null;
        syncParking(targetIdx, true);
        settledIndexRef.current = targetIdx;
        return;
      }
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      tr.scrollTo({ left: slide.offsetLeft, behavior: reduced ? 'auto' : 'smooth' });
    };

    if (resetOrderForMarker && currentParkRef.current !== -1) {
      const anchorIdx = getNearestSlideIndex();
      const anchorSlide = getSlides()[anchorIdx];
      track.style.overflowX = '';
      track.style.scrollSnapType = 'none';
      track.style.removeProperty('--current');
      currentParkRef.current = -1;
      const afterLeft = anchorSlide?.offsetLeft ?? 0;
      track.scrollTo({ left: afterLeft, behavior: 'instant' });
      void track.scrollLeft;
      setActiveIndex(targetIdx);
      pendingTargetRef.current = targetIdx;
      requestAnimationFrame(() => {
        if (trackRef.current) {
          trackRef.current.style.scrollSnapType = '';
        }
        launchScroll();
      });
      return;
    }

    launchScroll();
  }, [getNearestSlideIndex, syncParking]);

  const handleStep = (dir: -1 | 1) => {
    const track = trackRef.current;
    if (!track) return;
    if (currentParkRef.current === -1) {
      // Mid-marker trip: replay step once settled
      pendingStepRef.current = dir;
      return;
    }
    const slides = getSlides();
    const n = slides.length;
    if (n === 0) return;

    const baseIdx = pendingTargetRef.current ?? getNearestSlideIndex();
    const targetIdx = (baseIdx + dir + n) % n;
    scrollToSlide(targetIdx, false);
  };

  useLayoutEffect(() => {
    const track = trackRef.current;
    if (!track) return;

    track.style.setProperty('--children-count', String(track.children.length));

    // Park initial Slide 1 (index 0) before browser first paint
    pendingTargetRef.current = 0;
    syncParking(0, false);
    settledIndexRef.current = 0;
    setActiveIndex(0);
    const id = requestAnimationFrame(() => {
      pendingTargetRef.current = null;
      hasInteractedRef.current = false;
    });
    return () => cancelAnimationFrame(id);
  }, [syncParking]);

  useEffect(() => {
    const track = trackRef.current;
    if (!track) return;

    let rafId = 0;
    let debounceTimer: number | null = null;
    const supportsScrollEnd = 'onscrollend' in window;

    const handleSettle = () => {
      const idx = getNearestSlideIndex();

      // Ignore scrollend events only when scroll is still in flight toward another target
      if (
        pendingTargetRef.current !== null &&
        idx !== pendingTargetRef.current &&
        expectedTargetPosRef.current !== null &&
        Math.abs(track.scrollLeft - expectedTargetPosRef.current) > track.clientWidth * 0.6
      ) {
        // If scroll has actually come to rest on a snapped slide, accept idx cleanly instead of stalling
        pendingTargetRef.current = null;
        expectedTargetPosRef.current = null;
      } else {
        pendingTargetRef.current = null;
        expectedTargetPosRef.current = null;
      }

      if (currentParkRef.current !== idx) {
        syncParking(idx, pendingStepRef.current === null);
      }

      setActiveIndex(idx);
      if (hasInteractedRef.current && idx !== settledIndexRef.current) {
        settledIndexRef.current = idx;
        setAnnouncement(`Slide ${idx + 1} of ${SLIDE_META.length}: ${SLIDE_META[idx].title}`);
      } else {
        settledIndexRef.current = idx;
      }

      if (pendingStepRef.current !== null) {
        const queuedDir = pendingStepRef.current;
        pendingStepRef.current = null;
        const n = SLIDE_META.length;
        scrollToSlide((idx + queuedDir + n) % n, false);
      }
    };

    const onScroll = () => {
      hasInteractedRef.current = true;
      if (pendingTargetRef.current === null) {
        if (!rafId) {
          rafId = requestAnimationFrame(() => {
            rafId = 0;
            const nearest = getNearestSlideIndex();
            setActiveIndex((prev) => (prev === nearest ? prev : nearest));
          });
        }
      }
      if (!supportsScrollEnd) {
        if (debounceTimer !== null) window.clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(handleSettle, 120);
      }
    };

    const onScrollEnd = () => handleSettle();

    track.addEventListener('scroll', onScroll, { passive: true });
    if (supportsScrollEnd) {
      track.addEventListener('scrollend', onScrollEnd);
    }

    return () => {
      track.removeEventListener('scroll', onScroll);
      if (supportsScrollEnd) {
        track.removeEventListener('scrollend', onScrollEnd);
      }
      if (rafId) cancelAnimationFrame(rafId);
      if (debounceTimer !== null) window.clearTimeout(debounceTimer);
    };
  }, [getNearestSlideIndex, scrollToSlide, syncParking]);

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

  const engineDisplay =
    engineSource === 'osv5m-q4'
      ? `OSV-5M Q4 (${osv5mSizeLabel})`
      : engineSource === 'dual-q4'
        ? 'Dual: StreetCLIP + OSV-5M'
        : `Vision Q4 (${modelSizeLabel}) + 5k Geocells`;

  const handleProbeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const lines = probeInput
      .split(/\n|,/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (lines.length === 0 || !result) return;
    onRunCustomProbe(lines, useStreetTemplate);
  };

  const activeSlide = SLIDE_META[activeIndex] ?? SLIDE_META[0];

  return (
    <section
      className="panel-card details-carousel"
      aria-roledescription="carousel"
      aria-labelledby="explore-details-heading"
    >
      <div className="carousel-header">
        <div className="carousel-title-group">
          <h2 id="explore-details-heading" className="accordion-heading">
            Explore details
          </h2>
          <span className="carousel-slide-subtitle">
            {activeSlide.title}
          </span>
        </div>

        {/* Previous/Next controls placed before the track in DOM order */}
        <div className="carousel-controls">
          <span className="carousel-counter mono" aria-hidden="true">
            {activeIndex + 1} / {SLIDE_META.length}
          </span>
          <button
            type="button"
            className="btn btn-sm carousel-nav-btn"
            aria-label="Previous slide"
            onClick={() => handleStep(-1)}
          >
            <ChevronLeft size={16} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="btn btn-sm carousel-nav-btn"
            aria-label="Next slide"
            onClick={() => handleStep(1)}
          >
            <ChevronRight size={16} aria-hidden="true" />
          </button>
        </div>
      </div>

      <div
        ref={trackRef}
        className="carousel-track"
        tabIndex={0}
        role="group"
        aria-label="Detail panels"
      >
        {/* Slide 1: Geographic breakdown & hotspots */}
        <div
          className="carousel-slide"
          role="group"
          aria-roledescription="slide"
          aria-label={`1 of ${SLIDE_META.length}`}
        >
          {!result ? (
            <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '0.84rem' }}>
              Analyze a photo to inspect ranked countries, regions, candidate cities, and coordinate hotspots.
            </p>
          ) : (
            <>
              {result.geocell && result.geocell.hotspots.length > 0 && (
                <div>
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

        {/* Slide 2: Visual & environmental clues */}
        <div
          className="carousel-slide"
          role="group"
          aria-roledescription="slide"
          aria-label={`2 of ${SLIDE_META.length}`}
        >
          {!result ? (
            <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '0.84rem' }}>
              Analyze a photo to extract environmental, architectural, road, and script leads.
            </p>
          ) : !result.hasVisualClues ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '0.6rem' }}>
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
                  <div className="clue-card-title">{axis.title}</div>
                  {axis.items.slice(0, 3).map((item, idx) => (
                    <div key={item.id} className="clue-item">
                      <div className="clue-item-head">
                        <span className={idx === 0 ? 'clue-item-top' : 'clue-item-label'}>
                          {item.label}
                        </span>
                        <span className="mono" style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                          {(item.prob * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="bar-track">
                        <div
                          className="bar-fill"
                          style={{
                            width: `${Math.max(3, item.prob * 100)}%`,
                            opacity: idx === 0 ? 1 : 0.45,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Slide 3: Clue lookup & Bayesian evidence board */}
        <div
          className="carousel-slide"
          role="group"
          aria-roledescription="slide"
          aria-label={`3 of ${SLIDE_META.length}`}
        >
          <ClueBoardPanel
            result={result}
            evidenceEntries={evidenceEntries}
            onAddEvidence={onAddEvidence}
            onToggleEvidence={onToggleEvidence}
            onRemoveEvidence={onRemoveEvidence}
            onClearEvidence={onClearEvidence}
          />
        </div>

        {/* Slide 4: Test custom hypotheses (Text Q4) */}
        <div
          className="carousel-slide"
          role="group"
          aria-roledescription="slide"
          aria-label={`4 of ${SLIDE_META.length}`}
        >
          {result && !result.hasVisualClues ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '0.6rem' }}>
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

        {/* Slide 5: Runtime & quantization details */}
        <div
          className="carousel-slide"
          role="group"
          aria-roledescription="slide"
          aria-label={`5 of ${SLIDE_META.length}`}
        >
          <div className="field-group">
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
      </div>

      <p className="sr-only" role="status" aria-live="polite">
        {announcement}
      </p>

      <nav className="carousel-markers" aria-label="Choose a detail panel">
        {SLIDE_META.map((slide, idx) => (
          <button
            key={slide.id}
            type="button"
            className="carousel-marker"
            aria-label={`Go to ${slide.title}`}
            aria-current={activeIndex === idx ? 'true' : undefined}
            onClick={() => scrollToSlide(idx, true)}
          >
            {slide.label}
          </button>
        ))}
      </nav>
    </section>
  );
}
