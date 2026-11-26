import { useEffect, useRef, useState } from 'react';
import {
  Search,
  Plus,
  Trash2,
  Sun,
  Clock,
  Mountain,
  MapPin,
  Loader2,
  Compass,
  X,
} from 'lucide-react';
import { searchAllClues } from '../engine/clues';
import {
  chronolocateAtPoint,
  computeSolarBand,
  estimateHeadingFromLitFace,
  shadowRatioToElevation,
  solarPosition,
} from '../engine/sun';
import {
  OVERPASS_PRESETS,
  queryOverpassPreset,
  searchNominatimPoi,
  type OverpassPresetId,
} from '../engine/osm';
import { computeSkylineProfile } from '../engine/terrain';
import type {
  AnalysisResult,
  ClueMatch,
  EvidenceEntry,
  ExifMetadata,
  OsmPin,
  SkylineProfile,
  SolarBandPoint,
  SolarTimeWindow,
} from '../engine/types';

/* ── 1. Clue Lookup & Bayesian Evidence Board ── */

interface ClueBoardPanelProps {
  result: AnalysisResult | null;
  evidenceEntries: EvidenceEntry[];
  onAddEvidence: (entry: EvidenceEntry) => void;
  onToggleEvidence: (id: string) => void;
  onRemoveEvidence: (id: string) => void;
  onClearEvidence: () => void;
}

export function ClueBoardPanel({
  result,
  evidenceEntries,
  onAddEvidence,
  onToggleEvidence,
  onRemoveEvidence,
  onClearEvidence,
}: ClueBoardPanelProps) {
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [matches, setMatches] = useState<ClueMatch[]>([]);
  const [lookupError, setLookupError] = useState<string | null>(null);

  const runSearch = async (qOverride?: string) => {
    const q = (qOverride ?? query).trim();
    if (!q || !result) return;
    setSearching(true);
    setLookupError(null);
    try {
      const hits = await searchAllClues(q, result.countries, result.countries[0]?.code);
      setMatches(hits);
    } catch (err) {
      setLookupError(err instanceof Error ? err.message : String(err));
    } finally {
      setSearching(false);
    }
  };

  const applyMatch = (m: ClueMatch) => {
    onAddEvidence({
      id: `${m.id}-${Date.now()}`,
      kind: m.kind,
      label: m.title,
      detail: m.subtitle,
      status: m.status,
      countryCodes: m.countryCodes,
      likelihoodRatio: m.likelihoodRatio,
      active: true,
      source: 'lookup',
    });
  };

  return (
    <div className="subpanel-stack">
      <p className="subpanel-desc">
        Search offline <code>geo-sleuth</code> tables (281 calling codes, 241 driving sides, 60 territories, and 8 regional plate/area-code packs). Applied clues update country probabilities via clamped Bayesian likelihood ratios without discarding competing options.
      </p>

      <div className="preset-row">
        {[
          { label: 'Left-hand traffic', q: 'left' },
          { label: 'Right-hand traffic', q: 'right' },
          { label: '+81 Japan', q: '+81' },
          { label: '+594 Guiana', q: '+594' },
          { label: '010 Beijing', q: '010' },
          { label: 'Plate 06 Ankara', q: '06' },
        ].map((chip) => (
          <button
            key={chip.label}
            type="button"
            className="btn btn-sm btn-pill"
            onClick={() => {
              setQuery(chip.q);
              void runSearch(chip.q);
            }}
          >
            {chip.label}
          </button>
        ))}
      </div>

      <form
        className="inline-search-row"
        onSubmit={(e) => {
          e.preventDefault();
          void runSearch();
        }}
      >
        <input
          type="text"
          className="text-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter calling code (+81), left/right, plate prefix (WP, 京A), or territory…"
          aria-label="Clue lookup query"
        />
        <button
          type="submit"
          className="btn btn-sm btn-primary"
          disabled={!result || !query.trim() || searching}
        >
          {searching ? <Loader2 size={13} aria-hidden="true" /> : <Search size={13} aria-hidden="true" />}
          <span>Lookup</span>
        </button>
      </form>

      {lookupError && <div className="vlm-error">{lookupError}</div>}

      {matches.length > 0 && (
        <div className="clue-results-list">
          {matches.map((m) => (
            <div key={m.id} className="clue-match-row">
              <div style={{ minWidth: 0 }}>
                <div className="lead-name">{m.title}</div>
                <div className="lead-sub">{m.subtitle}</div>
              </div>
              <button
                type="button"
                className="btn btn-sm"
                disabled={m.countryCodes.length === 0}
                title={
                  m.countryCodes.length > 0
                    ? `Boost ${m.countryCodes.slice(0, 6).join(', ')} (${m.likelihoodRatio.toFixed(1)}×)`
                    : 'No matching catalog country codes'
                }
                onClick={() => applyMatch(m)}
              >
                <Plus size={13} aria-hidden="true" />
                <span>Apply ({m.likelihoodRatio.toFixed(1)}×)</span>
              </button>
            </div>
          ))}
        </div>
      )}

      {evidenceEntries.length > 0 && (
        <div className="evidence-ledger">
          <div className="geo-col-head">
            <span className="geo-col-title">Active Evidence Board ({evidenceEntries.filter((e) => e.active).length})</span>
            <button type="button" className="btn btn-sm btn-ghost" onClick={onClearEvidence}>
              Clear all
            </button>
          </div>
          {evidenceEntries.map((ev) => (
            <div key={ev.id} className={`evidence-row ${ev.active ? '' : 'muted'}`}>
              <label className="evidence-check">
                <input
                  type="checkbox"
                  checked={ev.active}
                  onChange={() => onToggleEvidence(ev.id)}
                />
                <span className={`status-tag status-${ev.status}`}>{ev.status}</span>
                <span className="evidence-label">{ev.label}</span>
              </label>
              <button
                type="button"
                className="btn btn-sm btn-ghost"
                aria-label={`Remove ${ev.label}`}
                onClick={() => onRemoveEvidence(ev.id)}
              >
                <Trash2 size={12} aria-hidden="true" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── 2. NOAA Solar, Shadow & Chronolocation Panel ── */

interface SunGeometryPanelProps {
  result: AnalysisResult | null;
  exif: ExifMetadata | null;
  solarBandCount: number;
  onPlotSolarBand: (points: SolarBandPoint[]) => void;
  onClearSolarBand: () => void;
}

export function SunGeometryPanel({
  result,
  exif,
  solarBandCount,
  onPlotSolarBand,
  onClearSolarBand,
}: SunGeometryPanelProps) {
  const defaultDate = exif?.dateTimeOriginal?.slice(0, 10) || '2025-06-21';
  const defaultTime = exif?.dateTimeOriginal?.slice(11, 16) || '14:30';
  const anchorLat = result?.geocell?.lat ?? result?.citiesHierarchical[0]?.lat ?? 37.77;
  const anchorLng = result?.geocell?.lng ?? result?.citiesHierarchical[0]?.lng ?? -122.42;

  const [dateStr, setDateStr] = useState(defaultDate);
  const [timeStr, setTimeStr] = useState(defaultTime);
  const [tzOffset, setTzOffset] = useState(String(Math.round(anchorLng / 15)));
  const tzTouchedRef = useRef(false);

  useEffect(() => {
    if (!tzTouchedRef.current && result) {
      const lng = result.geocell?.lng ?? result.citiesHierarchical[0]?.lng;
      if (typeof lng === 'number') {
        setTzOffset(String(Math.round(lng / 15)));
      }
    }
  }, [result]);

  useEffect(() => {
    if (exif?.dateTimeOriginal) {
      setDateStr(exif.dateTimeOriginal.slice(0, 10));
      setTimeStr(exif.dateTimeOriginal.slice(11, 16));
    }
  }, [exif?.dateTimeOriginal]);
  const [shadowRatio, setShadowRatio] = useState('1.1');
  const [shadowBearing, setShadowBearing] = useState('');
  const [litSide, setLitSide] = useState<'left' | 'right' | 'front' | 'back'>('left');
  const [windows, setWindows] = useState<SolarTimeWindow[] | null>(null);
  const [bandSummary, setBandSummary] = useState<string | null>(null);
  const [headingHint, setHeadingHint] = useState<string | null>(null);

  const elevDeg = shadowRatioToElevation(Number(shadowRatio) || 1);

  const handlePlotBand = () => {
    const [y, m, d] = dateStr.split('-').map(Number);
    const [hh, mm] = timeStr.split(':').map(Number);
    if (!y || !m || !d) return;
    const offsetH = Number(tzOffset) || 0;
    const utcMs = Date.UTC(y, m - 1, d, hh || 0, mm || 0, 0) - offsetH * 3600 * 1000;
    const az = shadowBearing.trim() !== '' ? Number(shadowBearing) : null;
    const pts = computeSolarBand({
      utcDate: new Date(utcMs),
      elevationDeg: elevDeg,
      elevTolDeg: 5.0,
      shadowBearingDeg: az,
      azTolDeg: 22,
    });
    onPlotSolarBand(pts);
    setBandSummary(
      pts.length > 0
        ? `Plotted ${pts.length} matching coordinates (sun elev ${elevDeg.toFixed(1)}° ±5° at ${timeStr} UTC${offsetH >= 0 ? '+' : ''}${offsetH}).`
        : 'No sunlit coordinates match those angles at that UTC time.',
    );
  };

  const handleSolveTime = () => {
    const offsetH = Number(tzOffset) || 0;
    const az = shadowBearing.trim() !== '' ? Number(shadowBearing) : null;
    const wins = chronolocateAtPoint({
      lat: anchorLat,
      lng: anchorLng,
      dateStr,
      tzOffsetHours: offsetH,
      elevationDeg: elevDeg,
      elevTolDeg: 5.0,
      shadowBearingDeg: az,
      azTolDeg: 20,
    });
    setWindows(wins);
    if (wins[0]) {
      const est = estimateHeadingFromLitFace(wins[0].azimuthDeg, litSide);
      setHeadingHint(
        `At peak match (${wins[0].peakMatchTime}), sun azimuth ${wins[0].azimuthDeg.toFixed(0)}° + lit side "${litSide}" → camera facing ~${est.headingDeg.toFixed(0)}° (${est.cardinal})`,
      );
    } else {
      setHeadingHint(null);
    }
  };

  const handleCheckPointHeading = () => {
    const [y, m, d] = dateStr.split('-').map(Number);
    const [hh, mm] = timeStr.split(':').map(Number);
    if (!y || !m || !d) return;
    const offsetH = Number(tzOffset) || 0;
    const utcMs = Date.UTC(y, m - 1, d, hh || 0, mm || 0, 0) - offsetH * 3600 * 1000;
    const pos = solarPosition(anchorLat, anchorLng, new Date(utcMs));
    const est = estimateHeadingFromLitFace(pos.azimuthDeg, litSide);
    setHeadingHint(
      `At (${anchorLat.toFixed(2)}°, ${anchorLng.toFixed(2)}°) on ${dateStr} ${timeStr}: sun elev ${pos.elevationDeg.toFixed(1)}°, shadow bearing ${pos.shadowBearingDeg.toFixed(0)}°, camera heading ~${est.headingDeg.toFixed(0)}° (${est.cardinal}).`,
    );
  };

  return (
    <div className="subpanel-stack">
      <p className="subpanel-desc">
        NOAA / Meeus solar trigonometry (<code>≤0.02°</code>). Compute matching geographic solar bands from known time + shadow ratio, or solve capture time-of-day &amp; camera heading at your top lead.
      </p>

      <div className="solar-grid">
        <label className="field-group">
          <span>Date</span>
          <input type="date" className="field-select" value={dateStr} onChange={(e) => setDateStr(e.target.value)} />
        </label>
        <label className="field-group">
          <span>Local Time</span>
          <input type="time" className="field-select" value={timeStr} onChange={(e) => setTimeStr(e.target.value)} />
        </label>
        <label className="field-group">
          <span>UTC Offset (h)</span>
          <input
            type="number"
            step="0.5"
            min="-12"
            max="14"
            className="field-select mono"
            value={tzOffset}
            onChange={(e) => {
              tzTouchedRef.current = true;
              setTzOffset(e.target.value);
            }}
          />
        </label>
        <label className="field-group">
          <span>Shadow Ratio (L / H)</span>
          <input
            type="number"
            step="0.1"
            min="0.05"
            max="15"
            className="field-select mono"
            value={shadowRatio}
            onChange={(e) => setShadowRatio(e.target.value)}
          />
        </label>
        <label className="field-group">
          <span>Shadow Bearing° (opt.)</span>
          <input
            type="number"
            step="5"
            min="0"
            max="359"
            placeholder="0-359"
            className="field-select mono"
            value={shadowBearing}
            onChange={(e) => setShadowBearing(e.target.value)}
          />
        </label>
        <label className="field-group">
          <span>Sunlit Wall Side</span>
          <select
            className="field-select"
            value={litSide}
            onChange={(e) => setLitSide(e.target.value as typeof litSide)}
          >
            <option value="left">Lit on Left</option>
            <option value="right">Lit on Right</option>
            <option value="front">Lit Ahead (backlit behind)</option>
            <option value="back">Lit From Behind</option>
          </select>
        </label>
      </div>

      <div className="preset-row">
        <button type="button" className="btn btn-sm btn-primary" onClick={handlePlotBand}>
          <Sun size={13} aria-hidden="true" />
          <span>Plot Solar Band on Map</span>
        </button>
        <button type="button" className="btn btn-sm" onClick={handleSolveTime}>
          <Clock size={13} aria-hidden="true" />
          <span>Solve Capture Time</span>
        </button>
        <button type="button" className="btn btn-sm" onClick={handleCheckPointHeading}>
          <Compass size={13} aria-hidden="true" />
          <span>Heading Check</span>
        </button>
        {solarBandCount > 0 && (
          <button type="button" className="btn btn-sm btn-ghost" onClick={onClearSolarBand}>
            <X size={13} aria-hidden="true" />
            <span>Clear Band</span>
          </button>
        )}
      </div>

      {bandSummary && <div className="subpanel-note">{bandSummary}</div>}
      {headingHint && <div className="subpanel-note mono">{headingHint}</div>}

      {windows && (
        <div className="subpanel-note">
          {windows.length === 0 ? (
            <span>No matching solar elevation window found at ({anchorLat.toFixed(2)}°, {anchorLng.toFixed(2)}°) on {dateStr}.</span>
          ) : (
            <div>
              <strong>Candidate local times at ({anchorLat.toFixed(2)}°, {anchorLng.toFixed(2)}°):</strong>{' '}
              {windows
                .map((w) => `${w.startTime}–${w.endTime} (peak ${w.peakMatchTime}, shadow→${w.shadowBearingDeg.toFixed(0)}°)`)
                .join(' · ')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── 3. OpenStreetMap POI/Structure + 3D Terrarium Skyline Overlay Panel ── */

interface OsmTerrainPanelProps {
  result: AnalysisResult | null;
  exif: ExifMetadata | null;
  osmPins: OsmPin[];
  skyline: SkylineProfile | null;
  onSetOsmPins: (pins: OsmPin[]) => void;
  onFocusCoords: (lat: number, lng: number) => void;
  onSetSkyline: (profile: SkylineProfile | null) => void;
}

export function OsmTerrainPanel({
  result,
  exif,
  osmPins,
  skyline,
  onSetOsmPins,
  onFocusCoords,
  onSetSkyline,
}: OsmTerrainPanelProps) {
  const anchorLat = result?.geocell?.lat ?? result?.citiesHierarchical[0]?.lat ?? 37.7749;
  const anchorLng = result?.geocell?.lng ?? result?.citiesHierarchical[0]?.lng ?? -122.4194;
  const cityHint = result?.citiesHierarchical[0]?.name ?? '';

  const [poiQuery, setPoiQuery] = useState('');
  const [presetId, setPresetId] = useState<OverpassPresetId>('tram_street');
  const [radiusKm, setRadiusKm] = useState('8');
  const [osmBusy, setOsmBusy] = useState(false);
  const [osmNote, setOsmNote] = useState<string | null>(null);

  const [headingDeg, setHeadingDeg] = useState(25);
  const [hfovDeg, setHfovDeg] = useState(Math.round(exif?.hfovDeg ?? 62));
  const [pitchDeg, setPitchDeg] = useState(0);
  const [skyBusy, setSkyBusy] = useState(false);
  const [skyNote, setSkyNote] = useState<string | null>(null);

  const handlePoiSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!poiQuery.trim()) return;
    setOsmBusy(true);
    setOsmNote(null);
    try {
      const pins = await searchNominatimPoi(poiQuery, { lat: anchorLat, lng: anchorLng }, 60);
      onSetOsmPins(pins);
      if (pins[0]) {
        onFocusCoords(pins[0].lat, pins[0].lng);
      }
      setOsmNote(
        pins.length > 0
          ? `Found ${pins.length} Nominatim matches near ${cityHint || 'lead'}`
          : 'No matching POIs found in Nominatim bbox.',
      );
    } catch (err) {
      setOsmNote(err instanceof Error ? err.message : String(err));
    } finally {
      setOsmBusy(false);
    }
  };

  const handleRunOverpass = async () => {
    setOsmBusy(true);
    setOsmNote(null);
    try {
      const { pins, mirrorUsed } = await queryOverpassPreset({
        preset: presetId,
        lat: anchorLat,
        lng: anchorLng,
        radiusM: (Number(radiusKm) || 6) * 1000,
      });
      onSetOsmPins(pins);
      setOsmNote(
        pins.length > 0
          ? `Found ${pins.length} OSM features via ${mirrorUsed}`
          : `0 OSM features returned (${mirrorUsed}). Note: OSM detail varies by region.`,
      );
    } catch (err) {
      setOsmNote(err instanceof Error ? err.message : String(err));
    } finally {
      setOsmBusy(false);
    }
  };

  const handleRenderSkyline = async (nextHeading = headingDeg, nextHfov = hfovDeg, nextPitch = pitchDeg) => {
    setSkyBusy(true);
    setSkyNote(null);
    try {
      const prof = await computeSkylineProfile({
        lat: anchorLat,
        lng: anchorLng,
        headingDeg: nextHeading,
        hfovDeg: nextHfov,
        pitchDeg: nextPitch,
        maxRangeKm: 28,
      });
      onSetSkyline(prof);
      setSkyNote(
        `Rendered Terrarium DEM horizon at (${anchorLat.toFixed(3)}°, ${anchorLng.toFixed(3)}°), obs elev ${Math.round(prof.observerElevM)} m, max ridge ${prof.maxElevDeg.toFixed(1)}°.`,
      );
    } catch (err) {
      setSkyNote(err instanceof Error ? err.message : String(err));
    } finally {
      setSkyBusy(false);
    }
  };

  return (
    <div className="subpanel-stack">
      <div className="geo-col-head">
        <span className="geo-col-title">OpenStreetMap POI &amp; Structure Finder</span>
        {osmPins.length > 0 && (
          <button type="button" className="btn btn-sm btn-ghost" onClick={() => onSetOsmPins([])}>
            Clear {osmPins.length} pins
          </button>
        )}
      </div>

      <form className="inline-search-row" onSubmit={handlePoiSearch}>
        <input
          type="text"
          className="text-input"
          value={poiQuery}
          onChange={(e) => setPoiQuery(e.target.value)}
          placeholder={`Search shop, sign, or street near ${cityHint || 'top lead'}…`}
          aria-label="Nominatim POI query"
        />
        <button type="submit" className="btn btn-sm btn-primary" disabled={osmBusy || !poiQuery.trim()}>
          {osmBusy ? <Loader2 size={13} aria-hidden="true" /> : <MapPin size={13} aria-hidden="true" />}
          <span>Find POI</span>
        </button>
      </form>

      <div className="inline-search-row">
        <select
          className="field-select"
          value={presetId}
          onChange={(e) => setPresetId(e.target.value as OverpassPresetId)}
          aria-label="OSM structural preset"
          style={{ flex: '1 1 200px' }}
        >
          {OVERPASS_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        <input
          type="number"
          min="1"
          max="20"
          className="field-select mono"
          value={radiusKm}
          onChange={(e) => setRadiusKm(e.target.value)}
          aria-label="Radius in km"
          style={{ inlineSize: 72 }}
        />
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>km</span>
        <button type="button" className="btn btn-sm" disabled={osmBusy} onClick={handleRunOverpass}>
          <span>Scan Structures</span>
        </button>
      </div>

      {osmNote && <div className="subpanel-note">{osmNote}</div>}

      {osmPins.length > 0 && (
        <div className="preset-row">
          {osmPins.slice(0, 6).map((pin) => (
            <button
              key={pin.id}
              type="button"
              className="btn btn-sm btn-pill"
              onClick={() => onFocusCoords(pin.lat, pin.lng)}
              title={`${pin.lat.toFixed(4)}, ${pin.lng.toFixed(4)}`}
            >
              {pin.name}
            </button>
          ))}
        </div>
      )}

      <hr className="subpanel-divider" />

      <div className="geo-col-head">
        <span className="geo-col-title">3D Terrain Skyline Overlay (AWS Terrarium DEM)</span>
        {skyline && (
          <button type="button" className="btn btn-sm btn-ghost" onClick={() => onSetSkyline(null)}>
            Hide overlay
          </button>
        )}
      </div>

      <div className="solar-grid">
        <label className="field-group">
          <span>Heading: {headingDeg}°</span>
          <input
            type="range"
            min="0"
            max="355"
            step="5"
            value={headingDeg}
            onChange={(e) => setHeadingDeg(Number(e.target.value))}
          />
        </label>
        <label className="field-group">
          <span>HFOV: {hfovDeg}°</span>
          <input
            type="range"
            min="25"
            max="110"
            step="2"
            value={hfovDeg}
            onChange={(e) => setHfovDeg(Number(e.target.value))}
          />
        </label>
        <label className="field-group">
          <span>Horizon Pitch: {pitchDeg}°</span>
          <input
            type="range"
            min="-15"
            max="15"
            step="1"
            value={pitchDeg}
            onChange={(e) => setPitchDeg(Number(e.target.value))}
          />
        </label>
      </div>

      <div className="preset-row">
        <button
          type="button"
          className="btn btn-sm btn-primary"
          disabled={skyBusy}
          onClick={() => handleRenderSkyline()}
        >
          {skyBusy ? <Loader2 size={13} aria-hidden="true" /> : <Mountain size={13} aria-hidden="true" />}
          <span>{skyBusy ? 'Fetching DEM tiles…' : 'Render Skyline on Photo'}</span>
        </button>
      </div>

      {skyNote && <div className="subpanel-note">{skyNote}</div>}
    </div>
  );
}
