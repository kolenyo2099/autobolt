import { useEffect, useMemo, useState } from 'react';
import { geoNaturalEarth1, geoPath, geoGraticule10 } from 'd3-geo';
import { feature } from 'topojson-client';
import worldAtlas from 'world-atlas/countries-110m.json';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import type { GeocellPrediction, ScoredCity, ScoredCountry } from '../engine/types';

interface WorldFeature {
  type: 'Feature';
  id?: string | number;
  properties?: { name?: string };
  geometry: unknown;
}

interface WorldMapProps {
  countries: ScoredCountry[];
  cities: ScoredCity[];
  geocell?: GeocellPrediction | null;
  selectedCountryCode: string | null;
  selectedCity: ScoredCity | null;
  focusTarget?: { points: Array<{ lat: number; lng: number }>; seq: number } | null;
  exifGps: { lat: number; lng: number } | null;
  onSelectCountry: (code: string) => void;
  onSelectCity: (city: ScoredCity) => void;
}

const WIDTH = 960;
const HEIGHT = 500;

export function WorldMap({
  countries,
  cities,
  geocell,
  selectedCountryCode,
  selectedCity,
  focusTarget,
  exifGps,
  onSelectCountry,
  onSelectCity,
}: WorldMapProps) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState<{ x: number; y: number; startPanX: number; startPanY: number } | null>(null);
  const [hoverText, setHoverText] = useState<string | null>(null);
  const [hoveredCityKey, setHoveredCityKey] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);

  const worldFeatures = useMemo(() => {
    const topo = worldAtlas as unknown as Parameters<typeof feature>[0] & {
      objects: { countries: Parameters<typeof feature>[1] };
    };
    const collection = feature(topo, topo.objects.countries) as unknown as {
      features: WorldFeature[];
    };
    return collection.features;
  }, []);

  const projection = useMemo(
    () => geoNaturalEarth1().scale(175).translate([WIDTH / 2, HEIGHT / 2 + 12]),
    [],
  );

  const pathGen = useMemo(() => geoPath(projection), [projection]);
  const graticulePath = useMemo(() => pathGen(geoGraticule10()) ?? '', [pathGen]);

  const countryByIsoNum = useMemo(() => {
    const map = new Map<string, ScoredCountry>();
    for (const c of countries) {
      map.set(c.isoNum, c);
    }
    return map;
  }, [countries]);

  const topProb = countries[0]?.prob || 1e-6;
  const topCities = cities.slice(0, 6);
  const maxHeatProb = geocell?.heatmap[0]?.prob || 1e-6;

  // Focus map automatically on lead point(s) when focusTarget updates
  useEffect(() => {
    if (!focusTarget || focusTarget.points.length === 0) return;
    const projected: Array<[number, number]> = [];
    for (const p of focusTarget.points) {
      const pt = projection([p.lng, p.lat]);
      if (pt) projected.push(pt);
    }
    if (projected.length === 0) return;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const [x, y] of projected) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const dx = maxX - minX;
    const dy = maxY - minY;

    const targetZoom =
      dx < 4 && dy < 4
        ? 3.6
        : Math.max(
            1.8,
            Math.min(
              4.2,
              Math.min((WIDTH * 0.58) / Math.max(dx, 1), (HEIGHT * 0.58) / Math.max(dy, 1)),
            ),
          );

    setZoom(targetZoom);
    setPan({
      x: WIDTH / 2 - cx * targetZoom,
      y: HEIGHT / 2 - cy * targetZoom,
    });
  }, [focusTarget, projection]);

  const visibleLabels = useMemo(() => {
    const show = new Set<string>();
    const labeledPts: Array<[number, number]> = [];

    const tryAdd = (city: ScoredCity, force: boolean) => {
      const key = `${city.countryCode}-${city.name}`;
      const pt = projection([city.lng, city.lat]);
      if (!pt) return;
      const sx = pt[0] * zoom;
      const sy = pt[1] * zoom;
      if (force) {
        show.add(key);
        labeledPts.push([sx, sy]);
        return;
      }
      if (zoom < 2.6) return;
      const tooClose = labeledPts.some(([lx, ly]) => Math.hypot(sx - lx, sy - ly) < 52);
      if (!tooClose) {
        show.add(key);
        labeledPts.push([sx, sy]);
      }
    };

    if (topCities[0]) tryAdd(topCities[0], true);
    for (const city of topCities) {
      const key = `${city.countryCode}-${city.name}`;
      const isSelected = selectedCity?.name === city.name && selectedCity?.countryCode === city.countryCode;
      if (isSelected || hoveredCityKey === key) {
        tryAdd(city, true);
      }
    }
    for (const city of topCities) {
      tryAdd(city, false);
    }
    return show;
  }, [topCities, projection, zoom, selectedCity, hoveredCityKey]);

  const geodesicPath = useMemo(() => {
    const anchor = geocell
      ? { lat: geocell.lat, lng: geocell.lng }
      : topCities[0]
        ? { lat: topCities[0].lat, lng: topCities[0].lng }
        : null;
    if (!exifGps || !anchor) return null;
    const line = {
      type: 'LineString' as const,
      coordinates: [
        [anchor.lng, anchor.lat],
        [exifGps.lng, exifGps.lat],
      ],
    };
    return pathGen(line);
  }, [exifGps, geocell, topCities, pathGen]);

  const focusPoint = (lng: number, lat: number, targetZoom = 3) => {
    const pt = projection([lng, lat]);
    if (!pt) return;
    const clampedZ = Math.max(1, Math.min(8, targetZoom));
    setZoom(clampedZ);
    setPan({
      x: WIDTH / 2 - pt[0] * clampedZ,
      y: HEIGHT / 2 - pt[1] * clampedZ,
    });
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.25 : 0.8;
    const nextZoom = Math.max(1, Math.min(8, zoom * factor));
    if (nextZoom === 1) {
      resetView();
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const svgX = ((e.clientX - rect.left) / rect.width) * WIDTH;
    const svgY = ((e.clientY - rect.top) / rect.height) * HEIGHT;
    const scaleRatio = nextZoom / zoom;
    setZoom(nextZoom);
    setPan({
      x: svgX - (svgX - pan.x) * scaleRatio,
      y: svgY - (svgY - pan.y) * scaleRatio,
    });
  };

  const handlePointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if ((e.target as Element).closest('[data-pin]')) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging({ x: e.clientX, y: e.clientY, startPanX: pan.x, startPanY: pan.y });
  };

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragging) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const dx = ((e.clientX - dragging.x) / rect.width) * WIDTH;
    const dy = ((e.clientY - dragging.y) / rect.height) * HEIGHT;
    setPan({ x: dragging.startPanX + dx, y: dragging.startPanY + dy });
  };

  const handlePointerUp = () => setDragging(null);

  return (
    <section className="panel-card" aria-label="Geolocation Lead Map">
      <div className="map-wrap">
        <svg
          className="map-svg"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          onWheel={handleWheel}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          role="img"
          aria-label="Interactive world map displaying predicted country probabilities, OSV-5M coordinate heatmap, and top city leads"
        >
          <defs>
            <radialGradient id="geocell-heat-grad">
              <stop offset="0%" stopColor="#c26a1b" stopOpacity="0.85" />
              <stop offset="55%" stopColor="#c26a1b" stopOpacity="0.32" />
              <stop offset="100%" stopColor="#c26a1b" stopOpacity="0" />
            </radialGradient>
          </defs>

          <g
            className={dragging ? undefined : 'map-viewport-layer'}
            transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}
          >
            <path d={graticulePath} fill="none" stroke="var(--border-subtle)" strokeWidth={0.45 / zoom} opacity={0.55} />

            {worldFeatures.map((feat, i) => {
              const isoNum = feat.id != null ? String(feat.id).padStart(3, '0') : '';
              const scored = isoNum ? countryByIsoNum.get(isoNum) : undefined;
              const ratio = scored ? Math.min(1, Math.pow(scored.prob / topProb, 0.55)) : 0;
              const isSelected = scored && selectedCountryCode === scored.code;

              let fill = 'var(--map-land)';
              let fillOpacity = 1;
              if (scored && scored.prob > 0.005) {
                fill = 'var(--accent)';
                fillOpacity = 0.16 + ratio * 0.78;
              }

              const d = pathGen(feat as Parameters<typeof pathGen>[0]);
              if (!d) return null;

              return (
                <path
                  key={isoNum || `feat-${i}`}
                  d={d}
                  fill={fill}
                  fillOpacity={fillOpacity}
                  stroke={isSelected ? 'var(--amber)' : 'var(--map-border)'}
                  strokeWidth={(isSelected ? 1.8 : 0.65) / zoom}
                  style={{ cursor: scored ? 'pointer' : 'default' }}
                  onMouseEnter={() => {
                    const label = scored
                      ? `${scored.name}: ${(scored.prob * 100).toFixed(1)}% (#${scored.rank})`
                      : feat.properties?.name ?? 'Territory';
                    setHoverText(label);
                  }}
                  onMouseLeave={() => setHoverText(null)}
                  onClick={() => {
                    if (scored) {
                      onSelectCountry(scored.code);
                      focusPoint(scored.lng, scored.lat, Math.max(zoom, 2.4));
                    }
                  }}
                />
              );
            })}

            {/* OSV-5M Geocell Heatmap Layer */}
            {showHeatmap && geocell && (
              <g className="geocell-heatmap" style={{ pointerEvents: 'none' }}>
                {geocell.heatmap.map((cell) => {
                  const pt = projection([cell.lng, cell.lat]);
                  if (!pt) return null;
                  const rel = Math.pow(cell.prob / maxHeatProb, 0.42);
                  const rad = (6 + rel * 18) / Math.pow(zoom, 0.55);
                  return (
                    <circle
                      key={`heat-${cell.cellId}`}
                      cx={pt[0]}
                      cy={pt[1]}
                      r={rad}
                      fill="url(#geocell-heat-grad)"
                      opacity={0.22 + rel * 0.72}
                    />
                  );
                })}
              </g>
            )}

            {geodesicPath && (
              <path
                d={geodesicPath}
                fill="none"
                stroke="var(--success)"
                strokeWidth={2 / zoom}
                strokeDasharray={`${5 / zoom} ${4 / zoom}`}
              />
            )}

            {topCities.map((city, idx) => {
              const pt = projection([city.lng, city.lat]);
              if (!pt) return null;
              const key = `${city.countryCode}-${city.name}`;
              const isFirst = idx === 0;
              const isSelected = selectedCity?.name === city.name && selectedCity?.countryCode === city.countryCode;
              const showLabel = visibleLabels.has(key);
              const r = (isFirst ? 5.8 : isSelected ? 5.5 : 4) / zoom;

              return (
                <g
                  key={key}
                  data-pin="true"
                  transform={`translate(${pt[0]}, ${pt[1]})`}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() => {
                    setHoveredCityKey(key);
                    setHoverText(
                      `#${idx + 1} ${city.name}, ${city.countryName} (${(city.hierarchicalProb * 100).toFixed(1)}%)`,
                    );
                  }}
                  onMouseLeave={() => {
                    setHoveredCityKey(null);
                    setHoverText(null);
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectCity(city);
                    focusPoint(city.lng, city.lat, Math.max(zoom, 3.2));
                  }}
                >
                  <circle
                    r={r}
                    fill={isFirst || isSelected ? 'var(--amber)' : 'var(--bg-surface)'}
                    stroke={isFirst || isSelected ? 'var(--bg-surface)' : 'var(--accent)'}
                    strokeWidth={1.7 / zoom}
                  />
                  {showLabel && (
                    <text
                      y={-r - 4 / zoom}
                      textAnchor="middle"
                      fontSize={10.5 / zoom}
                      fontWeight={700}
                      fill="var(--text-primary)"
                      paintOrder="stroke"
                      stroke="var(--bg-surface)"
                      strokeWidth={3 / zoom}
                    >
                      {isFirst ? city.name : `${idx + 1}. ${city.name}`}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Continuous OSV-5M Coordinate Crosshair */}
            {geocell && (() => {
              const pt = projection([geocell.lng, geocell.lat]);
              if (!pt) return null;
              const s = 6 / zoom;
              return (
                <g
                  data-pin="true"
                  transform={`translate(${pt[0]}, ${pt[1]})`}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() =>
                    setHoverText(
                      `OSV-5M Est. (${geocell.lat.toFixed(2)}, ${geocell.lng.toFixed(2)}) · ±${Math.round(geocell.spreadKm)} km`,
                    )
                  }
                  onMouseLeave={() => setHoverText(null)}
                  onClick={(e) => {
                    e.stopPropagation();
                    focusPoint(geocell.lng, geocell.lat, Math.max(zoom, 3.5));
                  }}
                >
                  <circle
                    r={s * 1.4}
                    fill="none"
                    stroke="var(--amber)"
                    strokeWidth={1.5 / zoom}
                  />
                  <line x1={-s * 1.8} y1={0} x2={s * 1.8} y2={0} stroke="var(--amber)" strokeWidth={1.3 / zoom} />
                  <line x1={0} y1={-s * 1.8} x2={0} y2={s * 1.8} stroke="var(--amber)" strokeWidth={1.3 / zoom} />
                </g>
              );
            })()}

            {exifGps && (() => {
              const pt = projection([exifGps.lng, exifGps.lat]);
              if (!pt) return null;
              return (
                <g transform={`translate(${pt[0]}, ${pt[1]})`}>
                  <circle
                    r={6.5 / zoom}
                    fill="var(--success)"
                    stroke="var(--bg-surface)"
                    strokeWidth={2 / zoom}
                  />
                  <text
                    y={15 / zoom}
                    textAnchor="middle"
                    fontSize={10 / zoom}
                    fontWeight={700}
                    fill="var(--success)"
                    paintOrder="stroke"
                    stroke="var(--bg-surface)"
                    strokeWidth={3 / zoom}
                  >
                    EXIF Ground Truth
                  </text>
                </g>
              );
            })()}
          </g>
        </svg>

        {hoverText && <div className="map-tooltip">{hoverText}</div>}

        <div className="map-controls">
          <button
            type="button"
            className="btn btn-sm"
            aria-label="Zoom in map"
            onClick={() => setZoom((z) => Math.min(8, z * 1.4))}
          >
            <ZoomIn size={14} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="btn btn-sm"
            aria-label="Zoom out map"
            onClick={() => {
              const next = Math.max(1, zoom / 1.4);
              if (next === 1) resetView();
              else setZoom(next);
            }}
          >
            <ZoomOut size={14} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="btn btn-sm"
            aria-label="Reset map view"
            onClick={resetView}
          >
            <RotateCcw size={14} aria-hidden="true" />
          </button>
        </div>

        <div className="map-legend">
          <span className="legend-swatch">
            <span className="swatch-dot" style={{ background: 'var(--accent)' }} />
            Country Signal
          </span>
          <span className="legend-swatch">
            <span className="swatch-dot" style={{ background: 'var(--amber)' }} />
            Coordinate &amp; City Lead
          </span>
          {geocell && (
            <button
              type="button"
              className="legend-toggle"
              aria-pressed={showHeatmap}
              onClick={() => setShowHeatmap((v) => !v)}
            >
              Heatmap: {showHeatmap ? 'On' : 'Off'}
            </button>
          )}
          {exifGps && (
            <span className="legend-swatch">
              <span className="swatch-dot" style={{ background: 'var(--success)' }} />
              EXIF GPS
            </span>
          )}
        </div>
      </div>
    </section>
  );
}
