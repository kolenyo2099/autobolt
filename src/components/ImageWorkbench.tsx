import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Upload,
  Crop,
  Scan,
  Columns3,
  Eye,
  EyeOff,
  MapPin,
  RefreshCw,
  FlipHorizontal2,
  Grid2x2,
  ScanText,
  X,
} from 'lucide-react';
import type {
  AnalysisMode,
  ExifMetadata,
  RoiRect,
  SamplePhoto,
  SkylineProfile,
  VlmModelId,
} from '../engine/types';
import { haversineDistanceKm } from '../engine/exif';
import { generateEdgeCornerCrops } from '../engine/preprocess';

interface ImageWorkbenchProps {
  imageUrl: string | null;
  imageName: string;
  imageBitmap: ImageBitmap | null;
  isMirrored: boolean;
  skyline: SkylineProfile | null;
  samples: SamplePhoto[];
  activeSampleId: string | null;
  mode: AnalysisMode;
  roi: RoiRect;
  exif: ExifMetadata | null;
  gpsRevealed: boolean;
  topCityCoords: { name: string; lat: number; lng: number } | null;
  analyzing: boolean;
  modelReady: boolean;
  vlmLoading: boolean;
  vlmModelId: VlmModelId;
  vlmGrounded: boolean;
  onSelectFile: (file: File) => void;
  onSelectSample: (sample: SamplePhoto) => void;
  onChangeMode: (mode: AnalysisMode) => void;
  onChangeRoi: (roi: RoiRect) => void;
  onToggleMirror: () => void;
  onClearSkyline: () => void;
  onRunAnalysis: (modeOverride?: AnalysisMode, roiOverride?: RoiRect) => void;
  onRunVlmOnRoi: (modelId: VlmModelId, grounded: boolean, roiOverride: RoiRect) => void;
  onToggleRevealGps: () => void;
}

const MODE_LABELS: Record<AnalysisMode, string> = {
  center: 'Center 336',
  multicrop: 'Wide 3-Crop',
  roi: 'Focus ROI',
};

const MODE_DESCRIPTIONS: Record<AnalysisMode, string> = {
  center: 'Fast standard 336×336 center crop matching StreetCLIP pretraining.',
  multicrop: 'Averages 3 spatial crops along the long edge for wide panoramas.',
  roi: 'Drag a box on the photo to isolate a specific street sign or facade.',
};

export function ImageWorkbench({
  imageUrl,
  imageName,
  imageBitmap,
  isMirrored,
  skyline,
  samples,
  activeSampleId,
  mode,
  roi,
  exif,
  gpsRevealed,
  topCityCoords,
  analyzing,
  modelReady,
  vlmLoading,
  vlmModelId,
  vlmGrounded,
  onSelectFile,
  onSelectSample,
  onChangeMode,
  onChangeRoi,
  onToggleMirror,
  onClearSkyline,
  onRunAnalysis,
  onRunVlmOnRoi,
  onToggleRevealGps,
}: ImageWorkbenchProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [drawingStart, setDrawingStart] = useState<{ x: number; y: number } | null>(null);
  const [edgesOpen, setEdgesOpen] = useState(false);

  const edgeCrops = useMemo(() => {
    if (!edgesOpen || !imageBitmap) return [];
    return generateEdgeCornerCrops(imageBitmap);
  }, [edgesOpen, imageBitmap]);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            onSelectFile(file);
            break;
          }
        }
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [onSelectFile]);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file && file.type.startsWith('image/')) {
      onSelectFile(file);
    }
  };

  const toNormalizedCoords = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = stageRef.current;
    if (!el) return { x: 0.5, y: 0.5 };
    const rect = el.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return { x, y };
  };

  const handleRoiPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (mode !== 'roi' || !imageUrl) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const pt = toNormalizedCoords(e);
    setDrawingStart(pt);
    onChangeRoi({ x: pt.x, y: pt.y, w: 0.02, h: 0.02 });
  };

  const handleRoiPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!drawingStart || mode !== 'roi') return;
    const pt = toNormalizedCoords(e);
    const x = Math.min(drawingStart.x, pt.x);
    const y = Math.min(drawingStart.y, pt.y);
    const w = Math.max(0.05, Math.abs(pt.x - drawingStart.x));
    const h = Math.max(0.05, Math.abs(pt.y - drawingStart.y));
    onChangeRoi({
      x,
      y,
      w: Math.min(1 - x, w),
      h: Math.min(1 - y, h),
    });
  };

  const handleRoiPointerUp = () => {
    if (!drawingStart) return;
    setDrawingStart(null);
  };

  const exifErrorKm =
    gpsRevealed && exif?.gps && topCityCoords
      ? haversineDistanceKm(exif.gps.lat, exif.gps.lng, topCityCoords.lat, topCityCoords.lng)
      : null;

  const exifSummaryPreview = exif
    ? [
        [exif.make, exif.model].filter(Boolean).join(' ') || 'Camera metadata',
        exif.gps ? (gpsRevealed ? 'GPS shown' : 'GPS hidden') : 'No GPS',
      ].join(' · ')
    : 'No EXIF data';

  return (
    <section className="panel-card" aria-label="Photo Evidence">
      <div className="panel-section">
        <div className="workbench-header">
          <div className="photo-meta-title">
            <span className="photo-name">{imageName || 'Select or drop a photo'}</span>
            {imageBitmap && (
              <span className="photo-dims mono">
                {imageBitmap.width}×{imageBitmap.height}
                {isMirrored ? ' · Mirrored' : ''}
                {exif?.hfovDeg ? ` · ~${Math.round(exif.hfovDeg)}° HFOV` : ''}
              </span>
            )}
          </div>

          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="sr-only"
              onChange={(e) => {
                handleFiles(e.target.files);
                e.currentTarget.value = '';
              }}
            />
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={14} aria-hidden="true" />
              <span>Upload photo</span>
            </button>
          </div>
        </div>

        <div
          ref={stageRef}
          className={`image-stage ${imageUrl ? 'has-image' : ''} ${dragOver ? 'drag-active' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
        >
          {imageUrl ? (
            <>
              <img
                src={imageUrl}
                alt={`Evidence photo: ${imageName}`}
                className="stage-photo"
                style={isMirrored ? { transform: 'scaleX(-1)' } : undefined}
                decoding="async"
              />
              {skyline && skyline.samples.length > 1 && (() => {
                const vf = (skyline.hfovDeg * 3) / 4;
                const yOfElev = (deg: number) => {
                  const yNorm = 0.5 - (deg - skyline.pitchDeg) / Math.max(12, vf);
                  return Math.max(2, Math.min(98, yNorm * 100));
                };
                const pts = skyline.samples
                  .map((s) => `${(s.xNorm * 100).toFixed(2)},${yOfElev(s.elevDeg).toFixed(2)}`)
                  .join(' ');
                const hY = yOfElev(0);
                return (
                  <div className="skyline-overlay" aria-label="3D terrain skyline overlay">
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="skyline-svg">
                      <line
                        x1="0"
                        y1={hY}
                        x2="100"
                        y2={hY}
                        stroke="rgba(255,255,255,0.5)"
                        strokeWidth="0.45"
                        strokeDasharray="1.5 1.5"
                      />
                      <polyline
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="1.15"
                        vectorEffect="non-scaling-stroke"
                        points={pts}
                      />
                    </svg>
                    <button
                      type="button"
                      className="skyline-badge"
                      onClick={onClearSkyline}
                      title="Dismiss DEM terrain skyline overlay"
                    >
                      <span>
                        DEM Skyline · Hdg {skyline.headingDeg}° · Max {skyline.maxElevDeg.toFixed(1)}°
                      </span>
                      <X size={12} aria-hidden="true" />
                    </button>
                  </div>
                );
              })()}
              {mode === 'roi' && (
                <div
                  className="roi-overlay"
                  onPointerDown={handleRoiPointerDown}
                  onPointerMove={handleRoiPointerMove}
                  onPointerUp={handleRoiPointerUp}
                  title="Drag to select a region of interest"
                >
                  <div
                    className="roi-box"
                    style={{
                      left: `${roi.x * 100}%`,
                      top: `${roi.y * 100}%`,
                      width: `${roi.w * 100}%`,
                      height: `${roi.h * 100}%`,
                    }}
                  >
                    <span className="roi-badge">
                      ROI {Math.round(roi.w * 100)}% × {Math.round(roi.h * 100)}%
                    </span>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="drop-empty">
              <Upload size={26} aria-hidden="true" />
              <strong>Drop a street photo, paste from clipboard, or pick a sample</strong>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Processed 100% locally in your browser. Photos never leave your device.
              </span>
            </div>
          )}
        </div>

        {mode === 'roi' && imageUrl && (
          <div className="roi-inline-bar">
            <span>Drag across the photo to isolate a sign, building, or detail.</span>
            <button
              type="button"
              className="btn btn-sm btn-primary"
              disabled={!imageBitmap || !modelReady || analyzing}
              onClick={() => onRunAnalysis()}
            >
              <RefreshCw size={13} aria-hidden="true" />
              <span>{analyzing ? 'Analyzing…' : 'Analyze ROI'}</span>
            </button>
          </div>
        )}

        {samples.length > 0 && (
          <div className="samples-strip" aria-label="Sample benchmark photos">
            {samples.map((s) => (
              <button
                key={s.id}
                type="button"
                className="sample-chip"
                aria-current={activeSampleId === s.id ? 'true' : undefined}
                onClick={() => onSelectSample(s)}
              >
                <img
                  src={s.url}
                  alt={s.title}
                  className="sample-thumb"
                  loading="lazy"
                  decoding="async"
                />
                <span className="sample-meta">
                  <span className="sample-title">{s.title}</span>
                  <span className="sample-sub">{s.subtitle}</span>
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Secondary control 1: Crop & framing options */}
      <details className="disclosure embedded">
        <summary>
          <span className="summary-title">Crop &amp; framing options</span>
          <span className="summary-preview">{MODE_LABELS[mode]}</span>
        </summary>
        <div className="disclosure-body">
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '0.5rem', marginTop: '0.35rem' }}>
            <div className="mode-pills" role="group" aria-label="Crop and analysis mode">
              <button
                type="button"
                className="mode-pill"
                aria-pressed={mode === 'center'}
                onClick={() => onChangeMode('center')}
              >
                <Scan size={13} style={{ verticalAlign: '-2px', marginRight: 4 }} aria-hidden="true" />
                Center 336
              </button>
              <button
                type="button"
                className="mode-pill"
                aria-pressed={mode === 'multicrop'}
                onClick={() => onChangeMode('multicrop')}
              >
                <Columns3 size={13} style={{ verticalAlign: '-2px', marginRight: 4 }} aria-hidden="true" />
                Wide 3-Crop
              </button>
              <button
                type="button"
                className="mode-pill"
                aria-pressed={mode === 'roi'}
                onClick={() => onChangeMode('roi')}
              >
                <Crop size={13} style={{ verticalAlign: '-2px', marginRight: 4 }} aria-hidden="true" />
                Focus ROI
              </button>
            </div>

            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
              <button
                type="button"
                className="btn btn-sm"
                aria-pressed={isMirrored}
                disabled={!imageBitmap || analyzing}
                onClick={onToggleMirror}
                title="Flip horizontally if video/photo looks mirrored"
              >
                <FlipHorizontal2 size={13} aria-hidden="true" />
                <span>{isMirrored ? 'Unflip' : 'Mirror Flip'}</span>
              </button>

              <button
                type="button"
                className="btn btn-sm btn-primary"
                disabled={!imageBitmap || !modelReady || analyzing}
                onClick={() => onRunAnalysis()}
              >
                <RefreshCw size={13} aria-hidden="true" />
                <span>{analyzing ? 'Analyzing…' : mode === 'roi' ? 'Analyze ROI' : 'Re-analyze'}</span>
              </button>
            </div>
          </div>
          <p style={{ margin: '0.55rem 0 0', fontSize: '0.79rem', color: 'var(--text-secondary)' }}>
            {MODE_DESCRIPTIONS[mode]}
          </p>
        </div>
      </details>

      {/* Secondary control 2: Inspect edges & sign zoom */}
      <details
        className="disclosure embedded"
        onToggle={(e) => setEdgesOpen((e.currentTarget as HTMLDetailsElement).open)}
      >
        <summary>
          <span className="summary-title">Inspect edges &amp; sign zoom</span>
          <span className="summary-preview">8 zoomed corners &amp; edges</span>
        </summary>
        <div className="disclosure-body">
          <p style={{ margin: '0.35rem 0 0.55rem', fontSize: '0.79rem', color: 'var(--text-secondary)' }}>
            Corners and edges hide small signs, license plates, and bollards. Click any tile to select it as Focus ROI, or send it straight to Qwen-VL for full-detail OCR.
          </p>
          {edgeCrops.length > 0 ? (
            <div className="edge-crops-grid">
              {edgeCrops.map((crop) => (
                <div key={crop.id} className="edge-crop-card">
                  <button
                    type="button"
                    className="edge-crop-img-btn"
                    onClick={() => {
                      onChangeMode('roi');
                      onChangeRoi(crop.roi);
                      onRunAnalysis('roi', crop.roi);
                    }}
                    title={`Focus StreetCLIP ROI on ${crop.label}`}
                  >
                    <img src={crop.dataUrl} alt={crop.label} className="edge-crop-img" />
                  </button>
                  <div className="edge-crop-foot">
                    <span className="edge-crop-label">{crop.label}</span>
                    <button
                      type="button"
                      className="btn btn-sm btn-ghost edge-vlm-btn"
                      disabled={vlmLoading}
                      onClick={() => {
                        onChangeMode('roi');
                        onChangeRoi(crop.roi);
                        onRunVlmOnRoi(vlmModelId, vlmGrounded, crop.roi);
                      }}
                      title={`Run Qwen-VL OCR on ${crop.label}`}
                    >
                      <ScanText size={12} aria-hidden="true" />
                      <span>VLM</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ fontSize: '0.79rem', color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <Grid2x2 size={14} aria-hidden="true" />
              <span>Upload or select a photo to inspect edge crops.</span>
            </div>
          )}
        </div>
      </details>

      {/* Secondary control 3: Camera & EXIF metadata */}
      <details className="disclosure embedded">
        <summary>
          <span className="summary-title">Camera &amp; EXIF metadata</span>
          <span className="summary-preview">{exifSummaryPreview}</span>
        </summary>
        <div className="disclosure-body">
          {exif ? (
            <>
              {exif.gps && (
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.25rem' }}>
                  <button type="button" className="btn btn-sm" onClick={onToggleRevealGps}>
                    {gpsRevealed ? <EyeOff size={13} aria-hidden="true" /> : <Eye size={13} aria-hidden="true" />}
                    <span>{gpsRevealed ? 'Hide GPS' : 'Reveal EXIF GPS'}</span>
                  </button>
                </div>
              )}
              <div className="exif-grid">
                {(exif.make || exif.model) && (
                  <div className="exif-kv">
                    <div className="exif-k">Camera</div>
                    <div className="exif-v">{[exif.make, exif.model].filter(Boolean).join(' ')}</div>
                  </div>
                )}
                {exif.dateTimeOriginal && (
                  <div className="exif-kv">
                    <div className="exif-k">Captured</div>
                    <div className="exif-v mono">{exif.dateTimeOriginal.slice(0, 19).replace('T', ' ')}</div>
                  </div>
                )}
                {(exif.focalLength35mm || exif.focalLengthMm) && (
                  <div className="exif-kv">
                    <div className="exif-k">Optics &amp; FOV</div>
                    <div className="exif-v mono">
                      {exif.focalLength35mm
                        ? `${exif.focalLength35mm}mm eq. (~${Math.round(exif.hfovDeg ?? 0)}° HFOV)`
                        : `${exif.focalLengthMm?.toFixed(1)}mm`}
                    </div>
                  </div>
                )}
                <div className="exif-kv">
                  <div className="exif-k">Embedded GPS</div>
                  <div className="exif-v mono">
                    {exif.gps ? (
                      gpsRevealed ? (
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                          <MapPin size={13} aria-hidden="true" />
                          {exif.gps.lat.toFixed(4)}, {exif.gps.lng.toFixed(4)}
                        </span>
                      ) : (
                        'Present (hidden for blind test)'
                      )
                    ) : (
                      'None in file'
                    )}
                  </div>
                </div>
                {exifErrorKm !== null && topCityCoords && (
                  <div className="exif-kv">
                    <div className="exif-k">Distance to #1 ({topCityCoords.name})</div>
                    <div className="exif-v mono" style={{ color: 'var(--accent)' }}>
                      {exifErrorKm.toFixed(1)} km
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <p style={{ margin: '0.35rem 0 0', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              No EXIF camera or GPS metadata found in this image (typical for web and social photos).
            </p>
          )}
        </div>
      </details>
    </section>
  );
}
