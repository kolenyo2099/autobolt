import { useCallback, useEffect, useRef, useState } from 'react';
import { HeaderBar } from './components/HeaderBar';
import { ImageWorkbench } from './components/ImageWorkbench';
import { LeadSummary } from './components/LeadSummary';
import { WorldMap } from './components/WorldMap';
import { LeadDisclosures } from './components/LeadDisclosures';
import { ModelDialog } from './components/ModelDialog';
import { StreetClipClient } from './engine/client';
import { extractImageExif } from './engine/exif';
import {
  preprocessCenterCrop,
  preprocessMultiCrop,
  preprocessRoiCrop,
} from './engine/preprocess';
import type {
  ActiveBackend,
  AnalysisMode,
  AnalysisResult,
  BackendMode,
  CustomProbeResult,
  EngineSource,
  ExifMetadata,
  LoadProgress,
  ModelBenchmark,
  RoiRect,
  SamplePhoto,
} from './engine/types';
import { buildMarkdownBrief, downloadJsonBrief } from './utils/exportBrief';

function formatBytes(bytes: number): string {
  if (!bytes) return '0 MB';
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function buildCrops(
  bitmap: ImageBitmap,
  mode: AnalysisMode,
  roi: RoiRect,
  size: number,
): Float32Array[] {
  if (mode === 'multicrop') return preprocessMultiCrop(bitmap, size);
  if (mode === 'roi') return [preprocessRoiCrop(bitmap, roi, size)];
  return [preprocessCenterCrop(bitmap, size)];
}

export function App() {
  const clientRef = useRef<StreetClipClient | null>(null);
  const currentObjectUrlRef = useRef<string | null>(null);

  const [engineSource, setEngineSource] = useState<EngineSource>('local-q4');
  const [backendMode, setBackendMode] = useState<BackendMode>('auto');
  const [activeBackend, setActiveBackend] = useState<ActiveBackend>('unknown');

  const [benchmark, setBenchmark] = useState<ModelBenchmark | null>(null);
  const [samples, setSamples] = useState<SamplePhoto[]>([]);
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);

  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'analyzing' | 'error'>('loading');
  const [loadProgress, setLoadProgress] = useState<LoadProgress | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string>('');
  const [imageBitmap, setImageBitmap] = useState<ImageBitmap | null>(null);
  const [exif, setExif] = useState<ExifMetadata | null>(null);
  const [gpsRevealed, setGpsRevealed] = useState(false);

  const [mode, setMode] = useState<AnalysisMode>('center');
  const [roi, setRoi] = useState<RoiRect>({ x: 0.25, y: 0.25, w: 0.5, h: 0.5 });

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedCountryCode, setSelectedCountryCode] = useState<string | null>(null);
  const [selectedCityName, setSelectedCityName] = useState<string | null>(null);
  const [focusTarget, setFocusTarget] = useState<{
    points: Array<{ lat: number; lng: number }>;
    seq: number;
  } | null>(null);

  const [customProbeResult, setCustomProbeResult] = useState<CustomProbeResult | null>(null);
  const [probeLoading, setProbeLoading] = useState(false);
  const [probeProgressMsg, setProbeProgressMsg] = useState<string | null>(null);
  const [probeProgressRatio, setProbeProgressRatio] = useState<number | null>(null);
  const [copiedBrief, setCopiedBrief] = useState(false);

  useEffect(() => {
    fetch('/models/model_benchmark.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setBenchmark(data as ModelBenchmark);
      })
      .catch(() => {});

    fetch('/samples/manifest.json')
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => {
        if (Array.isArray(data)) setSamples(data as SamplePhoto[]);
      })
      .catch(() => {});
  }, []);

  const initWorkerEngine = useCallback(async (source: EngineSource, backend: BackendMode) => {
    const client = clientRef.current;
    if (!client) return false;
    setModelStatus('loading');
    setErrorMsg(null);
    setLoadProgress({
      stage: 'vision',
      message: 'Initializing inference runtime…',
      loadedBytes: 0,
      totalBytes: 0,
    });
    try {
      const res = await client.init(source, backend);
      setActiveBackend(res.backend);
      setModelStatus('ready');
      return true;
    } catch (err) {
      setModelStatus('error');
      setErrorMsg(err instanceof Error ? err.message : String(err));
      return false;
    }
  }, []);

  useEffect(() => {
    const client = new StreetClipClient();
    clientRef.current = client;

    const unsub = client.onProgress((prog) => {
      if (prog.stage === 'text') {
        setProbeProgressMsg(prog.message);
        setProbeProgressRatio(prog.totalBytes > 0 ? prog.loadedBytes / prog.totalBytes : null);
      } else {
        setLoadProgress(prog);
      }
    });

    initWorkerEngine('local-q4', 'auto');

    return () => {
      unsub();
      client.dispose();
      clientRef.current = null;
    };
  }, [initWorkerEngine]);

  const runAnalysis = useCallback(
    async (
      bitmapOverride?: ImageBitmap,
      modeOverride?: AnalysisMode,
      roiOverride?: RoiRect,
      sourceOverride?: EngineSource,
    ) => {
      const client = clientRef.current;
      const bmp = bitmapOverride ?? imageBitmap;
      const activeMode = modeOverride ?? mode;
      const activeRoi = roiOverride ?? roi;
      const activeSource = sourceOverride ?? engineSource;
      if (!client || !bmp) return;

      setModelStatus('analyzing');
      setErrorMsg(null);
      try {
        const needs336 = activeSource !== 'osv5m-q4';
        const needs224 = activeSource === 'osv5m-q4' || activeSource === 'dual-q4';

        const crops336 = needs336 ? buildCrops(bmp, activeMode, activeRoi, 336) : undefined;
        const crops224 = needs224 ? buildCrops(bmp, activeMode, activeRoi, 224) : undefined;

        const analysis = await client.analyze({
          crops336,
          crops224,
          mode: activeMode,
        });
        setResult(analysis);
        setActiveBackend(analysis.backend);
        setSelectedCountryCode(null);
        setSelectedCityName(analysis.citiesHierarchical[0]?.name ?? null);
        setCustomProbeResult(null);

        const leadPoints: Array<{ lat: number; lng: number }> = [];
        if (analysis.geocell) {
          leadPoints.push({ lat: analysis.geocell.lat, lng: analysis.geocell.lng });
        }
        if (analysis.citiesHierarchical[0]) {
          leadPoints.push({
            lat: analysis.citiesHierarchical[0].lat,
            lng: analysis.citiesHierarchical[0].lng,
          });
        }
        if (leadPoints.length > 0) {
          setFocusTarget({ points: leadPoints, seq: Date.now() });
        }

        setModelStatus('ready');
      } catch (err) {
        setModelStatus('error');
        setErrorMsg(err instanceof Error ? err.message : String(err));
      }
    },
    [imageBitmap, mode, roi, engineSource],
  );

  const loadBlobImage = useCallback(
    async (blob: Blob, name: string, sampleId: string | null) => {
      if (currentObjectUrlRef.current) {
        URL.revokeObjectURL(currentObjectUrlRef.current);
      }
      const url = URL.createObjectURL(blob);
      currentObjectUrlRef.current = url;

      const [bitmap, parsedExif] = await Promise.all([
        createImageBitmap(blob),
        extractImageExif(blob),
      ]);

      setImageUrl(url);
      setImageName(name);
      setActiveSampleId(sampleId);
      setImageBitmap((prev) => {
        prev?.close();
        return bitmap;
      });
      setExif(parsedExif);
      setGpsRevealed(false);

      if (modelStatus === 'ready') {
        await runAnalysis(bitmap, mode, roi, engineSource);
      }
    },
    [modelStatus, mode, roi, engineSource, runAnalysis],
  );

  const handleSelectFile = useCallback(
    (file: File) => {
      loadBlobImage(file, file.name, null);
    },
    [loadBlobImage],
  );

  const handleSelectSample = useCallback(
    async (sample: SamplePhoto) => {
      const resp = await fetch(sample.url);
      if (!resp.ok) return;
      const blob = await resp.blob();
      await loadBlobImage(blob, sample.title, sample.id);
    },
    [loadBlobImage],
  );

  const didAutoLoadRef = useRef(false);
  useEffect(() => {
    if (!didAutoLoadRef.current && samples.length > 0 && modelStatus === 'ready' && !imageBitmap) {
      didAutoLoadRef.current = true;
      handleSelectSample(samples[0]);
    }
  }, [samples, modelStatus, imageBitmap, handleSelectSample]);

  useEffect(() => {
    if (modelStatus === 'ready' && imageBitmap && !result) {
      runAnalysis(imageBitmap, mode, roi, engineSource);
    }
  }, [modelStatus, imageBitmap, result, mode, roi, engineSource, runAnalysis]);

  const handleChangeEngine = async (source: EngineSource, backend: BackendMode) => {
    setEngineSource(source);
    setBackendMode(backend);
    const ok = await initWorkerEngine(source, backend);
    if (ok && imageBitmap) {
      await runAnalysis(imageBitmap, mode, roi, source);
    }
  };

  const handleClearCache = async () => {
    await clientRef.current?.clearCache();
    setLoadProgress({
      stage: 'ready',
      message: 'Browser model cache cleared.',
      loadedBytes: 1,
      totalBytes: 1,
    });
  };

  const handleRunCustomProbe = async (labels: string[], useStreetTemplate: boolean) => {
    if (!clientRef.current || !result) return;
    setProbeLoading(true);
    setProbeProgressMsg('Preparing Text Q4 encoder…');
    setProbeProgressRatio(null);
    setErrorMsg(null);
    try {
      const probeRes = await clientRef.current.customProbe(
        result.imageEmbed,
        labels,
        useStreetTemplate,
      );
      setCustomProbeResult(probeRes);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setProbeLoading(false);
      setProbeProgressMsg(null);
      setProbeProgressRatio(null);
    }
  };

  const handleCopyBrief = async () => {
    if (!result) return;
    const md = buildMarkdownBrief(result, imageName, exif, gpsRevealed);
    await navigator.clipboard.writeText(md);
    setCopiedBrief(true);
    setTimeout(() => setCopiedBrief(false), 2000);
  };

  const selectedCity =
    result?.citiesHierarchical.find((c) => c.name === selectedCityName) ??
    result?.citiesGlobal.find((c) => c.name === selectedCityName) ??
    result?.citiesHierarchical[0] ??
    null;

  const modelSizeLabel = benchmark
    ? `${benchmark.sizesMB.local_vision_q4.toFixed(0)} MB`
    : '160 MB';

  const osv5mSizeLabel = benchmark?.sizesMB.local_osv5m_q4
    ? `${benchmark.sizesMB.local_osv5m_q4.toFixed(0)} MB`
    : '~172 MB';

  const geocellsSizeLabel = benchmark?.sizesMB.local_geocells_f16
    ? `${benchmark.sizesMB.local_geocells_f16.toFixed(1)} MB`
    : '7.3 MB';

  const progressPct =
    loadProgress && loadProgress.totalBytes > 0
      ? Math.min(100, Math.round((loadProgress.loadedBytes / loadProgress.totalBytes) * 100))
      : undefined;

  return (
    <div className="app-shell">
      <HeaderBar
        backend={activeBackend}
        backendMode={backendMode}
        engineSource={engineSource}
        lastResult={result}
        status={modelStatus}
        copiedBrief={copiedBrief}
        onChangeEngine={handleChangeEngine}
        onCopyBrief={handleCopyBrief}
        onDownloadJson={() => result && downloadJsonBrief(result, imageName)}
      />

      {(modelStatus === 'loading' || errorMsg) && (
        <div className="loader-banner" role="status" aria-live="polite">
          <div className="loader-banner-row">
            <span>
              {errorMsg ? (
                <strong style={{ color: 'var(--danger)' }}>Error: {errorMsg}</strong>
              ) : (
                loadProgress?.message ?? 'Loading ONNX model…'
              )}
            </span>
            {loadProgress && loadProgress.totalBytes > 0 && !errorMsg && (
              <span className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                {formatBytes(loadProgress.loadedBytes)} / {formatBytes(loadProgress.totalBytes)} (
                {progressPct}%)
              </span>
            )}
          </div>
          {modelStatus === 'loading' && !errorMsg && (
            <progress
              className="scl-progress"
              aria-label="Model loading progress"
              max={100}
              value={progressPct}
            />
          )}
        </div>
      )}

      <main className="workspace-grid">
        <ImageWorkbench
          imageUrl={imageUrl}
          imageName={imageName}
          imageBitmap={imageBitmap}
          samples={samples}
          activeSampleId={activeSampleId}
          mode={mode}
          roi={roi}
          exif={exif}
          gpsRevealed={gpsRevealed}
          topCityCoords={
            result?.geocell
              ? {
                  name: result.geocell.nearestCity
                    ? `OSV-5M Est. near ${result.geocell.nearestCity}`
                    : 'OSV-5M Est.',
                  lat: result.geocell.lat,
                  lng: result.geocell.lng,
                }
              : result?.citiesHierarchical[0]
                ? {
                    name: result.citiesHierarchical[0].name,
                    lat: result.citiesHierarchical[0].lat,
                    lng: result.citiesHierarchical[0].lng,
                  }
                : null
          }
          analyzing={modelStatus === 'analyzing'}
          modelReady={modelStatus === 'ready'}
          onSelectFile={handleSelectFile}
          onSelectSample={handleSelectSample}
          onChangeMode={(nextMode) => {
            setMode(nextMode);
            if (imageBitmap && modelStatus === 'ready' && nextMode !== 'roi') {
              runAnalysis(imageBitmap, nextMode, roi, engineSource);
            }
          }}
          onChangeRoi={setRoi}
          onRunAnalysis={() => runAnalysis()}
          onToggleRevealGps={() => setGpsRevealed((v) => !v)}
        />

        <div className="right-column">
          <LeadSummary result={result} analyzing={modelStatus === 'analyzing'} />

          <WorldMap
            countries={result?.countries ?? []}
            cities={
              selectedCountryCode
                ? (result?.citiesGlobal ?? []).filter((c) => c.countryCode === selectedCountryCode)
                : result?.citiesHierarchical ?? []
            }
            geocell={result?.geocell}
            selectedCountryCode={selectedCountryCode}
            selectedCity={selectedCity}
            focusTarget={focusTarget}
            exifGps={gpsRevealed ? exif?.gps ?? null : null}
            onSelectCountry={(code) =>
              setSelectedCountryCode((prev) => (prev === code ? null : code))
            }
            onSelectCity={(city) => {
              setSelectedCityName(city.name);
              setSelectedCountryCode(city.countryCode);
            }}
          />

          <LeadDisclosures
            result={result}
            engineSource={engineSource}
            backendMode={backendMode}
            selectedCountryCode={selectedCountryCode}
            selectedCity={selectedCity}
            customProbeResult={customProbeResult}
            probeLoading={probeLoading}
            probeProgressMsg={probeProgressMsg}
            probeProgressRatio={probeProgressRatio}
            modelSizeLabel={modelSizeLabel}
            osv5mSizeLabel={osv5mSizeLabel}
            geocellsSizeLabel={geocellsSizeLabel}
            backend={activeBackend}
            onChangeEngine={handleChangeEngine}
            onSelectCountry={setSelectedCountryCode}
            onSelectCity={(city) => {
              setSelectedCityName(city.name);
              setFocusTarget({ points: [{ lat: city.lat, lng: city.lng }], seq: Date.now() });
            }}
            onFocusCoords={(lat, lng) =>
              setFocusTarget({ points: [{ lat, lng }], seq: Date.now() })
            }
            onRunCustomProbe={handleRunCustomProbe}
          />
        </div>
      </main>

      <ModelDialog
        benchmark={benchmark}
        engineSource={engineSource}
        backendMode={backendMode}
        onChangeEngine={handleChangeEngine}
        onClearCache={handleClearCache}
      />
    </div>
  );
}
