import { X, Trash2 } from 'lucide-react';
import type { BackendMode, EngineSource, ModelBenchmark } from '../engine/types';

interface ModelDialogProps {
  benchmark: ModelBenchmark | null;
  engineSource: EngineSource;
  backendMode: BackendMode;
  onChangeEngine: (source: EngineSource, backend: BackendMode) => void;
  onClearCache: () => void;
}

export function ModelDialog({
  benchmark,
  engineSource,
  backendMode,
  onChangeEngine,
  onClearCache,
}: ModelDialogProps) {
  const sizes = benchmark?.sizesMB;

  return (
    <dialog
      id="model-dialog"
      className="scl-dialog"
      closedby="any"
      aria-labelledby="model-dialog-title"
    >
      <div className="dialog-header">
        <h1 id="model-dialog-title" className="dialog-title">
          Investigations Lab&apos;s AutoBolt — Model &amp; Quantization Inspector
        </h1>
        <button
          type="button"
          className="btn btn-sm"
          {...({ command: 'close', commandfor: 'model-dialog' } as Record<string, unknown>)}
          aria-label="Close modal"
          autoFocus
        >
          <X size={15} aria-hidden="true" />
        </button>
      </div>

      <div className="dialog-body">
        <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
          Run <strong>StreetCLIP Vision Q4</strong> ({sizes ? `${sizes.local_vision_q4} MB` : '160 MB'}) for zero-shot countries, visual clues, and custom text probes;{' '}
          <strong>OSV-5M Baseline Q4</strong> ({sizes?.local_osv5m_q4 ? `${sizes.local_osv5m_q4} MB` : '~172 MB'}) for 11,399-cell + sub-cell regression across 225 territories; or{' '}
          <strong>Dual Mode</strong> to combine both locally in WebGPU/WASM.
        </p>

        {benchmark && (
          <table className="bench-table">
            <thead>
              <tr>
                <th>Variant</th>
                <th>Download Size</th>
                <th>Reduction vs FP32</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>HF Upstream StreetCLIP FP32 (<code>model.onnx</code>)</td>
                <td className="mono">{sizes!.hf_fp32.toFixed(1)} MB</td>
                <td className="mono">1.0×</td>
                <td>Monolithic Vision + Text</td>
              </tr>
              <tr>
                <td>HF Upstream StreetCLIP Q4F16 (<code>model_q4f16.onnx</code>)</td>
                <td className="mono">{sizes!.hf_q4f16.toFixed(1)} MB</td>
                <td className="mono">{(sizes!.hf_fp32 / sizes!.hf_q4f16).toFixed(1)}×</td>
                <td>Smallest upstream HF file</td>
              </tr>
              <tr className="highlight-row">
                <td>AutoBolt StreetCLIP Vision Q4 + Leads Pack</td>
                <td className="mono">
                  {(sizes!.local_vision_q4 + sizes!.local_embeddings_f16).toFixed(1)} MB
                </td>
                <td className="mono">
                  {(sizes!.hf_fp32 / (sizes!.local_vision_q4 + sizes!.local_embeddings_f16)).toFixed(1)}×
                </td>
                <td>
                  Split INT4 block-{benchmark.visionQuantization.blockSize} · cos sim{' '}
                  {benchmark.visionQuantization.minCosineSimilarityVsFp32.toFixed(4)}
                </td>
              </tr>
              <tr className="highlight-row">
                <td>OSV-5M 5,000-Cell Coordinate Head</td>
                <td className="mono">
                  {sizes!.local_geocells_f16 ? `${sizes!.local_geocells_f16.toFixed(2)} MB` : '7.32 MB'}
                </td>
                <td className="mono">2× vs F32</td>
                <td>
                  FP16 quadtree head (<code>Daggerabuse/geoguessr-streetclip-geocells</code>, 225 territories)
                </td>
              </tr>
              {sizes!.local_osv5m_q4 && (
                <tr className="highlight-row">
                  <td>OSV-5M Baseline Q4 (224×224)</td>
                  <td className="mono">{sizes!.local_osv5m_q4.toFixed(1)} MB</td>
                  <td className="mono">{(1296.33 / sizes!.local_osv5m_q4).toFixed(1)}× vs PyTorch</td>
                  <td>
                    11,399 quadtree cells + sub-cell regression (
                    {benchmark.osv5mQuantization
                      ? `block-${benchmark.osv5mQuantization.blockSize}, max shift ${benchmark.osv5mQuantization.maxShiftKmVsFp32} km`
                      : 'INT4'}
                    )
                  </td>
                </tr>
              )}
              <tr className="highlight-row">
                <td>On-Demand Custom Probe Text Q4</td>
                <td className="mono">{sizes!.local_text_q4.toFixed(1)} MB</td>
                <td className="mono">Lazy-loaded</td>
                <td>
                  INT4 MatMul + UINT8 token table · cos sim{' '}
                  {benchmark.textQuantization.minCosineSimilarityVsFp32.toFixed(4)}
                </td>
              </tr>
              <tr className="highlight-row">
                <td>Deep VLM Clue Check (Qwen3.5 0.8B / 2B Q4F16)</td>
                <td className="mono">~615 MB – 1.51 GB</td>
                <td className="mono">Lazy-loaded (HF CDN)</td>
                <td>
                  Split vision + embed + merged decoder Q4F16 on WebGPU for sign OCR &amp; clue verification
                </td>
              </tr>
            </tbody>
          </table>
        )}

        {benchmark && benchmark.visionQuantization.samples.length > 0 && (
          <div
            style={{
              padding: '0.7rem 0.85rem',
              borderRadius: 'var(--radius-md)',
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.8rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.35rem',
            }}
          >
            <div>
              <strong>StreetCLIP Q4 Accuracy vs FP32:</strong>{' '}
              {benchmark.visionQuantization.samples
                .map(
                  (s) =>
                    `${s.file} → ${s.topCountryQ4} / ${s.topCityQ4} (cos=${s.cosineSimVsFp32})`,
                )
                .join(' · ')}
            </div>
            {benchmark.osv5mQuantization && (
              <div>
                <strong>OSV-5M Q4 Accuracy:</strong>{' '}
                {benchmark.osv5mQuantization.samples
                  .map(
                    (s) =>
                      `${s.file} → (${s.estimatedLat}, ${s.estimatedLng}) · err ${s.errorKmVsGroundTruth} km (shift ${s.shiftKmVsFp32} km vs FP32)`,
                  )
                  .join(' · ')}
              </div>
            )}
          </div>
        )}

        <div className="settings-grid">
          <div className="field-group">
            <label htmlFor="engine-select">Model Source</label>
            <select
              id="engine-select"
              className="field-select"
              value={engineSource}
              onChange={(e) => onChangeEngine(e.target.value as EngineSource, backendMode)}
            >
              <option value="local-q4">
                StreetCLIP Vision Q4 ({sizes ? `${sizes.local_vision_q4} MB` : '160 MB'})
              </option>
              <option value="osv5m-q4">
                OSV-5M Baseline Q4 ({sizes?.local_osv5m_q4 ? `${sizes.local_osv5m_q4} MB` : '~172 MB'} · 11k cells)
              </option>
              <option value="dual-q4">
                Dual Mode: StreetCLIP + OSV-5M Baseline Q4
              </option>
              <option value="hf-q4f16">
                Remote HF onnx-community/StreetCLIP-ONNX (q4f16 · 298 MB)
              </option>
              <option value="hf-q4">
                Remote HF onnx-community/StreetCLIP-ONNX (q4 · 402 MB)
              </option>
            </select>
          </div>

          <div className="field-group">
            <label htmlFor="backend-select">Execution Provider</label>
            <select
              id="backend-select"
              className="field-select"
              value={backendMode}
              onChange={(e) => onChangeEngine(engineSource, e.target.value as BackendMode)}
            >
              <option value="auto">Auto (WebGPU → WASM fallback)</option>
              <option value="webgpu">WebGPU only</option>
              <option value="wasm">WASM (CPU SIMD / Threads)</option>
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.75rem' }}>
          <button type="button" className="btn btn-sm" onClick={onClearCache}>
            <Trash2 size={14} aria-hidden="true" />
            <span>Clear Browser Model Cache</span>
          </button>

          <form method="dialog">
            <button type="submit" className="btn btn-primary">
              Done
            </button>
          </form>
        </div>
      </div>
    </dialog>
  );
}
