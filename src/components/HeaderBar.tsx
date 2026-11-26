import {
  Zap,
  Download,
  SlidersHorizontal,
  Copy,
  Check,
  MoreHorizontal,
  ImageDown,
} from 'lucide-react';
import type {
  ActiveBackend,
  AnalysisResult,
  BackendMode,
  EngineSource,
} from '../engine/types';

interface HeaderBarProps {
  backend: ActiveBackend;
  backendMode: BackendMode;
  engineSource: EngineSource;
  lastResult: AnalysisResult | null;
  status: 'loading' | 'ready' | 'analyzing' | 'error';
  copiedBrief: boolean;
  onChangeEngine: (source: EngineSource, backend: BackendMode) => void;
  onCopyBrief: () => void;
  onDownloadJson: () => void;
  onDownloadEvidencePng: () => void;
}

const ENGINE_OPTIONS: ReadonlyArray<{
  id: 'local-q4' | 'osv5m-q4' | 'dual-q4';
  label: string;
  hint: string;
}> = [
  {
    id: 'local-q4',
    label: 'StreetCLIP',
    hint: 'StreetCLIP Vision Q4 + visual clue & geocell probes',
  },
  {
    id: 'osv5m-q4',
    label: 'OSV-5M',
    hint: 'OSV-5M Baseline Q4 quadtree coordinate & territory model',
  },
  {
    id: 'dual-q4',
    label: 'Dual',
    hint: 'Combine StreetCLIP + OSV-5M locally for coordinates and clues',
  },
];

const ENGINE_SHORT_LABELS: Record<EngineSource, string> = {
  'local-q4': 'StreetCLIP',
  'osv5m-q4': 'OSV-5M',
  'dual-q4': 'Dual',
  'hf-q4f16': 'HF Q4F16',
  'hf-q4': 'HF Q4',
};

export function HeaderBar({
  backend,
  backendMode,
  engineSource,
  lastResult,
  status,
  copiedBrief,
  onChangeEngine,
  onCopyBrief,
  onDownloadJson,
  onDownloadEvidencePng,
}: HeaderBarProps) {
  const dotClass =
    status === 'error'
      ? 'status-dot error'
      : status === 'loading' || status === 'analyzing'
        ? 'status-dot loading'
        : 'status-dot ready';

  const isBusy = status === 'loading' || status === 'analyzing';
  const isHfSource = engineSource === 'hf-q4f16' || engineSource === 'hf-q4';
  const eng = ENGINE_SHORT_LABELS[engineSource];
  const statusText =
    status === 'error'
      ? 'Error'
      : status === 'loading'
        ? `Loading ${eng}…`
        : status === 'analyzing'
          ? 'Analyzing…'
          : isHfSource
            ? backend !== 'unknown'
              ? `Ready · ${eng} · ${backend.toUpperCase()}`
              : `Ready · ${eng}`
            : backend !== 'unknown'
              ? `Ready · ${backend.toUpperCase()}`
              : 'Ready';

  const closeMenu = () => {
    const el = document.getElementById('header-actions-menu') as
      | (HTMLElement & { hidePopover?: () => void })
      | null;
    el?.hidePopover?.();
  };

  return (
    <header className="app-header">
      <div className="brand-group">
        <div className="brand-mark" aria-hidden="true">
          <Zap size={17} />
        </div>
        <h1 className="brand-title">
          <span className="brand-org">Investigations Lab&apos;s</span> AutoBolt
        </h1>
      </div>

      <div className="header-controls">
        <div className="header-engine-switch" role="group" aria-label="Geolocation model mode">
          {ENGINE_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              className="header-engine-btn"
              aria-pressed={engineSource === opt.id}
              disabled={isBusy}
              title={opt.hint}
              onClick={() => {
                if (engineSource !== opt.id) {
                  onChangeEngine(opt.id, backendMode);
                }
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <button
          type="button"
          className="status-pill"
          aria-live="polite"
          title="Open model & quantization inspector"
          {...({ command: 'show-modal', commandfor: 'model-dialog' } as Record<string, unknown>)}
        >
          <span className={dotClass} aria-hidden="true" />
          <span>{statusText}</span>
        </button>

        <button
          type="button"
          className="btn btn-sm actions-trigger"
          popoverTarget="header-actions-menu"
          aria-haspopup="menu"
        >
          <MoreHorizontal size={15} aria-hidden="true" />
          <span>Actions</span>
        </button>

        <div id="header-actions-menu" popover="auto" className="actions-popover">
          <div className="actions-popover-list" role="menu" aria-label="Workspace actions">
            <button
              type="button"
              role="menuitem"
              className="menu-item-btn"
              onClick={closeMenu}
              {...({ command: 'show-modal', commandfor: 'model-dialog' } as Record<string, unknown>)}
            >
              <SlidersHorizontal size={15} aria-hidden="true" />
              <span>Model &amp; Quantization…</span>
            </button>

            <button
              type="button"
              role="menuitem"
              className="menu-item-btn"
              disabled={!lastResult}
              onClick={() => {
                onCopyBrief();
                closeMenu();
              }}
            >
              {copiedBrief ? <Check size={15} aria-hidden="true" /> : <Copy size={15} aria-hidden="true" />}
              <span>{copiedBrief ? 'Copied Lead Brief' : 'Copy Lead Brief (Markdown)'}</span>
            </button>

            <button
              type="button"
              role="menuitem"
              className="menu-item-btn"
              disabled={!lastResult}
              onClick={() => {
                onDownloadJson();
                closeMenu();
              }}
            >
              <Download size={15} aria-hidden="true" />
              <span>Download Report (JSON)</span>
            </button>

            <button
              type="button"
              role="menuitem"
              className="menu-item-btn"
              disabled={!lastResult}
              onClick={() => {
                onDownloadEvidencePng();
                closeMenu();
              }}
            >
              <ImageDown size={15} aria-hidden="true" />
              <span>Download Evidence Sheet (PNG)</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
