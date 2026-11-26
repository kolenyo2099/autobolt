import type {
  ActiveBackend,
  AnalysisMode,
  AnalysisResult,
  BackendMode,
  CustomProbeResult,
  EngineSource,
  LoadProgress,
} from './types';

type ProgressListener = (progress: LoadProgress) => void;

export class StreetClipClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (val: any) => void; reject: (err: Error) => void }
  >();
  private progressListeners = new Set<ProgressListener>();

  constructor() {
    this.worker = new Worker(
      new URL('../workers/streetclip.worker.ts', import.meta.url),
      { type: 'module' },
    );
    this.worker.onmessage = (event: MessageEvent) => {
      const data = event.data;
      if (data?.type === 'progress') {
        for (const cb of this.progressListeners) {
          cb(data.progress as LoadProgress);
        }
        return;
      }
      if (data?.type === 'response') {
        const entry = this.pending.get(data.id);
        if (!entry) return;
        this.pending.delete(data.id);
        if (data.ok) entry.resolve(data.result);
        else entry.reject(new Error(data.error || 'Worker error'));
      }
    };
    this.worker.onerror = (err) => {
      for (const [, entry] of this.pending) {
        entry.reject(new Error(err.message || 'Worker crashed'));
      }
      this.pending.clear();
    };
  }

  onProgress(listener: ProgressListener): () => void {
    this.progressListeners.add(listener);
    return () => this.progressListeners.delete(listener);
  }

  private call<T>(payload: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id }, transfer);
    });
  }

  init(engineSource: EngineSource, backendMode: BackendMode): Promise<{ backend: ActiveBackend }> {
    return this.call({ type: 'init', engineSource, backendMode });
  }

  analyze(params: {
    crops336?: Float32Array[];
    crops224?: Float32Array[];
    mode: AnalysisMode;
  }): Promise<AnalysisResult> {
    const buffers: Transferable[] = [];
    if (params.crops336) {
      for (const c of params.crops336) buffers.push(c.buffer);
    }
    if (params.crops224) {
      for (const c of params.crops224) buffers.push(c.buffer);
    }
    return this.call(
      {
        type: 'analyze',
        crops336: params.crops336,
        crops224: params.crops224,
        mode: params.mode,
      },
      buffers,
    );
  }

  customProbe(
    imageEmbed: Float32Array,
    labels: string[],
    useStreetTemplate: boolean,
  ): Promise<CustomProbeResult> {
    const embedCopy = new Float32Array(imageEmbed);
    return this.call(
      { type: 'customProbe', imageEmbed: embedCopy, labels, useStreetTemplate },
      [embedCopy.buffer],
    );
  }

  clearCache(): Promise<boolean> {
    return this.call({ type: 'clearCache' });
  }

  dispose(): void {
    this.worker.terminate();
  }
}
