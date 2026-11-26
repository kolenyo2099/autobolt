import type { VlmModelId, VlmModelSpec, VlmProgress, VlmResult } from './types';

export const VLM_MODEL_SPECS: ReadonlyArray<VlmModelSpec> = [
  {
    id: 'qwen35-0.8b',
    label: 'Qwen3.5 0.8B Q4F16',
    repoId: 'onnx-community/Qwen3.5-0.8B-ONNX-OPT',
    approxSizeLabel: '~615 MB',
    enableThinkingFlag: true,
    notes: 'Fastest browser VLM for sign OCR & visual clue checks',
  },
  {
    id: 'qwen3-vl-2b',
    label: 'Qwen3-VL 2B Q4F16',
    repoId: 'onnx-community/Qwen3-VL-2B-Instruct-ONNX',
    approxSizeLabel: '~1.31 GB',
    enableThinkingFlag: false,
    notes: 'Balanced instruction-tuned 2B VLM for regional clues',
  },
  {
    id: 'qwen35-2b',
    label: 'Qwen3.5 2B Q4F16',
    repoId: 'onnx-community/Qwen3.5-2B-ONNX-OPT',
    approxSizeLabel: '~1.51 GB',
    enableThinkingFlag: true,
    notes: 'Highest visual & OCR accuracy within browser memory limits',
  },
];

type ProgressListener = (progress: VlmProgress) => void;
type TokenListener = (payload: { text: string; tokenCount: number }) => void;

export class VlmClient {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (val: any) => void; reject: (err: Error) => void }
  >();
  private progressListeners = new Set<ProgressListener>();
  private tokenListeners = new Set<TokenListener>();

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL('../workers/vlm.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (event: MessageEvent) => {
      const data = event.data;
      if (data?.type === 'progress') {
        for (const cb of this.progressListeners) cb(data.progress as VlmProgress);
        return;
      }
      if (data?.type === 'token') {
        for (const cb of this.tokenListeners) {
          cb({ text: String(data.text ?? ''), tokenCount: Number(data.tokenCount ?? 0) });
        }
        return;
      }
      if (data?.type === 'response') {
        const entry = this.pending.get(data.id);
        if (!entry) return;
        this.pending.delete(data.id);
        if (data.ok) entry.resolve(data.result);
        else entry.reject(new Error(data.error || 'VLM worker error'));
      }
    };
    worker.onerror = (err) => {
      for (const [, entry] of this.pending) {
        entry.reject(new Error(err.message || 'VLM worker crashed'));
      }
      this.pending.clear();
    };
    this.worker = worker;
    return worker;
  }

  onProgress(listener: ProgressListener): () => void {
    this.progressListeners.add(listener);
    return () => this.progressListeners.delete(listener);
  }

  onToken(listener: TokenListener): () => void {
    this.tokenListeners.add(listener);
    return () => this.tokenListeners.delete(listener);
  }

  private call<T>(payload: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    const worker = this.ensureWorker();
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      worker.postMessage({ ...payload, id }, transfer);
    });
  }

  analyze(params: {
    modelId: VlmModelId;
    imageData: Uint8ClampedArray;
    width: number;
    height: number;
    contextLeadSummary?: string;
    isRoiCrop?: boolean;
  }): Promise<VlmResult> {
    const copy = new Uint8ClampedArray(params.imageData);
    return this.call(
      {
        type: 'analyze',
        modelId: params.modelId,
        imageData: copy,
        width: params.width,
        height: params.height,
        contextLeadSummary: params.contextLeadSummary,
        isRoiCrop: Boolean(params.isRoiCrop),
      },
      [copy.buffer],
    );
  }

  cancel(): void {
    for (const [, entry] of this.pending) {
      entry.reject(new Error('VLM clue check cancelled.'));
    }
    this.pending.clear();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  async clearCache(): Promise<void> {
    if (typeof caches !== 'undefined') {
      await caches.delete('transformers-cache');
    }
    if (this.worker) {
      await this.call({ type: 'clearCache' });
    }
  }
}
