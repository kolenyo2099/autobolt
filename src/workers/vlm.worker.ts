import {
  AutoModelForImageTextToText,
  AutoProcessor,
  RawImage,
  TextStreamer,
  env,
} from '@huggingface/transformers';
import type { ActiveBackend, VlmModelId, VlmModelSpec, VlmProgress } from '../engine/types';

export const VLM_MODELS: Record<VlmModelId, VlmModelSpec> = {
  'qwen35-0.8b': {
    id: 'qwen35-0.8b',
    label: 'Qwen3.5 0.8B Q4F16',
    repoId: 'onnx-community/Qwen3.5-0.8B-ONNX-OPT',
    approxSizeLabel: '~615 MB',
    enableThinkingFlag: true,
    notes: 'Fastest browser VLM for sign OCR & visual clue checks',
  },
  'qwen3-vl-2b': {
    id: 'qwen3-vl-2b',
    label: 'Qwen3-VL 2B Q4F16',
    repoId: 'onnx-community/Qwen3-VL-2B-Instruct-ONNX',
    approxSizeLabel: '~1.31 GB',
    enableThinkingFlag: false,
    notes: 'Balanced instruction-tuned 2B VLM for regional clues',
  },
  'qwen35-2b': {
    id: 'qwen35-2b',
    label: 'Qwen3.5 2B Q4F16',
    repoId: 'onnx-community/Qwen3.5-2B-ONNX-OPT',
    approxSizeLabel: '~1.51 GB',
    enableThinkingFlag: true,
    notes: 'Highest visual & OCR accuracy within browser memory limits',
  },
};

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
}

let loadedModelId: VlmModelId | null = null;
let loadedProcessor: any = null;
let loadedModel: any = null;
let activeBackend: ActiveBackend = 'unknown';
let activeDtype: 'q4f16' | 'q4' = 'q4f16';

function emitProgress(progress: VlmProgress) {
  self.postMessage({ type: 'progress', progress });
}

async function detectDeviceAndDtype(): Promise<{
  device: 'webgpu' | 'wasm';
  dtype: 'q4f16' | 'q4';
}> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter?: () => Promise<any> } }).gpu;
  if (gpu && typeof gpu.requestAdapter === 'function') {
    try {
      const adapter = await gpu.requestAdapter();
      if (adapter) {
        const hasF16 = adapter.features?.has?.('shader-f16') ?? false;
        return {
          device: 'webgpu',
          dtype: hasF16 ? 'q4f16' : 'q4',
        };
      }
    } catch {
      // Fall through to WASM
    }
  }
  return { device: 'wasm', dtype: 'q4' };
}

async function ensureModelLoaded(modelId: VlmModelId): Promise<void> {
  if (loadedModelId === modelId && loadedProcessor && loadedModel) {
    return;
  }

  if (loadedModel) {
    try {
      await loadedModel.dispose?.();
    } catch {
      // Ignore cleanup failures before switching models
    }
    loadedModel = null;
    loadedProcessor = null;
    loadedModelId = null;
  }

  const spec = VLM_MODELS[modelId];
  const { device, dtype } = await detectDeviceAndDtype();
  activeBackend = device;
  activeDtype = dtype;

  emitProgress({
    stage: 'loading',
    message: `Loading ${spec.label} processor (${device.toUpperCase()} · ${dtype.toUpperCase()})…`,
    loadedBytes: 0,
    totalBytes: 0,
  });

  const fileProgress = new Map<string, { loaded: number; total: number }>();

  const progressCallback = (info: any) => {
    if (!info) return;
    if (info.status === 'progress' || info.status === 'done' || info.status === 'initiate') {
      const key = String(info.file ?? info.name ?? 'weights');
      const prev = fileProgress.get(key) ?? { loaded: 0, total: 0 };
      const total = typeof info.total === 'number' && info.total > 0 ? info.total : prev.total;
      const loaded =
        info.status === 'done'
          ? total || prev.loaded
          : typeof info.loaded === 'number'
            ? info.loaded
            : prev.loaded;
      fileProgress.set(key, { loaded, total });

      let sumLoaded = 0;
      let sumTotal = 0;
      for (const entry of fileProgress.values()) {
        sumLoaded += entry.loaded;
        sumTotal += entry.total > 0 ? entry.total : entry.loaded;
      }
      const percent =
        sumTotal > 0 ? Math.min(100, Math.round((sumLoaded / sumTotal) * 100)) : undefined;

      emitProgress({
        stage: 'downloading',
        message: `Fetching ${spec.label} (${key.split('/').pop() ?? key})…`,
        file: key,
        loadedBytes: sumLoaded,
        totalBytes: sumTotal,
        percent,
      });
    }
  };

  loadedProcessor = await AutoProcessor.from_pretrained(spec.repoId, {
    progress_callback: progressCallback,
  });

  emitProgress({
    stage: 'loading',
    message: `Compiling ${spec.label} ONNX sessions on ${device.toUpperCase()}…`,
    loadedBytes: 1,
    totalBytes: 1,
    percent: 100,
  });

  loadedModel = await AutoModelForImageTextToText.from_pretrained(spec.repoId, {
    device,
    dtype: {
      embed_tokens: dtype,
      vision_encoder: dtype,
      decoder_model_merged: dtype,
    },
    progress_callback: progressCallback,
  });

  loadedModelId = modelId;
}

function buildGeoPrompt(contextLeadSummary?: string, isRoiCrop = false): string {
  if (isRoiCrop) {
    const bgBlock = contextLeadSummary
      ? `Background context from full photo (optional):\n${contextLeadSummary}\n\n`
      : '';
    return (
      `${bgBlock}` +
      `This image is a ZOOMED-IN CROP (ROI) from a larger street photo. Answer concisely in 4 short single-sentence bullets (under 120 words total):\n` +
      `1. **Crop Subject**: Briefly state what object or scene detail is shown in this crop.\n` +
      `2. **Verbatim Text & Codes**: Transcribe any legible words, scripts, issuing authority, or phone/area codes (or "None legible").\n` +
      `3. **Clues**: Note plate colors/shape, road markings, poles, signage style, or mirror-flip cues.\n` +
      `4. **Assessment**: Country/region/city clue implied by this specific crop.`
    );
  }

  const leadBlock = contextLeadSummary
    ? `AutoBolt candidate hypotheses to verify or challenge:\n${contextLeadSummary}\n\n`
    : '';
  return (
    `${leadBlock}` +
    `Answer concisely in 4 short single-sentence bullets (under 140 words total; write "illegible" if unclear):\n` +
    `1. **Capture**: Platform (street/building/vehicle) and whether text/traffic appears horizontally mirrored.\n` +
    `2. **Text & Codes**: Quote exact legible signs, scripts, issuing authorities, or phone/area codes.\n` +
    `3. **Road & Scene**: Driving side, plate style, utility poles/bollards, architecture, and climate.\n` +
    `4. **Assessment**: Best city/region/country call and whether visual evidence confirms or challenges the leads.`
  );
}

self.onmessage = async (event: MessageEvent) => {
  const { id, type } = event.data ?? {};
  try {
    if (type === 'clearCache') {
      if (typeof caches !== 'undefined') {
        await caches.delete('transformers-cache');
      }
      self.postMessage({ type: 'response', id, ok: true, result: { cleared: true } });
      return;
    }

    if (type === 'analyze') {
      const modelId = event.data.modelId as VlmModelId;
      const imageData = event.data.imageData as Uint8ClampedArray;
      const width = event.data.width as number;
      const height = event.data.height as number;
      const contextLeadSummary = event.data.contextLeadSummary as string | undefined;
      const isRoiCrop = Boolean(event.data.isRoiCrop);

      await ensureModelLoaded(modelId);

      const spec = VLM_MODELS[modelId];
      emitProgress({
        stage: 'generating',
        message: `Running ${spec.label} on ${activeBackend.toUpperCase()}…`,
        loadedBytes: 1,
        totalBytes: 1,
        percent: 100,
      });

      const rawImage = new RawImage(imageData, width, height, 3);
      const promptText = buildGeoPrompt(contextLeadSummary, isRoiCrop);
      const messages = [
        {
          role: 'user',
          content: [
            { type: 'image' },
            { type: 'text', text: promptText },
          ],
        },
      ];

      const formattedText = loadedProcessor.apply_chat_template(messages, {
        add_generation_prompt: true,
        ...(spec.enableThinkingFlag ? { enable_thinking: false } : {}),
      });

      // Qwen2VLProcessor / Qwen3VLProcessor signature expects (text, images)
      const inputs = await loadedProcessor(formattedText, rawImage);

      const t0 = performance.now();
      let streamedText = '';
      let tokenCount = 0;

      const streamer = new TextStreamer(loadedProcessor.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (chunk: string) => {
          streamedText += chunk;
          tokenCount += 1;
          self.postMessage({
            type: 'token',
            id,
            chunk,
            text: streamedText,
            tokenCount,
          });
        },
      });

      const outputs = await loadedModel.generate({
        ...inputs,
        max_new_tokens: 512,
        do_sample: false,
        streamer,
      });

      const latencyMs = performance.now() - t0;

      let finalText = streamedText.trim();
      if (!finalText && outputs) {
        const promptLen = inputs.input_ids?.dims?.[1] ?? 0;
        const sliced = promptLen > 0 ? outputs.slice(null, [promptLen, null]) : outputs;
        const decoded = loadedProcessor.batch_decode(sliced, { skip_special_tokens: true });
        finalText = String(decoded?.[0] ?? '').trim();
      }

      const elapsedSec = Math.max(0.05, latencyMs / 1000);
      emitProgress({
        stage: 'ready',
        message: `Completed ${spec.label}`,
        loadedBytes: 1,
        totalBytes: 1,
        percent: 100,
      });

      self.postMessage({
        type: 'response',
        id,
        ok: true,
        result: {
          modelId,
          modelLabel: spec.label,
          backend: activeBackend,
          dtype: activeDtype,
          text: finalText,
          latencyMs,
          tokenCount,
          tokensPerSec: tokenCount / elapsedSec,
          groundedWithLeads: Boolean(contextLeadSummary),
          isRoiCrop,
          cropWidth: width,
          cropHeight: height,
        },
      });
      return;
    }

    throw new Error(`Unknown VLM worker message type: ${String(type)}`);
  } catch (err) {
    self.postMessage({
      type: 'response',
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
};
