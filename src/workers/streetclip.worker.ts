/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web/webgpu';
import {
  AutoTokenizer,
  CLIPModel,
  Tensor,
  env as hfEnv,
  type PreTrainedTokenizer,
} from '@huggingface/transformers';
import {
  decodeFloat16Array,
  l2Normalize,
  scoreImageEmbedding,
  scoreOsv5mOutputs,
  scoreWithTemperature,
} from '../engine/scoring';
import type {
  ActiveBackend,
  AnalysisMode,
  AnalysisResult,
  BackendMode,
  CustomProbeResult,
  EngineSource,
  LeadsCatalog,
  LoadProgress,
} from '../engine/types';

const CACHE_NAME = 'streetclip-lens-v1';

// Configure ONNX Runtime Web assets: point only `wasm` to public/ort/ so Vite
// uses the bundled JS module inside ort.webgpu.bundle.min.mjs instead of
// attempting a dynamic ES import() from /public.
ort.env.wasm.wasmPaths = {
  wasm: '/ort/ort-wasm-simd-threaded.asyncify.wasm',
};
ort.env.wasm.numThreads = 1;

interface InitRequest {
  type: 'init';
  id: number;
  engineSource: EngineSource;
  backendMode: BackendMode;
}

interface AnalyzeRequest {
  type: 'analyze';
  id: number;
  crops336?: Float32Array[];
  crops224?: Float32Array[];
  mode: AnalysisMode;
}

interface CustomProbeRequest {
  type: 'customProbe';
  id: number;
  imageEmbed: Float32Array;
  labels: string[];
  useStreetTemplate: boolean;
}

interface ClearCacheRequest {
  type: 'clearCache';
  id: number;
}

type WorkerRequest = InitRequest | AnalyzeRequest | CustomProbeRequest | ClearCacheRequest;

let visionSession: ort.InferenceSession | null = null;
let osv5mSession: ort.InferenceSession | null = null;
let textSession: ort.InferenceSession | null = null;
let tokenizer: PreTrainedTokenizer | null = null;
let catalog: LeadsCatalog | null = null;
let embeddingsF32: Float32Array | null = null;
let geocellHeadF32: Float32Array | null = null;
let geocellCoordsF32: Float32Array | null = null;
let osv5mQuadtreeF32: Float32Array | null = null;
let hfModel: any = null;
let activeEngineSource: EngineSource = 'local-q4';
let activeBackendMode: BackendMode = 'auto';
let activeBackend: ActiveBackend = 'unknown';

function postProgress(progress: LoadProgress) {
  self.postMessage({ type: 'progress', progress });
}

async function fetchCachedArrayBuffer(
  url: string,
  stage: LoadProgress['stage'],
  label: string,
): Promise<ArrayBuffer> {
  const cache = 'caches' in self ? await caches.open(CACHE_NAME) : null;
  if (cache) {
    const cached = await cache.match(url);
    if (cached) {
      const buf = await cached.arrayBuffer();
      postProgress({
        stage,
        message: `${label} (from browser cache)`,
        loadedBytes: buf.byteLength,
        totalBytes: buf.byteLength,
        cached: true,
      });
      return buf;
    }
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} fetching ${url}`);
  }

  const totalBytes = Number(response.headers.get('content-length') || 0);
  const reader = response.body?.getReader();

  if (!reader) {
    const buf = await response.arrayBuffer();
    if (cache) {
      await cache.put(
        url,
        new Response(buf.slice(0), {
          headers: { 'content-type': response.headers.get('content-type') || 'application/octet-stream' },
        }),
      );
    }
    return buf;
  }

  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  let lastEmit = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loadedBytes += value.byteLength;
      const now = performance.now();
      if (now - lastEmit > 110) {
        lastEmit = now;
        postProgress({
          stage,
          message: label,
          loadedBytes,
          totalBytes: totalBytes || loadedBytes,
          cached: false,
        });
      }
    }
  }

  const merged = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  postProgress({
    stage,
    message: label,
    loadedBytes,
    totalBytes: totalBytes || loadedBytes,
    cached: false,
  });

  if (cache) {
    await cache.put(
      url,
      new Response(merged.buffer.slice(0), {
        headers: { 'content-type': response.headers.get('content-type') || 'application/octet-stream' },
      }),
    );
  }

  return merged.buffer;
}

async function hasWebGpuAdapter(): Promise<boolean> {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown | null> } }).gpu;
    if (!gpu) return false;
    const adapter = await gpu.requestAdapter();
    return adapter != null;
  } catch {
    return false;
  }
}

async function createOrtSession(
  modelBytes: ArrayBuffer,
  mode: BackendMode,
): Promise<{ session: ort.InferenceSession; backend: ActiveBackend }> {
  const canWebGpu = mode !== 'wasm' && (await hasWebGpuAdapter());

  if (canWebGpu) {
    try {
      const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['webgpu'],
      });
      return { session, backend: 'webgpu' };
    } catch (err) {
      if (mode === 'webgpu') {
        throw err;
      }
      console.warn('[streetclip.worker] WebGPU session failed, falling back to WASM:', err);
    }
  }

  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ['wasm'],
  });
  return { session, backend: 'wasm' };
}

async function ensureCatalogOnly(): Promise<void> {
  if (catalog) return;
  const catResp = await fetch('/models/leads_catalog.json');
  if (!catResp.ok) {
    throw new Error(`Failed to fetch leads_catalog.json: HTTP ${catResp.status}`);
  }
  catalog = (await catResp.json()) as LeadsCatalog;
}

async function ensureLeadsPack(): Promise<void> {
  await ensureCatalogOnly();
  if (!embeddingsF32) {
    postProgress({
      stage: 'catalog',
      message: 'Loading precomputed geographic & visual leads pack…',
      loadedBytes: 0,
      totalBytes: 0,
    });
    const binBuf = await fetchCachedArrayBuffer(
      '/models/leads_embeddings.f16.bin',
      'catalog',
      'Loading leads embedding vectors',
    );
    embeddingsF32 = decodeFloat16Array(binBuf);
  }

  if (!geocellHeadF32 || !geocellCoordsF32) {
    try {
      const [headBuf, coordsBuf] = await Promise.all([
        fetchCachedArrayBuffer(
          '/models/geocells_head.f16.bin',
          'geocells',
          'Loading OSV-5M 5,000-cell coordinate head',
        ),
        fetchCachedArrayBuffer(
          '/models/geocells_coords.f32.bin',
          'geocells',
          'Loading OSV-5M cell centroids',
        ),
      ]);
      geocellHeadF32 = decodeFloat16Array(headBuf);
      geocellCoordsF32 = new Float32Array(coordsBuf);
    } catch (err) {
      console.warn('[streetclip.worker] Optional geocell head unavailable:', err);
    }
  }
}

async function ensureOsv5mQuadtree(): Promise<void> {
  await ensureCatalogOnly();
  if (!osv5mQuadtreeF32) {
    const qtBuf = await fetchCachedArrayBuffer(
      '/models/osv5m_quadtree.f32.bin',
      'osv5m',
      'Loading OSV-5M 11,399-cell quadtree coordinates',
    );
    osv5mQuadtreeF32 = new Float32Array(qtBuf);
  }
}

async function releaseSessions(): Promise<void> {
  if (visionSession) {
    await visionSession.release();
    visionSession = null;
  }
  if (osv5mSession) {
    await osv5mSession.release();
    osv5mSession = null;
  }
  if (textSession) {
    await textSession.release();
    textSession = null;
  }
  if (hfModel) {
    if (typeof hfModel.dispose === 'function') {
      await hfModel.dispose();
    }
    hfModel = null;
  }
}

async function initEngine(engineSource: EngineSource, backendMode: BackendMode): Promise<{ backend: ActiveBackend }> {
  await releaseSessions();
  activeEngineSource = engineSource;
  activeBackendMode = backendMode;

  if (engineSource === 'local-q4' || engineSource === 'dual-q4') {
    await ensureLeadsPack();
    const modelBytes = await fetchCachedArrayBuffer(
      '/models/vision_q4.onnx',
      'vision',
      'Loading quantized StreetCLIP Vision Q4 encoder',
    );
    postProgress({
      stage: 'vision',
      message: 'Compiling StreetCLIP Vision Q4 graph…',
      loadedBytes: modelBytes.byteLength,
      totalBytes: modelBytes.byteLength,
    });
    const created = await createOrtSession(modelBytes, backendMode);
    visionSession = created.session;
    activeBackend = created.backend;
  }

  if (engineSource === 'osv5m-q4' || engineSource === 'dual-q4') {
    await ensureOsv5mQuadtree();
    const osvBytes = await fetchCachedArrayBuffer(
      '/models/osv5m_q4.onnx',
      'osv5m',
      'Loading quantized OSV-5M Baseline Q4 model',
    );
    postProgress({
      stage: 'osv5m',
      message: 'Compiling OSV-5M Baseline Q4 graph…',
      loadedBytes: osvBytes.byteLength,
      totalBytes: osvBytes.byteLength,
    });
    const createdOsv = await createOrtSession(osvBytes, backendMode);
    osv5mSession = createdOsv.session;
    activeBackend = createdOsv.backend;
  }

  if (engineSource === 'hf-q4f16' || engineSource === 'hf-q4') {
    await ensureLeadsPack();
    hfEnv.allowRemoteModels = true;
    hfEnv.allowLocalModels = false;
    const dtype = engineSource === 'hf-q4f16' ? 'q4f16' : 'q4';
    const canWebGpu = backendMode !== 'wasm' && (await hasWebGpuAdapter());
    const device = canWebGpu ? 'webgpu' : 'wasm';

    postProgress({
      stage: 'hf',
      message: `Downloading onnx-community/StreetCLIP-ONNX (${dtype}) from Hugging Face…`,
      loadedBytes: 0,
      totalBytes: 0,
    });

    hfModel = await CLIPModel.from_pretrained('onnx-community/StreetCLIP-ONNX', {
      dtype,
      device,
      progress_callback: (info: any) => {
        if (info?.status === 'progress' && typeof info.loaded === 'number') {
          postProgress({
            stage: 'hf',
            message: `Downloading ${info.file ?? 'model'}…`,
            loadedBytes: info.loaded,
            totalBytes: info.total ?? info.loaded,
          });
        }
      },
    });
    activeBackend = device;
  }

  postProgress({
    stage: 'ready',
    message: `Ready (${activeBackend.toUpperCase()})`,
    loadedBytes: 1,
    totalBytes: 1,
  });

  return { backend: activeBackend };
}

async function ensureTokenizer(): Promise<PreTrainedTokenizer> {
  if (tokenizer) return tokenizer;
  hfEnv.allowLocalModels = true;
  hfEnv.allowRemoteModels = false;
  hfEnv.localModelPath = '/models/';
  tokenizer = await AutoTokenizer.from_pretrained('tokenizer', { local_files_only: true });
  return tokenizer;
}

async function ensureTextSession(): Promise<ort.InferenceSession> {
  if (textSession) return textSession;
  const textBytes = await fetchCachedArrayBuffer(
    '/models/text_q4.onnx',
    'text',
    'Loading on-demand Text Q4 encoder for custom probes',
  );
  postProgress({
    stage: 'text',
    message: 'Compiling Text Q4 inference graph…',
    loadedBytes: textBytes.byteLength,
    totalBytes: textBytes.byteLength,
  });
  const created = await createOrtSession(textBytes, activeBackendMode);
  textSession = created.session;
  return textSession;
}

async function runVisionCrops(crops: Float32Array[]): Promise<Float32Array> {
  const dim = catalog!.dim;
  const accum = new Float32Array(dim);

  if (activeEngineSource === 'local-q4' || activeEngineSource === 'dual-q4') {
    if (!visionSession) throw new Error('Vision session not initialized');
    for (const crop of crops) {
      const tensor = new ort.Tensor('float32', crop, [1, 3, 336, 336]);
      const outputs = await visionSession.run({ pixel_values: tensor });
      const embedTensor = outputs.image_embeds ?? outputs[Object.keys(outputs)[0]];
      const data = embedTensor.data as Float32Array;
      const normCrop = l2Normalize(data);
      for (let i = 0; i < dim; i++) {
        accum[i] += normCrop[i];
      }
      tensor.dispose?.();
      embedTensor.dispose?.();
    }
  } else {
    if (!hfModel) throw new Error('HF model not initialized');
    const tok = await ensureTokenizer();
    const dummyTokens = tok(['a photo'], { padding: true, truncation: true });

    for (const crop of crops) {
      const pv = new Tensor('float32', crop, [1, 3, 336, 336]);
      const outputs = await hfModel({
        pixel_values: pv,
        input_ids: dummyTokens.input_ids,
        attention_mask: dummyTokens.attention_mask,
      });
      const data = outputs.image_embeds.data as Float32Array;
      const normCrop = l2Normalize(data);
      for (let i = 0; i < dim; i++) {
        accum[i] += normCrop[i];
      }
    }
  }

  return l2Normalize(accum);
}

async function runOsv5mCrops(
  crops224: Float32Array[],
): Promise<{ classLogits: Float32Array; regOffsets: Float32Array }> {
  if (!osv5mSession || !osv5mQuadtreeF32) {
    throw new Error('OSV-5M session not initialized');
  }
  const nCells = Math.floor(osv5mQuadtreeF32.length / 6);
  const logitsAccum = new Float32Array(nCells);
  const regAccum = new Float32Array(nCells * 2);

  for (const crop of crops224) {
    const tensor = new ort.Tensor('float32', crop, [1, 3, 224, 224]);
    const outputs = await osv5mSession.run({ pixel_values: tensor });
    const logitsT = outputs.class_logits ?? outputs[Object.keys(outputs)[0]];
    const regT = outputs.reg_offsets ?? outputs[Object.keys(outputs)[1]];
    const lData = logitsT.data as Float32Array;
    const rData = regT.data as Float32Array;

    for (let i = 0; i < nCells; i++) logitsAccum[i] += lData[i];
    for (let i = 0; i < nCells * 2; i++) regAccum[i] += rData[i];

    tensor.dispose?.();
    logitsT.dispose?.();
    regT.dispose?.();
  }

  const inv = 1 / Math.max(1, crops224.length);
  for (let i = 0; i < nCells; i++) logitsAccum[i] *= inv;
  for (let i = 0; i < nCells * 2; i++) regAccum[i] *= inv;

  return { classLogits: logitsAccum, regOffsets: regAccum };
}

async function handleAnalyze(req: AnalyzeRequest): Promise<AnalysisResult> {
  if (!catalog) {
    throw new Error('Catalog not loaded');
  }
  const t0 = performance.now();

  if (activeEngineSource === 'osv5m-q4') {
    if (!req.crops224 || !osv5mQuadtreeF32) {
      throw new Error('Missing 224x224 crops or OSV-5M quadtree');
    }
    const { classLogits, regOffsets } = await runOsv5mCrops(req.crops224);
    const latencyMs = performance.now() - t0;
    return scoreOsv5mOutputs(classLogits, regOffsets, osv5mQuadtreeF32, catalog, {
      latencyMs,
      cropCount: req.crops224.length,
      mode: req.mode,
      backend: activeBackend,
    });
  }

  if (!embeddingsF32 || !req.crops336) {
    throw new Error('StreetCLIP embeddings or 336x336 crops missing');
  }

  if (activeEngineSource === 'dual-q4') {
    if (!req.crops224 || !osv5mQuadtreeF32) {
      throw new Error('Missing 224x224 crops for Dual mode');
    }
    const imageEmbed = await runVisionCrops(req.crops336);
    const { classLogits, regOffsets } = await runOsv5mCrops(req.crops224);
    const latencyMs = performance.now() - t0;
    const osvRes = scoreOsv5mOutputs(classLogits, regOffsets, osv5mQuadtreeF32, catalog, {
      latencyMs,
      cropCount: req.crops224.length,
      mode: req.mode,
      backend: activeBackend,
    });

    return scoreImageEmbedding(imageEmbed, catalog, embeddingsF32, {
      engineSource: activeEngineSource,
      latencyMs,
      cropCount: req.crops336.length,
      mode: req.mode,
      backend: activeBackend,
      geocellHead: geocellHeadF32,
      geocellCoords: geocellCoordsF32,
      osv5mOverride: osvRes.geocell,
    });
  }

  const imageEmbed = await runVisionCrops(req.crops336);
  const latencyMs = performance.now() - t0;

  return scoreImageEmbedding(imageEmbed, catalog, embeddingsF32, {
    engineSource: activeEngineSource,
    latencyMs,
    cropCount: req.crops336.length,
    mode: req.mode,
    backend: activeBackend,
    geocellHead: geocellHeadF32,
    geocellCoords: geocellCoordsF32,
  });
}

async function handleCustomProbe(req: CustomProbeRequest): Promise<CustomProbeResult> {
  if (!catalog) throw new Error('Catalog not initialized');
  if (req.imageEmbed.length === 0) {
    throw new Error('Custom Probe requires StreetCLIP or Dual Mode image embeddings.');
  }
  const t0 = performance.now();

  const tok = await ensureTokenizer();
  const sess = await ensureTextSession();

  const prompts = req.labels.map((label) =>
    req.useStreetTemplate ? `A Street View photo in ${label}.` : label,
  );

  const encoded = tok(prompts, {
    padding: true,
    truncation: true,
    max_length: 77,
    return_tensor: false,
  }) as { input_ids: number[][]; attention_mask: number[][] };

  const batch = prompts.length;
  const seqLen = 77;
  const eosId = BigInt(49407);
  const inputIdsArr = new BigInt64Array(batch * seqLen);
  const attnMaskArr = new BigInt64Array(batch * seqLen);

  for (let b = 0; b < batch; b++) {
    const idsRow = encoded.input_ids[b] ?? [];
    const maskRow = encoded.attention_mask[b] ?? [];
    for (let s = 0; s < seqLen; s++) {
      const idx = b * seqLen + s;
      if (s < idsRow.length) {
        inputIdsArr[idx] = BigInt(idsRow[s]);
        attnMaskArr[idx] = BigInt(maskRow[s] ?? 1);
      } else {
        inputIdsArr[idx] = eosId;
        attnMaskArr[idx] = 0n;
      }
    }
  }

  const inputIdsTensor = new ort.Tensor('int64', inputIdsArr, [batch, seqLen]);
  const attnMaskTensor = new ort.Tensor('int64', attnMaskArr, [batch, seqLen]);
  const outputs = await sess.run({
    input_ids: inputIdsTensor,
    attention_mask: attnMaskTensor,
  });
  const textEmbedsTensor = outputs.text_embeds ?? outputs[Object.keys(outputs)[0]];
  const textEmbeds = textEmbedsTensor.data as Float32Array;

  const dim = catalog.dim;
  const img = l2Normalize(req.imageEmbed);
  const rawScores = new Float32Array(batch);

  for (let b = 0; b < batch; b++) {
    const slice = l2Normalize(textEmbeds.subarray(b * dim, (b + 1) * dim));
    let dot = 0;
    for (let d = 0; d < dim; d++) {
      dot += img[d] * slice[d];
    }
    rawScores[b] = dot;
  }

  const probs = scoreWithTemperature(rawScores, catalog.logitScale);
  const items = req.labels
    .map((label, idx) => ({
      label,
      prompt: prompts[idx],
      score: rawScores[idx],
      prob: probs[idx],
      rank: 0,
    }))
    .sort((a, b) => b.prob - a.prob)
    .map((item, idx) => ({ ...item, rank: idx + 1 }));

  inputIdsTensor.dispose?.();
  attnMaskTensor.dispose?.();
  textEmbedsTensor.dispose?.();

  return {
    items,
    latencyMs: performance.now() - t0,
    backend: activeBackend,
  };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.type === 'init') {
      const result = await initEngine(msg.engineSource, msg.backendMode);
      self.postMessage({ type: 'response', id: msg.id, ok: true, result });
    } else if (msg.type === 'analyze') {
      const result = await handleAnalyze(msg);
      const transfer = result.imageEmbed.buffer.byteLength > 0 ? [result.imageEmbed.buffer] : [];
      self.postMessage({ type: 'response', id: msg.id, ok: true, result }, transfer);
    } else if (msg.type === 'customProbe') {
      const result = await handleCustomProbe(msg);
      self.postMessage({ type: 'response', id: msg.id, ok: true, result });
    } else if (msg.type === 'clearCache') {
      if ('caches' in self) {
        await caches.delete(CACHE_NAME);
      }
      self.postMessage({ type: 'response', id: msg.id, ok: true, result: true });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'response', id: msg.id, ok: false, error: message });
  }
};
