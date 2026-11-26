export type BackendMode = 'auto' | 'webgpu' | 'wasm';
export type EngineSource = 'local-q4' | 'osv5m-q4' | 'dual-q4' | 'hf-q4f16' | 'hf-q4';
export type ActiveBackend = 'webgpu' | 'wasm' | 'unknown';
export type AnalysisMode = 'center' | 'multicrop' | 'roi';

export interface RoiRect {
  x: number; // normalized [0, 1]
  y: number;
  w: number;
  h: number;
}

export interface CountryEntry {
  code: string;
  isoNum: string;
  name: string;
  continent: string;
  lat: number;
  lng: number;
  driving: 'left' | 'right';
  scripts: string[];
}

export interface RegionEntry {
  name: string;
  countryCode: string;
  lat: number;
  lng: number;
}

export interface CityEntry {
  name: string;
  region: string;
  countryCode: string;
  lat: number;
  lng: number;
}

export interface VisualProbeItem {
  id: string;
  label: string;
}

export interface VisualAxisSpec {
  id: string;
  title: string;
  items: VisualProbeItem[];
}

export interface LeadsCatalog {
  version: number;
  dim: number;
  logitScale: number;
  countries: CountryEntry[];
  regions: RegionEntry[];
  cities: CityEntry[];
  visualAxes: VisualAxisSpec[];
  offsets: {
    countries: [number, number];
    regions: [number, number];
    cities: [number, number];
    visualAxes: Record<string, [number, number]>;
  };
}

export interface ScoredCountry extends CountryEntry {
  prob: number;
  score: number;
  rank: number;
}

export interface ScoredRegion extends RegionEntry {
  countryName: string;
  prob: number;
  hierarchicalProb: number;
  score: number;
  rank: number;
}

export interface ScoredCity extends CityEntry {
  countryName: string;
  prob: number;
  hierarchicalProb: number;
  score: number;
  rank: number;
}

export interface ScoredVisualItem extends VisualProbeItem {
  prob: number;
  score: number;
}

export interface ScoredVisualAxis {
  id: string;
  title: string;
  items: ScoredVisualItem[];
  topItem: ScoredVisualItem;
}

export interface ConfidenceSummary {
  tier: 'High Signal' | 'Moderate Lead' | 'Ambiguous';
  entropyNormalized: number; // 0 = completely certain, 1 = uniform
  top1Margin: number; // top1 - top2 probability
  top1Prob: number;
  brief: string;
}

export interface GeocellPoint {
  cellId: number;
  lat: number;
  lng: number;
  prob: number;
}

export interface GeocellHotspot extends GeocellPoint {
  nearestCity?: string;
  nearestCountry?: string;
  distanceToCityKm?: number;
}

export interface GeocellPrediction {
  source: 'osv5m-11k' | 'streetclip-5k';
  numCells: number;
  lat: number;
  lng: number;
  topCellLat: number;
  topCellLng: number;
  topCellProb: number;
  spreadKm: number;
  hotspots: GeocellHotspot[];
  heatmap: GeocellPoint[];
  nearestCity?: string;
  nearestCountry?: string;
  distanceToCityKm?: number;
  secondaryCoord?: {
    lat: number;
    lng: number;
    label: string;
    divergenceKm: number;
  };
}

export interface AnalysisResult {
  imageEmbed: Float32Array;
  engineSource: EngineSource;
  hasVisualClues: boolean;
  countries: ScoredCountry[];
  regionsGlobal: ScoredRegion[];
  regionsHierarchical: ScoredRegion[];
  citiesGlobal: ScoredCity[];
  citiesHierarchical: ScoredCity[];
  visualAxes: ScoredVisualAxis[];
  confidence: ConfidenceSummary;
  geocell?: GeocellPrediction;
  latencyMs: number;
  cropCount: number;
  mode: AnalysisMode;
  backend: ActiveBackend;
}

export interface CustomProbeResultItem {
  label: string;
  prompt: string;
  prob: number;
  score: number;
  rank: number;
}

export interface CustomProbeResult {
  items: CustomProbeResultItem[];
  latencyMs: number;
  backend: ActiveBackend;
}

export interface ExifMetadata {
  make?: string;
  model?: string;
  lensModel?: string;
  dateTimeOriginal?: string;
  gps?: {
    lat: number;
    lng: number;
  };
  width?: number;
  height?: number;
}

export interface SamplePhoto {
  id: string;
  title: string;
  subtitle: string;
  url: string;
  hint?: string;
  expectedCountry?: string;
  expectedCity?: string;
}

export interface ModelBenchmark {
  generatedAt: string;
  sourceRepo: string;
  logitScale: number;
  sizesMB: {
    hf_fp32: number;
    hf_fp16: number;
    hf_int8: number;
    hf_q4: number;
    hf_q4f16: number;
    local_vision_q4: number;
    local_text_q4: number;
    local_embeddings_f16: number;
    local_tokenizer: number;
    local_geocells_f16?: number;
    local_osv5m_q4?: number;
  };
  visionQuantization: {
    method: string;
    blockSize: number;
    isSymmetric: boolean;
    accuracyLevel: number;
    minCosineSimilarityVsFp32: number;
    samples: Array<{
      file: string;
      cosineSimVsFp32: number;
      topCountryFp32: string;
      topCountryQ4: string;
      topCityFp32: string;
      topCityQ4: string;
    }>;
  };
  textQuantization: {
    method: string;
    blockSize: number;
    tokenEmbeddingQuantization: string;
    minCosineSimilarityVsFp32: number;
    promptsTested: number;
  };
  geocellProbe?: {
    sourceRepo: string;
    dataset: string;
    license: string;
    numCells: number;
    maxProbDiffVsF32: number;
    samples: Array<{
      file: string;
      topCellId: number;
      topCellProb: number;
      topCellMatchF32: boolean;
      estimatedLat: number;
      estimatedLng: number;
      errorKmVsGroundTruth: number;
    }>;
  };
  osv5mQuantization?: {
    sourceRepo: string;
    inputSize: number;
    numCells: number;
    method: string;
    blockSize: number;
    maxShiftKmVsFp32: number;
    samples: Array<{
      file: string;
      estimatedLat: number;
      estimatedLng: number;
      topCellProb: number;
      shiftKmVsFp32: number;
      errorKmVsGroundTruth: number;
    }>;
  };
}

export interface LoadProgress {
  stage: 'vision' | 'catalog' | 'geocells' | 'osv5m' | 'text' | 'hf' | 'ready';
  message: string;
  loadedBytes: number;
  totalBytes: number;
  cached?: boolean;
}

export type VlmModelId = 'qwen35-0.8b' | 'qwen3-vl-2b' | 'qwen35-2b';

export interface VlmModelSpec {
  id: VlmModelId;
  label: string;
  repoId: string;
  approxSizeLabel: string;
  enableThinkingFlag: boolean;
  notes: string;
}

export interface VlmProgress {
  stage: 'downloading' | 'loading' | 'generating' | 'ready';
  message: string;
  file?: string;
  loadedBytes: number;
  totalBytes: number;
  percent?: number;
}

export interface VlmResult {
  modelId: VlmModelId;
  modelLabel: string;
  backend: ActiveBackend;
  dtype: 'q4f16' | 'q4';
  text: string;
  latencyMs: number;
  tokenCount: number;
  tokensPerSec: number;
  groundedWithLeads: boolean;
}
