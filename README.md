# Investigations Lab's AutoBolt

100% browser-native image geolocation leads workbench powered by **StreetCLIP ONNX** ([`onnx-community/StreetCLIP-ONNX`](https://huggingface.co/onnx-community/StreetCLIP-ONNX)) and **OSV-5M Baseline** ([`osv5m/baseline`](https://huggingface.co/osv5m/baseline)), running directly in your browser via **WebGPU** (with **WASM** fallback) — zero server uploads, zero external inference APIs.

## Key Features

- **Split & Further-Quantized StreetCLIP ONNX**:
  - Splits the monolithic 1,712 MB CLIP ViT-L/14@336 model into independent Vision (`159.6 MB`) and on-demand Text (`91.7 MB`) encoders.
  - Precomputes FP32-precision prompt-ensembled embeddings for **125 countries**, **164 regions/states**, **252 cities**, and **6 visual clue axes** into a `1.0 MB` IEEE-754 Float16 binary (`public/models/leads_embeddings.f16.bin`).
  - Pairs `vision_q4.onnx` with an **OSV-5M 5,000-cell coordinate head** (`public/models/geocells_head.f16.bin`, `7.32 MB` FP16, derived from [`Daggerabuse/geoguessr-streetclip-geocells`](https://huggingface.co/Daggerabuse/geoguessr-streetclip-geocells), CC BY-SA 4.0 / CC BY-NC 4.0) to output continuous `(lat, lng)` spherical estimates, spatial spread, 5 distinct hotspots, and a world heatmap across **225 countries/territories** with zero extra vision runtime cost.
  - Optional **OSV-5M Baseline Q4** (`public/models/osv5m_q4.onnx`, exported from [`osv5m/baseline`](https://huggingface.co/osv5m/baseline) and 4-bit quantized): `224×224` model with **11,399 quadtree geocells + continuous sub-cell regression**, selectable standalone or in **Dual Mode** alongside StreetCLIP.
  - **46.5% smaller** than Hugging Face's smallest `model_q4f16.onnx` (`298.5 MB`) and **90.7% smaller** than FP32 (`1,712.5 MB`), while preserving `>= 0.983` cosine similarity against FP32.
- **Hierarchical Linear Probing** (`arXiv:2302.00275`):
  - Scores Countries → Regions/States → Cities/Towns sequentially using StreetCLIP's domain-specific prompt templates, plus entropy/margin confidence calibration.
- **Visual & Environmental Lead Axes**:
  - Independently scores Environment & Density, Climate & Biome, Architectural Vernacular, Road/Poles/Transit Cues, Scripts & License Plates, and Lighting/Weather.
- **Interactive Investigation Tools**:
  - Drag-and-drop, file upload, and clipboard paste (`Cmd/Ctrl+V`).
  - **Center 336×336**, **Wide 3-Crop** (spatial averaging for wide panoramas), and **Focus ROI Box** (drag a box around a sign, storefront, or landmark to isolate leads).
  - **Interactive SVG Vector World Map** (`d3-geo` Natural Earth + `world-atlas` TopoJSON) with country probability choropleth, top city pins, zoom/pan, and great-circle geodesic line to revealed EXIF GPS.
  - **EXIF Blind Ground-Truth Check** via `exifr` (camera metadata + hidden GPS spoiler with Haversine km accuracy check).
  - **Custom Hypothesis Probe** tab with lazy-loaded quantized Text Q4 encoder + local CLIP tokenizer.
  - **Markdown & JSON Lead Brief Export**.

## Model Quantization Benchmark

| Variant | Size | Reduction vs FP32 | Notes |
| --- | --- | --- | --- |
| Upstream HF `onnx/model.onnx` (FP32) | 1,712.5 MB | 1.0× | Monolithic Vision + Text |
| Upstream HF `onnx/model_fp16.onnx` | 856.6 MB | 2.0× | Monolithic FP16 |
| Upstream HF `onnx/model_q4.onnx` | 402.1 MB | 4.3× | Monolithic 4-bit + FP32 |
| Upstream HF `onnx/model_q4f16.onnx` | 298.5 MB | 5.7× | Smallest upstream HF file |
| **AutoBolt `vision_q4.onnx` + Leads Pack** | **160.6 MB** | **10.7×** | Split INT4 (`block_size=128`), min cos `0.9834` |
| **On-Demand `text_q4.onnx` (Custom Probe)** | **91.7 MB** | Lazy-loaded | INT4 MatMul (`block_size=32` asym) + UINT8 token table, min cos `0.9814` |

## Local Development

```bash
pnpm install
pnpm dev
```

To regenerate the quantized ONNX models, precomputed leads binary, and benchmark JSON from `onnx-community/StreetCLIP-ONNX`:

```bash
uv run --python 3.12 --with onnx --with onnxruntime --with onnx-ir --with huggingface_hub --with numpy --with pillow --with tokenizers scripts/prepare_models.py
```

Production build:

```bash
pnpm build
```
