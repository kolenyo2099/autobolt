"""Prepare split, further-quantized StreetCLIP ONNX models and precomputed geolocation leads."""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
from PIL import Image
from tokenizers import Tokenizer

from leads_seed import COUNTRIES, REGIONS, CITIES, VISUAL_AXES

REPO_ID = "onnx-community/StreetCLIP-ONNX"
BASE_MODEL_REPO = "geolocal/StreetCLIP"

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "public" / "models"
TOKENIZER_DIR = MODELS_DIR / "tokenizer"
SAMPLES_DIR = ROOT_DIR / "public" / "samples"
WORK_DIR = Path("/tmp/streetclip-work")

CLIP_SIZE = 336
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

MAX_SEQ_LEN = 77
BOS_ID = 49406
EOS_ID = 49407


def mb_size(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def download_url(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "StreetCLIPLens/1.0 (https://localhost)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return False
            data = resp.read()
        if len(data) < 5000:
            return False
        dest.write_bytes(data)
        with Image.open(dest) as img:
            img.verify()
        return True
    except Exception as exc:
        print(f"[warn] Failed to download {url}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def preprocess_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = CLIP_SIZE / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    left = (nw - CLIP_SIZE) // 2
    top = (nh - CLIP_SIZE) // 2
    img = img.crop((left, top, left + CLIP_SIZE, top + CLIP_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)


def tokenize_batch(tokenizer: Tokenizer, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    batch_size = len(texts)
    input_ids = np.full((batch_size, MAX_SEQ_LEN), EOS_ID, dtype=np.int64)
    attention_mask = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)

    for i, text in enumerate(texts):
        enc = tokenizer.encode(text.lower(), add_special_tokens=False)
        ids = [BOS_ID] + enc.ids[: MAX_SEQ_LEN - 2] + [EOS_ID]
        n = len(ids)
        input_ids[i, :n] = ids
        attention_mask[i, :n] = 1
    return input_ids, attention_mask


def encode_texts_batched(
    sess: ort.InferenceSession,
    tokenizer: Tokenizer,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    all_embeds = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        input_ids, attention_mask = tokenize_batch(tokenizer, chunk)
        (embeds,) = sess.run(
            ["text_embeds"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / np.maximum(norms, 1e-12)
        all_embeds.append(embeds.astype(np.float32))
    return np.vstack(all_embeds)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    return float(
        np.dot(a_flat, b_flat)
        / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)
    )


def quantize_token_embedding_uint8(model_path: Path, out_path: Path) -> None:
    """Replace FP32 [49408, 768] token_embedding.weight with per-row uint8 + scale + zero_point."""
    model = onnx.load(str(model_path))
    target_init_name = "text_model.embeddings.token_embedding.weight"
    target_init = None
    for init in model.graph.initializer:
        if init.name == target_init_name:
            target_init = init
            break
    if target_init is None:
        raise RuntimeError(f"Could not find {target_init_name} in text graph")

    w = numpy_helper.to_array(target_init).astype(np.float32)  # [49408, 768]
    w_min = w.min(axis=1, keepdims=True)
    w_max = w.max(axis=1, keepdims=True)
    scale = np.maximum((w_max - w_min) / 255.0, 1e-8).astype(np.float32)
    zp_float = np.round(-w_min / scale).clip(0, 255).astype(np.float32)
    w_q = np.round(w / scale + zp_float).clip(0, 255).astype(np.uint8)

    # Remove original huge FP32 initializer
    model.graph.initializer.remove(target_init)

    q_name = "text_model.embeddings.token_embedding.weight_u8"
    scale_name = "text_model.embeddings.token_embedding.scale"
    zp_name = "text_model.embeddings.token_embedding.zp"

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(w_q, name=q_name),
            numpy_helper.from_array(scale, name=scale_name),
            numpy_helper.from_array(zp_float, name=zp_name),
        ]
    )

    gather_idx = None
    gather_node = None
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Gather" and target_init_name in node.input:
            gather_idx = idx
            gather_node = node
            break
    if gather_node is None or gather_idx is None:
        raise RuntimeError("Could not find token embedding Gather node")

    ids_input = gather_node.input[1]
    orig_output = gather_node.output[0]
    axis = 0
    for attr in gather_node.attribute:
        if attr.name == "axis":
            axis = attr.i

    n_gather_q = helper.make_node(
        "Gather", [q_name, ids_input], ["tok_emb_q_gathered"], name="TokEmbGatherU8", axis=axis
    )
    n_cast = helper.make_node(
        "Cast", ["tok_emb_q_gathered"], ["tok_emb_f32"], name="TokEmbCastF32", to=TensorProto.FLOAT
    )
    n_gather_zp = helper.make_node(
        "Gather", [zp_name, ids_input], ["tok_emb_zp_gathered"], name="TokEmbGatherZP", axis=axis
    )
    n_sub = helper.make_node(
        "Sub", ["tok_emb_f32", "tok_emb_zp_gathered"], ["tok_emb_shifted"], name="TokEmbSubZP"
    )
    n_gather_scale = helper.make_node(
        "Gather", [scale_name, ids_input], ["tok_emb_scale_gathered"], name="TokEmbGatherScale", axis=axis
    )
    n_mul = helper.make_node(
        "Mul", ["tok_emb_shifted", "tok_emb_scale_gathered"], [orig_output], name="TokEmbMulScale"
    )

    model.graph.node.remove(gather_node)
    for offset, new_node in enumerate(
        [n_gather_q, n_cast, n_gather_zp, n_sub, n_gather_scale, n_mul]
    ):
        model.graph.node.insert(gather_idx + offset, new_node)

    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


def quantize_matmul_4bit(
    src_path: Path,
    dst_path: Path,
    block_size: int,
    is_symmetric: bool = True,
    nodes_to_exclude: list[str] | None = None,
) -> None:
    config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=is_symmetric,
        accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=4,
    )
    model = quant_utils.load_model_with_shape_infer(src_path)
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model=model,
        algo_config=config,
        nodes_to_exclude=nodes_to_exclude,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(dst_path))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Downloading StreetCLIP-ONNX assets from HuggingFace Hub...")
    model_onnx_path = Path(hf_hub_download(REPO_ID, "onnx/model.onnx"))
    tokenizer_json_path = Path(hf_hub_download(REPO_ID, "tokenizer.json"))
    tokenizer_cfg_path = Path(hf_hub_download(REPO_ID, "tokenizer_config.json"))
    hf_hub_download(REPO_ID, "config.json")

    shutil.copy2(tokenizer_json_path, TOKENIZER_DIR / "tokenizer.json")
    shutil.copy2(tokenizer_cfg_path, TOKENIZER_DIR / "tokenizer_config.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_json_path))

    print("2. Downloading sample test images...")
    sample_candidates = [
        {
            "id": "nagasaki",
            "title": "Nagasaki Street",
            "subtitle": "StreetCLIP Benchmark",
            "filename": "nagasaki.jpg",
            "url": f"https://huggingface.co/{BASE_MODEL_REPO}/resolve/main/nagasaki.jpg",
            "expectedCountry": "Japan",
            "expectedCity": "Nagasaki",
        },
        {
            "id": "sanfrancisco",
            "title": "San Francisco",
            "subtitle": "StreetCLIP Benchmark",
            "filename": "sanfrancisco.jpeg",
            "url": f"https://huggingface.co/{BASE_MODEL_REPO}/resolve/main/sanfrancisco.jpeg",
            "expectedCountry": "United States",
            "expectedCity": "San Francisco",
        },
    ]

    manifest = []
    for item in sample_candidates:
        dest = SAMPLES_DIR / item["filename"]
        if not dest.exists():
            ok = download_url(item["url"], dest)
            if not ok:
                continue
        manifest.append(
            {
                "id": item["id"],
                "title": item["title"],
                "subtitle": item["subtitle"],
                "url": f"/samples/{item['filename']}",
                "expectedCountry": item.get("expectedCountry"),
                "expectedCity": item.get("expectedCity"),
            }
        )
    (SAMPLES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"   Saved {len(manifest)} sample images to public/samples/")

    print("3. Extracting logit_scale and splitting Vision & Text FP32 graphs...")
    full_model = onnx.load(str(model_onnx_path))
    logit_scale = 100.0
    for init in full_model.graph.initializer:
        if init.name == "/Cast_1_output_0":
            logit_scale = float(numpy_helper.to_array(init).flat[0])
            break
    print(f"   Extracted exact logit_scale = {logit_scale:.6f}")

    vision_fp32_path = WORK_DIR / "vision_fp32.onnx"
    text_fp32_path = WORK_DIR / "text_fp32.onnx"

    onnx.utils.extract_model(
        str(model_onnx_path),
        str(vision_fp32_path),
        ["pixel_values"],
        ["image_embeds"],
    )
    onnx.utils.extract_model(
        str(model_onnx_path),
        str(text_fp32_path),
        ["input_ids", "attention_mask"],
        ["text_embeds"],
    )
    print(f"   vision_fp32.onnx: {mb_size(vision_fp32_path)} MB")
    print(f"   text_fp32.onnx:   {mb_size(text_fp32_path)} MB")

    print("4. Precomputing FP32 prompt-ensembled embeddings for curated leads...")
    text_fp32_sess = ort.InferenceSession(str(text_fp32_path), providers=["CPUExecutionProvider"])

    # Countries (2-prompt ensemble faithful to StreetCLIP paper)
    country_p1 = [f"A Street View photo in {c['name']}." for c in COUNTRIES]
    country_p2 = [f"A photo taken in {c['name']}." for c in COUNTRIES]
    c_emb1 = encode_texts_batched(text_fp32_sess, tokenizer, country_p1)
    c_emb2 = encode_texts_batched(text_fp32_sess, tokenizer, country_p2)
    country_embs = c_emb1 + c_emb2
    country_embs /= np.maximum(np.linalg.norm(country_embs, axis=1, keepdims=True), 1e-12)

    # Regions
    country_name_by_code = {c["code"]: c["name"] for c in COUNTRIES}
    region_prompts = [
        f"A Street View photo in the region of {name} in {country_name_by_code[cc]}."
        for name, cc, _, _ in REGIONS
    ]
    region_embs = encode_texts_batched(text_fp32_sess, tokenizer, region_prompts)

    # Cities (2-prompt ensemble faithful to StreetCLIP paper)
    city_p1 = [
        f"A Street View photo close to the town of {city} in the region of {region} in {country_name_by_code[cc]}."
        for city, region, cc, _, _ in CITIES
    ]
    city_p2 = [
        f"A Street View photo from {city}, {country_name_by_code[cc]}."
        for city, _, cc, _, _ in CITIES
    ]
    city_e1 = encode_texts_batched(text_fp32_sess, tokenizer, city_p1)
    city_e2 = encode_texts_batched(text_fp32_sess, tokenizer, city_p2)
    city_embs = city_e1 + city_e2
    city_embs /= np.maximum(np.linalg.norm(city_embs, axis=1, keepdims=True), 1e-12)

    # Visual Clue Axes
    axis_embs_list = []
    visual_axes_meta = []
    visual_offsets = {}
    cursor = len(COUNTRIES) + len(REGIONS) + len(CITIES)

    for axis in VISUAL_AXES:
        prompts = [item["prompt"] for item in axis["items"]]
        embs = encode_texts_batched(text_fp32_sess, tokenizer, prompts)
        axis_embs_list.append(embs)
        visual_offsets[axis["id"]] = [cursor, len(axis["items"])]
        cursor += len(axis["items"])
        visual_axes_meta.append(
            {
                "id": axis["id"],
                "title": axis["title"],
                "items": [{"id": it["id"], "label": it["label"]} for it in axis["items"]],
            }
        )

    all_embs = np.vstack([country_embs, region_embs, city_embs, *axis_embs_list]).astype(np.float16)
    emb_bin_path = MODELS_DIR / "leads_embeddings.f16.bin"
    emb_bin_path.write_bytes(all_embs.tobytes())

    catalog = {
        "version": 1,
        "dim": int(all_embs.shape[1]),
        "logitScale": logit_scale,
        "countries": COUNTRIES,
        "regions": [
            {"name": name, "countryCode": cc, "lat": lat, "lng": lng}
            for name, cc, lat, lng in REGIONS
        ],
        "cities": [
            {"name": city, "region": region, "countryCode": cc, "lat": lat, "lng": lng}
            for city, region, cc, lat, lng in CITIES
        ],
        "visualAxes": visual_axes_meta,
        "offsets": {
            "countries": [0, len(COUNTRIES)],
            "regions": [len(COUNTRIES), len(REGIONS)],
            "cities": [len(COUNTRIES) + len(REGIONS), len(CITIES)],
            "visualAxes": visual_offsets,
        },
    }
    (MODELS_DIR / "leads_catalog.json").write_text(json.dumps(catalog))
    print(
        f"   Saved leads_embeddings.f16.bin ({mb_size(emb_bin_path)} MB, {all_embs.shape[0]} vectors)"
    )

    print("5. Quantizing Vision Encoder (4-bit symmetric MatMul, comparing block_size=128 vs 32)...")
    vision_fp32_sess = ort.InferenceSession(str(vision_fp32_path), providers=["CPUExecutionProvider"])
    sample_inputs = {
        Path(m["url"]).name: preprocess_image(SAMPLES_DIR / Path(m["url"]).name) for m in manifest
    }
    fp32_vis_embeds = {
        fn: vision_fp32_sess.run(["image_embeds"], {"pixel_values": arr})[0]
        for fn, arr in sample_inputs.items()
    }

    best_vision_path = None
    best_vision_block = 128
    best_vision_stats = None

    for block_size in (128, 32):
        cand_path = WORK_DIR / f"vision_q4_b{block_size}.onnx"
        t0 = time.time()
        quantize_matmul_4bit(vision_fp32_path, cand_path, block_size=block_size)
        dt = time.time() - t0
        cand_sess = ort.InferenceSession(str(cand_path), providers=["CPUExecutionProvider"])

        sample_reports = []
        min_cos = 1.0
        exact_match = True
        for m in manifest:
            fn = Path(m["url"]).name
            q_emb = cand_sess.run(["image_embeds"], {"pixel_values": sample_inputs[fn]})[0]
            cos = cosine_sim(fp32_vis_embeds[fn], q_emb)
            min_cos = min(min_cos, cos)

            # Top country & city predictions
            fp_c_scores = (fp32_vis_embeds[fn][0] @ country_embs.T)
            q_c_scores = (q_emb[0] @ country_embs.T)
            fp_country = COUNTRIES[int(np.argmax(fp_c_scores))]["name"]
            q_country = COUNTRIES[int(np.argmax(q_c_scores))]["name"]

            fp_city_scores = (fp32_vis_embeds[fn][0] @ city_embs.T)
            q_city_scores = (q_emb[0] @ city_embs.T)
            fp_city = CITIES[int(np.argmax(fp_city_scores))][0]
            q_city = CITIES[int(np.argmax(q_city_scores))][0]

            if fp_country != q_country or fp_city != q_city:
                exact_match = False

            sample_reports.append(
                {
                    "file": fn,
                    "cosineSimVsFp32": round(cos, 5),
                    "topCountryFp32": fp_country,
                    "topCountryQ4": q_country,
                    "topCityFp32": fp_city,
                    "topCityQ4": q_city,
                }
            )

        print(
            f"   block_size={block_size}: {mb_size(cand_path)} MB (quantized in {dt:.1f}s), "
            f"min_cos={min_cos:.5f}, exact_top1_match={exact_match}"
        )
        for rep in sample_reports:
            print(f"     {rep['file']}: cos={rep['cosineSimVsFp32']}, country={rep['topCountryQ4']}, city={rep['topCityQ4']}")

        if min_cos >= 0.980 and exact_match and best_vision_path is None:
            best_vision_path = cand_path
            best_vision_block = block_size
            best_vision_stats = (min_cos, sample_reports)

    if best_vision_path is None:
        # Fall back to block_size=32 if 128 didn't hit threshold
        best_vision_path = WORK_DIR / "vision_q4_b32.onnx"
        best_vision_block = 32

    final_vision_path = MODELS_DIR / "vision_q4.onnx"
    shutil.copy2(best_vision_path, final_vision_path)
    print(f"   Selected public/models/vision_q4.onnx (block_size={best_vision_block}, {mb_size(final_vision_path)} MB)")

    print("6. Quantizing Text Encoder (per-row uint8 token embeddings + 4-bit asymmetric MatMul)...")
    text_emb_q_path = WORK_DIR / "text_emb_u8.onnx"
    quantize_token_embedding_uint8(text_fp32_path, text_emb_q_path)
    final_text_path = MODELS_DIR / "text_q4.onnx"
    quantize_matmul_4bit(
        text_emb_q_path,
        final_text_path,
        block_size=32,
        is_symmetric=False,
        nodes_to_exclude=["/text_projection/MatMul"],
    )

    text_q4_sess = ort.InferenceSession(str(final_text_path), providers=["CPUExecutionProvider"])
    test_prompts = [
        "A Street View photo in Japan.",
        "A Street View photo from San Francisco, United States.",
        "A Street View photo close to the town of Nagasaki in the region of Nagasaki Prefecture in Japan.",
        "A coastal Mediterranean town with terracotta roofs and palm trees.",
        "Yellow center lane markings with wooden utility poles and conifer forest.",
    ]
    t_ids, t_mask = tokenize_batch(tokenizer, test_prompts)
    (t_fp_out,) = text_fp32_sess.run(["text_embeds"], {"input_ids": t_ids, "attention_mask": t_mask})
    (t_q_out,) = text_q4_sess.run(["text_embeds"], {"input_ids": t_ids, "attention_mask": t_mask})
    text_cos_sims = [cosine_sim(t_fp_out[i], t_q_out[i]) for i in range(len(test_prompts))]
    min_text_cos = min(text_cos_sims)
    print(
        f"   Saved public/models/text_q4.onnx ({mb_size(final_text_path)} MB), "
        f"min_cos={min_text_cos:.5f} across {len(test_prompts)} prompts"
    )

    # Record benchmark summary
    benchmark = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sourceRepo": REPO_ID,
        "logitScale": logit_scale,
        "sizesMB": {
            "hf_fp32": 1712.53,
            "hf_fp16": 856.64,
            "hf_int8": 433.62,
            "hf_q4": 402.05,
            "hf_q4f16": 298.49,
            "local_vision_q4": mb_size(final_vision_path),
            "local_text_q4": mb_size(final_text_path),
            "local_embeddings_f16": mb_size(emb_bin_path),
            "local_tokenizer": round(
                (TOKENIZER_DIR / "tokenizer.json").stat().st_size / (1024 * 1024), 2
            ),
        },
        "visionQuantization": {
            "method": "MatMulNBitsQuantizer (RTN symmetric weight-only INT4)",
            "blockSize": best_vision_block,
            "isSymmetric": True,
            "accuracyLevel": 4,
            "minCosineSimilarityVsFp32": round(best_vision_stats[0], 5) if best_vision_stats else 0,
            "samples": best_vision_stats[1] if best_vision_stats else [],
        },
        "textQuantization": {
            "method": "MatMulNBitsQuantizer (block_size=32, asymmetric INT4, FP32 text_projection)",
            "blockSize": 32,
            "tokenEmbeddingQuantization": "Per-row UINT8 (49408x768) reconstructed with Gather+Cast+Sub+Mul",
            "minCosineSimilarityVsFp32": round(min_text_cos, 5),
            "promptsTested": len(test_prompts),
        },
    }
    (MODELS_DIR / "model_benchmark.json").write_text(json.dumps(benchmark, indent=2))
    print("Done! All model artifacts & benchmark written to public/models/.")


if __name__ == "__main__":
    main()
