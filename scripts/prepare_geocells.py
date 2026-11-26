"""Download and quantize (FP16) the OSV-5M 5,000-cell coordinate probe head for StreetCLIP."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

GEOCELL_REPO = "Daggerabuse/geoguessr-streetclip-geocells"
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "public" / "models"
SAMPLES_DIR = ROOT_DIR / "public" / "samples"

CLIP_SIZE = 336
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

GROUND_TRUTH = {
    "nagasaki.jpg": (32.7503, 129.8779),
    "sanfrancisco.jpeg": (37.7749, -122.4194),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def spherical_cluster_mean(
    lats: np.ndarray, lons: np.ndarray, probs: np.ndarray, radius_km: float = 300.0
) -> tuple[float, float]:
    top_idx = int(np.argmax(probs))
    top_lat, top_lon = float(lats[top_idx]), float(lons[top_idx])

    wx = wy = wz = wsum = 0.0
    for i in range(len(probs)):
        p = float(probs[i])
        if p < 1e-4:
            continue
        if haversine_km(top_lat, top_lon, float(lats[i]), float(lons[i])) <= radius_km:
            lat_r = math.radians(float(lats[i]))
            lon_r = math.radians(float(lons[i]))
            wx += p * math.cos(lat_r) * math.cos(lon_r)
            wy += p * math.cos(lat_r) * math.sin(lon_r)
            wz += p * math.sin(lat_r)
            wsum += p

    if wsum <= 0:
        return top_lat, top_lon
    wx /= wsum
    wy /= wsum
    wz /= wsum
    hyp = math.hypot(wx, wy)
    return math.degrees(math.atan2(wz, hyp)), math.degrees(math.atan2(wy, wx))


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
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    ex = np.exp(x)
    return ex / np.sum(ex)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {GEOCELL_REPO} head_f32.bin & cells.json...")
    head_f32_path = Path(hf_hub_download(GEOCELL_REPO, "head_f32.bin"))
    cells_json_path = Path(hf_hub_download(GEOCELL_REPO, "cells.json"))

    cells = json.loads(cells_json_path.read_text())
    n_cells = int(cells["n_cells"])
    lats = np.asarray(cells["centroid_lat"], dtype=np.float32)
    lons = np.asarray(cells["centroid_lon"], dtype=np.float32)
    assert lats.shape == (n_cells,) and lons.shape == (n_cells,)

    head_f32 = np.fromfile(head_f32_path, dtype="<f4").reshape(n_cells, 768)
    head_f16 = head_f32.astype("<f2")

    out_head = MODELS_DIR / "geocells_head.f16.bin"
    out_coords = MODELS_DIR / "geocells_coords.f32.bin"

    out_head.write_bytes(head_f16.tobytes())
    coords_interleaved = np.stack([lats, lons], axis=1).astype("<f4")
    out_coords.write_bytes(coords_interleaved.tobytes())

    head_f16_mb = round(out_head.stat().st_size / (1024 * 1024), 2)
    print(f"Saved {out_head.name} ({head_f16_mb} MB) and {out_coords.name} ({out_coords.stat().st_size / 1024:.1f} KB)")

    # Verify with our existing vision_q4.onnx
    vision_sess = ort.InferenceSession(str(MODELS_DIR / "vision_q4.onnx"), providers=["CPUExecutionProvider"])
    head_f16_up = head_f16.astype(np.float32)

    sample_results = []
    max_prob_diff = 0.0
    for fn, (gt_lat, gt_lon) in GROUND_TRUTH.items():
        img_path = SAMPLES_DIR / fn
        if not img_path.exists():
            continue
        pv = preprocess_image(img_path)
        (img_emb,) = vision_sess.run(["image_embeds"], {"pixel_values": pv})
        emb = img_emb[0] / max(float(np.linalg.norm(img_emb[0])), 1e-12)

        probs_f32 = softmax(100.0 * (head_f32 @ emb))
        probs_f16 = softmax(100.0 * (head_f16_up @ emb))

        top_f32 = int(np.argmax(probs_f32))
        top_f16 = int(np.argmax(probs_f16))
        prob_diff = float(np.max(np.abs(probs_f32 - probs_f16)))
        max_prob_diff = max(max_prob_diff, prob_diff)

        est_lat, est_lon = spherical_cluster_mean(lats, lons, probs_f16, radius_km=300.0)
        err_top1 = haversine_km(gt_lat, gt_lon, float(lats[top_f16]), float(lons[top_f16]))
        err_est = haversine_km(gt_lat, gt_lon, est_lat, est_lon)

        print(
            f"  {fn}: top_f32==top_f16? {top_f32 == top_f16} (cell {top_f16}, prob={probs_f16[top_f16]:.3f}, max_diff={prob_diff:.5f}) | "
            f"top1=({lats[top_f16]:.3f}, {lons[top_f16]:.3f}, err={err_top1:.1f}km) | "
            f"cluster=({est_lat:.3f}, {est_lon:.3f}, err={err_est:.1f}km)"
        )
        sample_results.append(
            {
                "file": fn,
                "topCellId": top_f16,
                "topCellProb": round(float(probs_f16[top_f16]), 4),
                "topCellMatchF32": top_f32 == top_f16,
                "estimatedLat": round(est_lat, 4),
                "estimatedLng": round(est_lon, 4),
                "errorKmVsGroundTruth": round(err_est, 1),
            }
        )

    bench_path = MODELS_DIR / "model_benchmark.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        bench.setdefault("sizesMB", {})["local_geocells_f16"] = head_f16_mb
        bench["geocellProbe"] = {
            "sourceRepo": GEOCELL_REPO,
            "dataset": "osv5m/osv5m (500k subset, 5,000 quadtree cells)",
            "license": "CC BY-SA 4.0 (cells) / CC BY-NC 4.0 (StreetCLIP backbone)",
            "numCells": n_cells,
            "maxProbDiffVsF32": round(max_prob_diff, 5),
            "samples": sample_results,
        }
        bench_path.write_text(json.dumps(bench, indent=2))
        print("Updated model_benchmark.json")


if __name__ == "__main__":
    main()
