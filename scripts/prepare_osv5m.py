"""Export, 4-bit quantize, and benchmark osv5m/baseline for browser WebGPU/WASM inference."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
from PIL import Image
import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

OSV5M_REPO = "osv5m/baseline"
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "public" / "models"
SAMPLES_DIR = ROOT_DIR / "public" / "samples"
WORK_DIR = Path("/tmp/streetclip-work")

CLIP_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

GROUND_TRUTH = {
    "nagasaki.jpg": (32.7503, 129.8779),
    "sanfrancisco.jpeg": (37.7749, -122.4194),
}


def mb_size(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def preprocess_224(path: Path) -> np.ndarray:
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


class DecomposedGroupNorm(nn.Module):
    """GroupNorm decomposed into Reshape/Mean/Rsqrt/Mul/Add for universal ORT WebGPU/WASM support."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        g = self.num_groups
        xg = x.view(b, g, self.num_channels // g)
        mean = xg.mean(dim=-1, keepdim=True)
        diff = xg - mean
        var = (diff * diff).mean(dim=-1, keepdim=True)
        normed = (diff * torch.rsqrt(var + self.eps)).view(b, self.num_channels)
        return normed * self.weight + self.bias


class MatMulLinear(nn.Module):
    """Linear layer exported as MatMul + Add so MatMulNBitsQuantizer quantizes MLP weights."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t()) + self.bias


def make_mlp_branch(final_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(1024),
        MatMulLinear(1024, 1024),
        DecomposedGroupNorm(4, 1024),
        nn.GELU(),
        MatMulLinear(1024, 512),
        DecomposedGroupNorm(4, 512),
        nn.GELU(),
        MatMulLinear(512, final_dim),
    )


class Osv5mOnnxModel(nn.Module):
    def __init__(self, vision_config: CLIPVisionConfig, num_cells: int = 11399):
        super().__init__()
        self.vision_model = CLIPVisionModel(vision_config)
        self.classif = make_mlp_branch(num_cells)
        self.reg = make_mlp_branch(num_cells * 2)
        self.num_cells = num_cells

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.vision_model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        class_logits = self.classif(feats)
        reg_raw = self.reg(feats)
        reg_offsets = (1.2 * torch.tanh(reg_raw)).view(-1, self.num_cells, 2)
        return class_logits, reg_offsets


def decode_top1_coords(
    class_logits: np.ndarray, reg_offsets: np.ndarray, quadtree: np.ndarray
) -> tuple[int, float, float, float]:
    x = class_logits[0] - np.max(class_logits[0])
    probs = np.exp(x) / np.sum(np.exp(x))
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    mean_lat, mean_lon, up_lat, up_lon, down_lat, down_lon = quadtree[top_idx]
    off_lat, off_lon = reg_offsets[0, top_idx]
    s_lat = up_lat if off_lat > 0 else down_lat
    s_lon = up_lon if off_lon > 0 else down_lon

    lat = max(-1.0, min(1.0, float(mean_lat + off_lat * s_lat))) * 90.0
    lon = max(-1.0, min(1.0, float(mean_lon + off_lon * s_lon))) * 180.0
    return top_idx, top_prob, lat, lon


def quantize_osv5m_4bit(
    src_path: Path, dst_path: Path, block_size: int, exclude_reg_head: bool = False
) -> None:
    nodes_to_exclude = []
    if exclude_reg_head:
        m = onnx.load(str(src_path), load_external_data=False)
        for n in m.graph.node:
            if n.op_type == "MatMul" and ("reg/7" in n.name or "reg.7" in n.name):
                nodes_to_exclude.append(n.name)

    cfg = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=4,
    )
    model = quant_utils.load_model_with_shape_infer(src_path)
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model=model,
        algo_config=cfg,
        nodes_to_exclude=nodes_to_exclude or None,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(dst_path))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading osv5m/baseline checkpoint...")
    ckpt_path = Path(hf_hub_download(OSV5M_REPO, "pytorch_model.bin"))
    raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw_sd, dict) and "state_dict" in raw_sd:
        raw_sd = raw_sd["state_dict"]

    vision_cfg = CLIPVisionConfig(
        hidden_act="gelu",
        hidden_size=1024,
        image_size=224,
        intermediate_size=4096,
        layer_norm_eps=1e-5,
        num_attention_heads=16,
        num_hidden_layers=24,
        patch_size=14,
        projection_dim=512,
    )
    model = Osv5mOnnxModel(vision_cfg, num_cells=11399)

    mapped_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("backbone.clip."):
            new_k = k[len("backbone.clip.") :]
            mapped_sd[new_k] = v
        elif k.startswith("mid.classif."):
            mapped_sd[k[len("mid.") :]] = v
        elif k.startswith("mid.reg."):
            mapped_sd[k[len("mid.") :]] = v

    missing, unexpected = model.load_state_dict(mapped_sd, strict=True)
    assert not missing and not unexpected, f"Mismatch: missing={missing}, unexpected={unexpected}"
    model.eval()
    print("   Loaded all weights strictly into Osv5mOnnxModel!")

    # Pack quadtree buffers directly from checkpoint
    center = raw_sd["head.cell_center"].cpu().numpy().astype(np.float32)
    up = raw_sd["head.cell_size_up"].cpu().numpy().astype(np.float32)
    down = raw_sd["head.cell_size_down"].cpu().numpy().astype(np.float32)
    quadtree = np.concatenate([center, up, down], axis=1).astype("<f4")  # (11399, 6)
    quadtree_path = MODELS_DIR / "osv5m_quadtree.f32.bin"
    quadtree_path.write_bytes(quadtree.tobytes())
    print(f"   Saved {quadtree_path.name} ({quadtree_path.stat().st_size / 1024:.1f} KB)")

    print("2. Exporting osv5m_fp32.onnx...")
    fp32_path = WORK_DIR / "osv5m_fp32.onnx"
    dummy = torch.randn(1, 3, CLIP_SIZE, CLIP_SIZE)
    torch.onnx.export(
        model,
        (dummy,),
        str(fp32_path),
        input_names=["pixel_values"],
        output_names=["class_logits", "reg_offsets"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "class_logits": {0: "batch"},
            "reg_offsets": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"   osv5m_fp32.onnx: {mb_size(fp32_path)} MB")

    # Verify ONNX FP32 vs PyTorch FP32
    fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sample_inputs = {}
    fp32_preds = {}
    for fn, (gt_lat, gt_lon) in GROUND_TRUTH.items():
        img_path = SAMPLES_DIR / fn
        arr = preprocess_224(img_path)
        sample_inputs[fn] = arr
        with torch.no_grad():
            pt_logits, pt_reg = model(torch.from_numpy(arr))
        ort_logits, ort_reg = fp32_sess.run(["class_logits", "reg_offsets"], {"pixel_values": arr})
        max_diff = float(np.max(np.abs(pt_logits.numpy() - ort_logits)))
        top_idx, top_prob, lat, lon = decode_top1_coords(ort_logits, ort_reg, quadtree)
        err_km = haversine_km(gt_lat, gt_lon, lat, lon)
        fp32_preds[fn] = (top_idx, top_prob, lat, lon, err_km)
        print(
            f"   FP32 {fn}: max_pt_diff={max_diff:.6f} | cell={top_idx} ({top_prob:.3f}) -> "
            f"({lat:.3f}, {lon:.3f}), err={err_km:.1f} km"
        )

    print("3. Quantizing OSV-5M to 4-bit INT4 (comparing block_size=128 vs 32)...")
    best_path = None
    best_block = 128
    best_max_shift = 99999.0
    best_reports = []

    for bs in (128, 32):
        cand_path = WORK_DIR / f"osv5m_q4_b{bs}.onnx"
        t0 = time.time()
        quantize_osv5m_4bit(fp32_path, cand_path, block_size=bs, exclude_reg_head=False)
        dt = time.time() - t0
        qs = ort.InferenceSession(str(cand_path), providers=["CPUExecutionProvider"])

        max_shift = 0.0
        reports = []
        for fn, (gt_lat, gt_lon) in GROUND_TRUTH.items():
            q_logits, q_reg = qs.run(["class_logits", "reg_offsets"], {"pixel_values": sample_inputs[fn]})
            top_idx, top_prob, lat, lon = decode_top1_coords(q_logits, q_reg, quadtree)
            _, _, fp_lat, fp_lon, _ = fp32_preds[fn]
            shift_km = haversine_km(fp_lat, fp_lon, lat, lon)
            err_km = haversine_km(gt_lat, gt_lon, lat, lon)
            max_shift = max(max_shift, shift_km)
            reports.append(
                {
                    "file": fn,
                    "estimatedLat": round(lat, 4),
                    "estimatedLng": round(lon, 4),
                    "topCellProb": round(top_prob, 4),
                    "shiftKmVsFp32": round(shift_km, 1),
                    "errorKmVsGroundTruth": round(err_km, 1),
                }
            )
            print(
                f"     bs={bs} {fn}: ({lat:.3f}, {lon:.3f}) prob={top_prob:.3f} | "
                f"shift_vs_fp32={shift_km:.1f} km | gt_err={err_km:.1f} km"
            )

        print(f"   bs={bs}: size={mb_size(cand_path)} MB (in {dt:.1f}s), max_shift={max_shift:.1f} km")
        if max_shift < 25.0 and best_path is None:
            best_path = cand_path
            best_block = bs
            best_max_shift = max_shift
            best_reports = reports

    if best_path is None:
        best_path = WORK_DIR / "osv5m_q4_b32.onnx"
        best_block = 32
        best_max_shift = max_shift
        best_reports = reports

    final_path = MODELS_DIR / "osv5m_q4.onnx"
    shutil.copy2(best_path, final_path)
    print(f"Saved {final_path} ({mb_size(final_path)} MB, block_size={best_block})")

    bench_path = MODELS_DIR / "model_benchmark.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        bench.setdefault("sizesMB", {})["local_osv5m_q4"] = mb_size(final_path)
        bench["osv5mQuantization"] = {
            "sourceRepo": OSV5M_REPO,
            "inputSize": CLIP_SIZE,
            "numCells": 11399,
            "method": "MatMulNBitsQuantizer (RTN symmetric weight-only INT4)",
            "blockSize": best_block,
            "maxShiftKmVsFp32": round(best_max_shift, 1),
            "samples": best_reports,
        }
        bench_path.write_text(json.dumps(bench, indent=2))
        print("Updated model_benchmark.json")


if __name__ == "__main__":
    main()
