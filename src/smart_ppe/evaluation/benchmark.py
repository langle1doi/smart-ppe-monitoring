"""Latency / throughput benchmark under one controlled protocol.

Protocol: batch size 1, images pre-decoded into RAM (decoding NOT timed), same images/imgsz/conf/device for every
model, `warmup` untimed passes, CUDA synchronised. Component times (preprocess / inference / postprocess) come from
Ultralytics' own per-image timers; wall time additionally covers Python wrapper overhead. Visualization is NOT timed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from smart_ppe.utils.device import (free_gpu, peak_gpu_memory_mb, reset_peak_memory, select_device, sync)
from smart_ppe.utils.logging_utils import get_logger
from smart_ppe.utils.paths import (apply_overrides, ensure_dir, find_split_dir, list_images, load_yaml,
                                   resolve_dataset_root)

log = get_logger(__name__)


def _stats(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}


def benchmark_detector(detector: Any, images: Sequence[np.ndarray], warmup: int = 20, conf: float = 0.25,
                       device: str = "cpu") -> dict[str, Any]:
    """Time `detector.predict` over `images` (after warmup). Returns a JSON-able dict."""
    if not images:
        raise ValueError("benchmark needs at least one image")
    for i in range(warmup):
        detector.predict(images[i % len(images)], conf=conf)
    sync(device)
    reset_peak_memory(device)
    pre, inf, post, wall = [], [], [], []
    for img in images:
        t0 = time.perf_counter()
        _, t = detector.predict(img, conf=conf)
        sync(device)
        wall.append((time.perf_counter() - t0) * 1000)
        pre.append(t.preprocess_ms)
        inf.append(t.inference_ms)
        post.append(t.postprocess_ms)
    total = [a + b + c for a, b, c in zip(pre, inf, post)]
    return {
        "images": len(images), "warmup": warmup, "conf": conf, "device": device, "batch_size": 1,
        "imgsz": detector.imgsz,
        "latency_ms": {"preprocess": _stats(pre), "inference": _stats(inf), "postprocess": _stats(post),
                       "pipeline_total": _stats(total), "wall_per_image": _stats(wall)},
        "fps_pipeline": 1000.0 / float(np.mean(total)),   # pre+inference+post, from Ultralytics timers
        "fps_wall": 1000.0 / float(np.mean(wall)),        # also includes Python call overhead
        "includes_video_decoding": False, "includes_visualization": False,
        "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
        "model_size_mb": round(Path(detector.weights).stat().st_size / 2**20, 2),
        "params_m": round(detector.num_params() / 1e6, 2),
    }


def load_split_images(ds_cfg: dict[str, Any], split: str, n: int) -> list[np.ndarray]:
    sdir = find_split_dir(resolve_dataset_root(ds_cfg), split)
    if sdir is None:
        raise FileNotFoundError(f"split '{split}' not found in dataset root")
    imgs = []
    for p in list_images(sdir / "images"):
        im = cv2.imread(str(p))
        if im is not None:
            imgs.append(im)
        if len(imgs) >= n:
            break
    return imgs


def run_benchmark(kind: str, weights: str, images: list[np.ndarray], device: str, imgsz: int, conf: float,
                  iou: float, warmup: int, name: str | None = None) -> dict[str, Any]:
    from smart_ppe.models import load_detector

    det = load_detector(kind, weights, device, imgsz, iou)
    res = {"name": name or kind, "model": kind, "weights": weights,
           **benchmark_detector(det, images, warmup, conf, device)}
    del det
    free_gpu()
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark one detector's latency/FPS/memory")
    ap.add_argument("--model", required=True, choices=["yolo", "rtdetr"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default="configs/dataset.yaml", help="dataset config YAML")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name")
    ap.add_argument("--metrics-dir", default="outputs/metrics")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dataset config override")
    args = ap.parse_args(argv)
    try:
        device = select_device(args.device)
        images = load_split_images(apply_overrides(load_yaml(args.config), args.set), args.split, args.num_images)
        res = run_benchmark(args.model, args.weights, images, device, args.imgsz, args.conf, args.iou,
                            args.warmup, args.name)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        return 1
    out = ensure_dir(args.metrics_dir) / f"{res['name']}_benchmark.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info(f"{res['name']}: {res['fps_pipeline']:.1f} FPS (pipeline) on {device} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
