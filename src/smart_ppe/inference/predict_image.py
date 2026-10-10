"""Image inference with optional (experimental) compliance analysis.

    python -m smart_ppe.inference.predict_image --model yolo --weights best.pt --source "my images"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from smart_ppe.inference.visualize import draw
from smart_ppe.safety.compliance import ComplianceConfig, analyze_compliance, summarize
from smart_ppe.utils.device import select_device
from smart_ppe.utils.logging_utils import get_logger
from smart_ppe.utils.paths import IMAGE_EXTS, InputError, ensure_dir, list_images, load_yaml

log = get_logger(__name__)


def collect_images(source: str | Path) -> list[Path]:
    """A single image file or a directory of images; raises InputError otherwise."""
    p = Path(source)
    if not p.exists():
        raise InputError(f"Source not found: {p}")
    if p.is_dir():
        files = list_images(p)
        if not files:
            raise InputError(f"No supported images ({sorted(IMAGE_EXTS)}) in {p}")
        return files
    if p.suffix.lower() not in IMAGE_EXTS:
        raise InputError(f"Unsupported image format '{p.suffix}' (supported: {sorted(IMAGE_EXTS)})")
    return [p]


def predict_array(detector: Any, image: np.ndarray, conf: float,
                  comp_cfg: ComplianceConfig | None) -> dict[str, Any]:
    """Detect (+ analyse compliance if comp_cfg) on one BGR image; returns structured result + annotated image."""
    dets, timing = detector.predict(image, conf=conf)
    subjects = analyze_compliance(dets, comp_cfg) if comp_cfg else None
    info = f"{detector.family} {timing.total_ms:.1f} ms ({1000 / max(timing.total_ms, 1e-6):.0f} FPS model-only)"
    return {
        "detections": [d.to_dict() for d in dets],
        "timing_ms": {"preprocess": timing.preprocess_ms, "inference": timing.inference_ms,
                      "postprocess": timing.postprocess_ms},
        "compliance": None if subjects is None else
        {"experimental": True, "summary": summarize(subjects), "subjects": [s.to_dict() for s in subjects]},
        "annotated": draw(image, dets, subjects, info),
        "_draw_args": (dets, subjects, info),    # lets video code re-draw without re-analysing
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run detector (+ experimental compliance) on image(s)")
    ap.add_argument("--model", required=True, choices=["yolo", "rtdetr"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--conf", type=float, default=None, help="confidence threshold (default from config)")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default="outputs/predictions")
    ap.add_argument("--no-compliance", action="store_true", help="detection only")
    args = ap.parse_args(argv)
    try:
        from smart_ppe.models import load_detector

        cfg = load_yaml(args.config)
        conf = args.conf if args.conf is not None else cfg.get("conf", 0.25)
        files = collect_images(args.source)
        det = load_detector(args.model, args.weights, select_device(args.device or cfg.get("device", "auto")),
                            args.imgsz or cfg.get("imgsz", 640), cfg.get("iou", 0.7))
        comp = None if args.no_compliance else ComplianceConfig.from_dict(cfg.get("compliance"))
        out_dir = ensure_dir(args.output_dir)
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                log.warning(f"cannot decode, skipped: {f}")
                continue
            r = predict_array(det, img, conf, comp)
            cv2.imwrite(str(out_dir / f"{f.stem}_{args.model}.jpg"), r.pop("annotated"))
            r.pop("_draw_args")
            (out_dir / f"{f.stem}_{args.model}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
            log.info(f"{f.name}: {len(r['detections'])} detections -> {out_dir}")
    except (InputError, ValueError, FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
