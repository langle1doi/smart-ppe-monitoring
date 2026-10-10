"""Video inference: decode -> detect -> (experimental) compliance -> annotate -> write, with FPS statistics.

    python -m smart_ppe.inference.predict_video --model rtdetr --weights best.pt --source clip.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

from smart_ppe.inference.predict_image import predict_array
from smart_ppe.inference.visualize import draw
from smart_ppe.safety.compliance import ComplianceConfig
from smart_ppe.utils.device import select_device
from smart_ppe.utils.logging_utils import get_logger
from smart_ppe.utils.paths import VIDEO_EXTS, InputError, ensure_dir, load_yaml

log = get_logger(__name__)


def open_video(source: str | Path) -> cv2.VideoCapture:
    p = Path(source)
    if not p.is_file():
        raise InputError(f"Video not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        raise InputError(f"Unsupported video format '{p.suffix}' (supported: {sorted(VIDEO_EXTS)})")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise InputError(f"OpenCV could not open video (corrupt or missing codec): {p}")
    return cap


def process_video(detector: Any, source: str | Path, out_path: str | Path, conf: float = 0.25,
                  comp_cfg: ComplianceConfig | None = None, stride: int = 1, max_frames: int | None = None,
                  codec: str = "mp4v") -> dict[str, Any]:
    """Process a video; every `stride`-th frame is detected, others reuse the last annotation. Returns stats.

    `end_to_end_fps` includes decoding, inference, drawing and encoding; `model_fps` is model pipeline only.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cap = open_video(source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps_in, (w, h))
    if not writer.isOpened():
        cap.release()
        raise InputError(f"Cannot open video writer for {out_path} (codec '{codec}')")
    frames = detected = 0
    totals: list[float] = []
    summary_counts: dict[str, int] = {}
    last: tuple | None = None
    t0 = time.perf_counter()
    try:
        while max_frames is None or frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frames % stride == 0 or last is None:
                r = predict_array(detector, frame, conf, comp_cfg)
                t = r["timing_ms"]
                totals.append(t["preprocess"] + t["inference"] + t["postprocess"])
                dets, subs, _ = r["_draw_args"]
                # overlay shows the running model-only rate (stable) rather than per-frame noise
                last = (dets, subs, f"{detector.family} {totals[-1]:.1f} ms | model "
                                    f"{1000 * len(totals) / sum(totals):.1f} FPS")
                detected += 1
                if r["compliance"]:
                    for k, v in r["compliance"]["summary"].items():
                        summary_counts[k] = summary_counts.get(k, 0) + v
            writer.write(draw(frame, *last))
            frames += 1
    finally:
        cap.release()
        writer.release()
    wall = time.perf_counter() - t0
    if frames == 0:
        raise InputError(f"No frames could be read from {source}")
    return {"source": str(source), "output": str(out_path), "frames": frames, "frames_detected": detected,
            "stride": stride, "wall_seconds": round(wall, 3),
            "end_to_end_fps": round(frames / wall, 2),
            "model_fps": round(1000 * len(totals) / sum(totals), 2) if totals else None,
            "mean_model_ms": round(sum(totals) / len(totals), 2) if totals else None,
            "end_to_end_includes": "decode+inference+compliance+drawing+encode",
            "compliance_subject_counts_over_detected_frames": summary_counts or None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run detector (+ experimental compliance) on a video")
    ap.add_argument("--model", required=True, choices=["yolo", "rtdetr"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="video file (.mp4/.avi/.mov/.mkv)")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default="outputs/predictions")
    ap.add_argument("--stride", type=int, default=1, help="run the model on every Nth frame")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--codec", default="mp4v", help="FourCC for the output writer")
    ap.add_argument("--no-compliance", action="store_true")
    args = ap.parse_args(argv)
    try:
        from smart_ppe.models import load_detector

        cfg = load_yaml(args.config)
        src = Path(args.source)
        open_video(src).release()          # fail fast before loading a model
        det = load_detector(args.model, args.weights, select_device(args.device or cfg.get("device", "auto")),
                            args.imgsz or cfg.get("imgsz", 640), cfg.get("iou", 0.7))
        out = ensure_dir(args.output_dir) / f"{src.stem}_{args.model}.mp4"
        comp = None if args.no_compliance else ComplianceConfig.from_dict(cfg.get("compliance"))
        stats = process_video(det, src, out, args.conf if args.conf is not None else cfg.get("conf", 0.25),
                              comp, args.stride, args.max_frames, args.codec)
    except (InputError, ValueError, FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        return 2
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info(f"{stats['frames']} frames, end-to-end {stats['end_to_end_fps']} FPS, model {stats['model_fps']} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
