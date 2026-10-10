"""Detection metrics (Ultralytics validator) + FP/FN error analysis with example images.

    python -m smart_ppe.evaluation.evaluate --model yolo --weights best.pt --split test

Metrics use conf=0.001 (standard for mAP). Error analysis uses the deployment threshold (--conf, default 0.25)
and class-aware greedy IoU matching at --iou-thr (0.5), so FP/FN counts are not comparable with mAP numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from smart_ppe.models.base import Detection
from smart_ppe.utils.device import free_gpu, select_device
from smart_ppe.utils.logging_utils import get_logger
from smart_ppe.utils.paths import apply_overrides, ensure_dir, list_images, load_yaml

log = get_logger(__name__)


def parse_ultralytics_metrics(res: Any, names: dict[int, str]) -> dict[str, Any]:
    """Convert an Ultralytics DetMetrics-like result into plain JSON-able numbers."""
    box = res.box
    per_class = []
    for i, cid in enumerate(np.asarray(box.ap_class_index).astype(int).tolist()):
        per_class.append({"class_id": cid, "class": names.get(cid, str(cid)),
                          "precision": float(box.p[i]), "recall": float(box.r[i]),
                          "map50": float(box.ap50[i]), "map50_95": float(box.ap[i])})
    return {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50),
            "map50_95": float(box.map), "per_class": per_class,
            "speed_ms": {k: float(v) for k, v in dict(getattr(res, "speed", {})).items()}}


def run_val(kind: str, weights: str, data_yaml: str, split: str, imgsz: int = 640, batch: int = 8,
            device: str = "cpu", out_dir: str | Path = "outputs/eval", name: str = "model",
            conf: float = 0.001) -> dict[str, Any]:
    """Run the Ultralytics validator on `split` ('val' or 'test') and parse the metrics."""
    from ultralytics import RTDETR, YOLO

    if not Path(weights).is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")
    model = {"yolo": YOLO, "rtdetr": RTDETR}[kind](weights)
    out_dir = Path(out_dir).resolve()
    res = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch, conf=conf, plots=True,
                    device=device.replace("cuda:", "") if device != "cpu" else "cpu",
                    project=str(out_dir.parent), name=out_dir.name, exist_ok=True, verbose=False)
    metrics = parse_ultralytics_metrics(res, {int(k): v for k, v in model.names.items()})
    metrics["plots_dir"] = str(out_dir)
    del model
    free_gpu()
    return metrics


# ---------------------------------------------------------------- error analysis (pure numpy) ----

def read_yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO txt -> (xyxy pixel boxes [N,4], class ids [N]); missing file means no objects."""
    if not label_path.is_file():
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    rows = [ln.split() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    a = np.array([[float(v) for v in r[:5]] for r in rows])
    cx, cy, w, h = a[:, 1] * width, a[:, 2] * height, a[:, 3] * width, a[:, 4] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1), a[:, 0].astype(int)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_detections(gt_boxes: np.ndarray, gt_cls: np.ndarray, preds: Sequence[Detection],
                     iou_thr: float = 0.5) -> dict[str, list[int]]:
    """Class-aware greedy matching, highest-confidence predictions first. Returns tp/fp pred idx and fn gt idx."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i].conf)
    pb = np.array([preds[i].xyxy for i in range(len(preds))]).reshape(-1, 4)
    ious = iou_matrix(pb, gt_boxes)
    used: set[int] = set()
    tp, fp = [], []
    for i in order:
        best, best_iou = -1, iou_thr
        for j in range(len(gt_cls)):
            if j not in used and gt_cls[j] == preds[i].cls_id and ious[i, j] >= best_iou:
                best, best_iou = j, ious[i, j]
        if best >= 0:
            used.add(best)
            tp.append(i)
        else:
            fp.append(i)
    return {"tp": tp, "fp": fp, "fn": [j for j in range(len(gt_cls)) if j not in used]}


def _annotate(img: np.ndarray, preds: Sequence[Detection], m: dict[str, list[int]],
              gt_boxes: np.ndarray, gt_cls: np.ndarray, names: dict[int, str]) -> np.ndarray:
    out = img.copy()

    def box(xyxy, color, text):
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for i in m["tp"]:
        box(preds[i].xyxy, (0, 200, 0), f"TP {preds[i].cls_name} {preds[i].conf:.2f}")
    for i in m["fp"]:
        box(preds[i].xyxy, (0, 0, 230), f"FP {preds[i].cls_name} {preds[i].conf:.2f}")
    for j in m["fn"]:
        box(gt_boxes[j], (230, 120, 0), f"FN {names.get(int(gt_cls[j]), gt_cls[j])}")
    return out


def error_analysis(detector: Any, image_dir: Path, names: dict[int, str], conf: float = 0.25,
                   iou_thr: float = 0.5, out_dir: Path | None = None, max_examples: int = 8,
                   max_images: int | None = None) -> dict[str, Any]:
    """Count TP/FP/FN per class at a fixed threshold and save the worst FP / FN and clean TP example images."""
    label_dir = image_dir.parent / "labels"
    totals = {n: {"tp": 0, "fp": 0, "fn": 0} for n in names.values()}
    per_image = []
    images = list_images(image_dir)[:max_images]
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            log.warning(f"unreadable image skipped: {p}")
            continue
        gt_b, gt_c = read_yolo_labels(label_dir / f"{p.stem}.txt", img.shape[1], img.shape[0])
        preds, _ = detector.predict(img, conf=conf)
        m = match_detections(gt_b, gt_c, preds, iou_thr)
        for i in m["tp"]:
            totals[preds[i].cls_name]["tp"] += 1
        for i in m["fp"]:
            totals[preds[i].cls_name]["fp"] += 1
        for j in m["fn"]:
            totals[names[int(gt_c[j])]]["fn"] += 1
        per_image.append((p, img, preds, m, gt_b, gt_c))

    if out_dir is not None and per_image:
        ensure_dir(out_dir)
        picks = {"fp": sorted((r for r in per_image if r[3]["fp"]), key=lambda r: (-len(r[3]["fp"]), r[0].name)),
                 "fn": sorted((r for r in per_image if r[3]["fn"]), key=lambda r: (-len(r[3]["fn"]), r[0].name)),
                 "tp": sorted((r for r in per_image if r[3]["tp"] and not r[3]["fp"] and not r[3]["fn"]),
                              key=lambda r: (-len(r[3]["tp"]), r[0].name))}
        for kind, rows in picks.items():
            for p, img, preds, m, gt_b, gt_c in rows[:max_examples]:
                cv2.imwrite(str(out_dir / f"{kind}_{p.stem}.jpg"), _annotate(img, preds, m, gt_b, gt_c, names))
    return {"conf": conf, "iou_thr": iou_thr, "images": len(per_image), "per_class": totals,
            "total": {k: sum(v[k] for v in totals.values()) for k in ("tp", "fp", "fn")}}


# ------------------------------------------------------------------------------------- CLI ----

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained detector (metrics + FP/FN analysis)")
    ap.add_argument("--model", required=True, choices=["yolo", "rtdetr"])
    ap.add_argument("--weights", required=True, help="path to .pt checkpoint")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="use 'val' for model selection; 'test' only for the final report")
    ap.add_argument("--name", help="result name (default: model family)")
    ap.add_argument("--config", default="configs/dataset.yaml", help="dataset config YAML")
    ap.add_argument("--data", help="reuse an existing resolved data YAML (skips dataset validation)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25, help="threshold for FP/FN analysis")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--metrics-dir", default="outputs/metrics")
    ap.add_argument("--no-error-analysis", action="store_true")
    ap.add_argument("--max-images", type=int, help="limit images in error analysis")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dataset config override")
    args = ap.parse_args(argv)
    try:
        device = select_device(args.device)
        name = args.name or args.model
        if args.data:
            data_yaml = args.data
        else:
            from smart_ppe.data.validate_dataset import prepare_dataset
            data_yaml = str(prepare_dataset(apply_overrides(load_yaml(args.config), args.set))[0])
        mdir = ensure_dir(args.metrics_dir)
        out_dir = Path(args.metrics_dir).parent / "eval" / f"{name}_{args.split}"
        result = {"name": name, "model": args.model, "weights": args.weights, "split": args.split,
                  "imgsz": args.imgsz, "device": device}
        result.update(run_val(args.model, args.weights, data_yaml, args.split, args.imgsz, args.batch,
                              device, out_dir, name))
        if not args.no_error_analysis:
            from smart_ppe.models import load_detector
            ds = load_yaml(data_yaml)
            detector = load_detector(args.model, args.weights, device, args.imgsz)
            names = {int(k): v for k, v in ds["names"].items()}
            result["error_analysis"] = error_analysis(detector, Path(ds[args.split]), names, args.conf,
                                                      args.iou_thr, out_dir / "examples",
                                                      max_images=args.max_images)
            del detector
            free_gpu()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        return 1
    stem = mdir / f"{name}_{args.split}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(f"{stem}_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class", "precision", "recall", "map50", "map50_95"])
        w.writeheader()
        w.writerows(result["per_class"])
    log.info(f"{name}/{args.split}: P={result['precision']:.3f} R={result['recall']:.3f} "
             f"mAP50={result['map50']:.3f} mAP50-95={result['map50_95']:.3f} -> {stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
