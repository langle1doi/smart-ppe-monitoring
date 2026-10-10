"""Benchmark both detectors under identical conditions and build comparison tables/plots from the JSON files.

    python -m smart_ppe.evaluation.compare_models --config configs/inference.yaml

Metrics JSON comes from `evaluate` (run it first on the test split); missing values are shown as 'pending'.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from smart_ppe.utils.device import select_device
from smart_ppe.utils.logging_utils import get_logger
from smart_ppe.utils.paths import apply_overrides, ensure_dir, load_yaml

log = get_logger(__name__)

COLUMNS = [("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP@50"), ("map50_95", "mAP@50:95"),
           ("latency_ms", "Latency ms (pre+inf+post)"), ("fps_pipeline", "FPS (model pipeline)"),
           ("fps_wall", "FPS (wall, per-image)"), ("model_size_mb", "Size MB"), ("params_m", "Params M"),
           ("peak_gpu_mem_mb", "Peak GPU MiB")]


def _read(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def build_rows(metrics_dir: Path, models: list[str], split: str = "test") -> list[dict[str, Any]]:
    """One row per model, merged from `<m>_<split>.json` and `<m>_benchmark.json`; absent values are None."""
    rows = []
    for m in models:
        ev, bm = _read(metrics_dir / f"{m}_{split}.json") or {}, _read(metrics_dir / f"{m}_benchmark.json") or {}
        lat = bm.get("latency_ms", {}).get("pipeline_total", {}).get("mean")
        rows.append({"model": m, "precision": ev.get("precision"), "recall": ev.get("recall"),
                     "map50": ev.get("map50"), "map50_95": ev.get("map50_95"), "latency_ms": lat,
                     "fps_pipeline": bm.get("fps_pipeline"), "fps_wall": bm.get("fps_wall"),
                     "model_size_mb": bm.get("model_size_mb"), "params_m": bm.get("params_m"),
                     "peak_gpu_mem_mb": bm.get("peak_gpu_mem_mb")})
    return rows


def _fmt(v: Any) -> str:
    return "pending" if v is None else (f"{v:.3f}" if isinstance(v, float) and abs(v) < 2 else f"{v:.1f}"
                                         if isinstance(v, float) else str(v))


def to_markdown(rows: list[dict[str, Any]]) -> str:
    head = "| Model | " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 1)
    body = ["| " + r["model"] + " | " + " | ".join(_fmt(r[k]) for k, _ in COLUMNS) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def per_class_table(metrics_dir: Path, models: list[str], split: str = "test") -> pd.DataFrame:
    frames = []
    for m in models:
        p = metrics_dir / f"{m}_{split}_per_class.csv"
        if p.is_file():
            df = pd.read_csv(p)
            df.insert(0, "model", m)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_comparison(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    for fname, keys, title in [("detection_metrics.png", ["precision", "recall", "map50", "map50_95"], "Detection"),
                               ("speed.png", ["latency_ms", "fps_pipeline"], "Speed")]:
        have = [r for r in rows if all(r[k] is not None for k in keys)]
        if not have:
            continue
        fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2))
        for ax, k in zip(axes, keys):
            ax.bar([r["model"] for r in have], [r[k] for r in have], color=["#4c78a8", "#f58518"][:len(have)])
            ax.set_title(k)
        fig.suptitle(title)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p)
    return out


def plot_per_class(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if df.empty:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = df.pivot(index="class", columns="model", values="map50_95")
    ax = piv.plot.bar(figsize=(6, 3.5), rot=0, title="Per-class mAP@50:95")
    p = out_dir / "per_class_map.png"
    ax.figure.tight_layout()
    ax.figure.savefig(p, dpi=150)
    plt.close(ax.figure)
    return p


def plot_training_curves(results_csv: Path, out_png: Path) -> Path | None:
    """Plot loss and metric columns from an Ultralytics results.csv."""
    if not results_csv.is_file():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    for c in df.columns:
        if c.startswith(("train/", "val/")):
            a.plot(df["epoch"], df[c], label=c)
        elif c.startswith("metrics/"):
            b.plot(df["epoch"], df[c], label=c)
    a.set_title("Losses")
    b.set_title("Validation metrics")
    for ax in (a, b):
        ax.set_xlabel("epoch")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark YOLO and RT-DETR and write comparison tables/plots")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--yolo-weights", help="override weights in the config")
    ap.add_argument("--rtdetr-weights", help="override weights in the config")
    ap.add_argument("--skip-benchmark", action="store_true", help="only rebuild tables/plots from existing JSON")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    args = ap.parse_args(argv)
    try:
        cfg = apply_overrides(load_yaml(args.config), args.set)
        mdir = ensure_dir(cfg.get("metrics_dir", "outputs/metrics"))
        models = list(cfg["models"])
        for m, w in (("yolo", args.yolo_weights), ("rtdetr", args.rtdetr_weights)):
            if w:
                cfg["models"][m]["weights"] = w
        if not args.skip_benchmark:
            from smart_ppe.evaluation.benchmark import load_split_images, run_benchmark
            device = select_device(cfg.get("device", "auto"))
            b = cfg.get("benchmark", {})
            images = load_split_images(load_yaml(cfg["dataset_config"]), b.get("split", "test"),
                                       b.get("num_images", 200))
            for m in models:     # sequential: one model on the GPU at a time
                res = run_benchmark(m, cfg["models"][m]["weights"], images, device, cfg.get("imgsz", 640),
                                    cfg.get("conf", 0.25), cfg.get("iou", 0.7), b.get("warmup", 20))
                (mdir / f"{m}_benchmark.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
                log.info(f"{m}: {res['fps_pipeline']:.1f} FPS")
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as e:
        log.error(f"{type(e).__name__}: {e}")
        return 1
    rows = build_rows(mdir, models)
    (mdir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(mdir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (mdir / "comparison.md").write_text(to_markdown(rows) + "\n", encoding="utf-8")
    plot_comparison(rows, mdir)
    plot_per_class(per_class_table(mdir, models), mdir)
    runs = Path(cfg.get("runs_dir", "outputs/runs"))
    for m in models:
        plot_training_curves(runs / m / "results.csv", mdir / f"{m}_training_curves.png")
    print(to_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
