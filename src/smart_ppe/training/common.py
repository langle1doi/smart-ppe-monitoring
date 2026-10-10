"""Shared fine-tuning driver for YOLO and RT-DETR (identical data, seed, resolution, epochs; see configs)."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_ppe.data.validate_dataset import prepare_dataset
from smart_ppe.evaluation.evaluate import run_val
from smart_ppe.utils.device import free_gpu, peak_gpu_memory_mb, reset_peak_memory, select_device
from smart_ppe.utils.logging_utils import get_logger
from smart_ppe.utils.paths import apply_overrides, ensure_dir, load_yaml

log = get_logger(__name__)


def _set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _disable_integrations() -> None:
    """Turn off Ultralytics' optional logger integrations (Ray Tune, W&B, MLflow, ...). On Kaggle the preinstalled
    `ray` has a newer API than the Ultralytics callback expects and crashes training at the end of epoch 1."""
    from ultralytics import settings

    keys = [k for k in ("raytune", "wandb", "mlflow", "comet", "clearml", "neptune", "dvc", "hub") if k in settings]
    settings.update({k: False for k in keys})


def _build_model(family: str, weights: str) -> Any:
    from ultralytics import RTDETR, YOLO

    return {"yolo": YOLO, "rtdetr": RTDETR}[family](weights)


def run_training(cfg: dict[str, Any], ds_cfg: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune, then evaluate the best checkpoint on the *validation* split. Test split is never touched here."""
    import torch
    import ultralytics

    _disable_integrations()
    family, name = cfg["family"], cfg["name"]
    data_yaml, _ = prepare_dataset(ds_cfg)
    device = select_device(cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 0))
    _set_seeds(seed)

    runs_dir = ensure_dir(Path(cfg.get("runs_dir", "outputs/runs")).resolve())
    run_dir = ensure_dir(runs_dir / name)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train_config": cfg, "dataset_config": ds_cfg}),
                                         encoding="utf-8")
    shutil.copy(data_yaml, run_dir / "data_resolved.yaml")

    log.info(f"training {family} model={cfg['model']} device={device} seed={seed}")
    model = _build_model(family, cfg["model"])
    reset_peak_memory(device)
    t0 = time.perf_counter()
    model.train(data=str(data_yaml), project=str(runs_dir), name=name, exist_ok=True, seed=seed,
                device=device.replace("cuda:", "") if device != "cpu" else "cpu", **cfg["train"])
    train_seconds = time.perf_counter() - t0
    peak_mem = peak_gpu_memory_mb(device)

    best = Path(model.trainer.best)
    if not best.is_file():
        best = Path(model.trainer.last)
    del model
    free_gpu()

    val = run_val(family, str(best), str(data_yaml), "val", imgsz=cfg["train"].get("imgsz", 640),
                  batch=cfg["train"].get("batch", 8), device=device, out_dir=run_dir / "final_val", name=name)
    summary = {
        "name": name, "family": family, "pretrained": cfg["model"], "seed": seed, "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "train_seconds": round(train_seconds, 1), "epochs_requested": cfg["train"].get("epochs"),
        "peak_train_gpu_mem_mb": peak_mem, "best_weights": str(best),
        "model_size_mb": round(best.stat().st_size / 2**20, 2), "val_metrics": val,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"done: {best} train_seconds={summary['train_seconds']} val_map50_95={val['map50_95']:.4f}")
    return summary


def cli(family: str, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"Fine-tune the {family} detector on the PPE dataset")
    ap.add_argument("--config", default=f"configs/{family}.yaml", help="training config YAML")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override config (YAML value); prefix 'dataset.' targets the dataset config, "
                         "e.g. --set train.epochs=1 --set train.fraction=0.05 --set dataset.min_images.train=10")
    args = ap.parse_args(argv)
    try:
        cfg = load_yaml(args.config)
        if cfg.get("family") != family:
            raise ValueError(f"{args.config} has family '{cfg.get('family')}', expected '{family}'")
        apply_overrides(cfg, [s for s in args.set if not s.startswith("dataset.")])
        ds_cfg = load_yaml(cfg["dataset_config"])
        apply_overrides(ds_cfg, [s[len("dataset."):] for s in args.set if s.startswith("dataset.")])
        run_training(cfg, ds_cfg)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        return 1
    return 0
