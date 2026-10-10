"""Config / path helpers. Dataset root comes from env PPE_DATASET_ROOT or the YAML, never hardcoded."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

SPLIT_DIR_ALIASES = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


class InputError(ValueError):
    """User-supplied input (image/video path or file) is missing, unsupported or unreadable."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} must contain a YAML mapping")
    return data


def apply_overrides(cfg: dict[str, Any], items: list[str]) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides (value parsed as YAML) in place."""
    for item in items:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        *path, last = key.split(".")
        node = cfg
        for k in path:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(val)
    return cfg


def resolve_dataset_root(cfg: dict[str, Any]) -> Path:
    raw = os.environ.get("PPE_DATASET_ROOT") or cfg.get("root")
    if not raw:
        raise ValueError("Dataset root not set: export PPE_DATASET_ROOT or set `root` in configs/dataset.yaml")
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")
    return root


def find_split_dir(root: Path, split: str) -> Path | None:
    for name in SPLIT_DIR_ALIASES[split]:
        if (root / name).is_dir():
            return root / name
    return None


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
