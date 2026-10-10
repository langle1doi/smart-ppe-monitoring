"""Validate a YOLO-format detection dataset and emit a resolved Ultralytics data YAML.

    python -m smart_ppe.data.validate_dataset --config configs/dataset.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from smart_ppe.utils.logging_utils import get_logger
from smart_ppe.utils.paths import (apply_overrides, ensure_dir, find_split_dir, list_images,
                                   load_yaml, resolve_dataset_root)

log = get_logger(__name__)


def read_class_names(data_yaml: Path) -> dict[int, str]:
    """Class id -> name from a Roboflow/Ultralytics data.yaml (`names` as list or dict)."""
    names = load_yaml(data_yaml).get("names")
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError(f"{data_yaml}: missing or malformed 'names'")


def validate_label_file(path: Path, num_classes: int) -> tuple[list[str], Counter]:
    """Return (error messages, class counts) for one YOLO .txt label file."""
    errors: list[str] = []
    counts: Counter = Counter()
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{ln} expected 5 fields (class cx cy w h), got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{ln} non-numeric value")
            continue
        if not 0 <= cls < num_classes:
            errors.append(f"{path.name}:{ln} class id {cls} outside [0,{num_classes - 1}]")
        elif w <= 0 or h <= 0:
            errors.append(f"{path.name}:{ln} non-positive box size w={w} h={h}")
        elif not (0 <= cx <= 1 and 0 <= cy <= 1):
            errors.append(f"{path.name}:{ln} center outside [0,1]")
        elif min(cx - w / 2, cy - h / 2) < -1e-3 or max(cx + w / 2, cy + h / 2) > 1 + 1e-3:
            errors.append(f"{path.name}:{ln} box extends outside image")
        else:
            counts[cls] += 1
    return errors, counts


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def validate_dataset(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the dataset under `root`; returns a JSON-serialisable report with errors/warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    data_yaml = root / cfg.get("data_yaml", "data.yaml")
    if not data_yaml.is_file():
        return {"errors": [f"data.yaml not found at {data_yaml}"], "warnings": [], "splits": {}}
    names = read_class_names(data_yaml)
    by_name = {n: i for i, n in names.items()}
    for role, cname in cfg.get("roles", {}).items():
        if cname not in by_name:
            errors.append(f"role '{role}' expects class '{cname}' but dataset classes are {sorted(by_name)}")

    report: dict[str, Any] = {"root": str(root), "classes": names, "splits": {}}
    hashes: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        sdir = find_split_dir(root, split)
        if sdir is None:
            errors.append(f"split '{split}' directory not found under {root} (need a train/valid/test export)")
            continue
        img_dir, lbl_dir = sdir / "images", sdir / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            errors.append(f"{sdir}: expected images/ and labels/ subfolders")
            continue
        images = list_images(img_dir)
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")}
        counts: Counter = Counter()
        n_bad = 0
        for lp in labels.values():
            e, c = validate_label_file(lp, len(names))
            errors.extend(e)
            n_bad += bool(e)
            counts.update(c)
        stems = {p.stem for p in images}
        orphan = sorted(set(labels) - stems)
        if orphan:
            errors.append(f"{split}: {len(orphan)} label files without image (e.g. {orphan[0]})")
        unlabeled = len(stems - set(labels))
        if unlabeled:
            warnings.append(f"{split}: {unlabeled} images without label file (treated as background)")
        minimum = cfg.get("min_images", {}).get(split, 1)
        if len(images) < minimum:
            errors.append(f"{split}: only {len(images)} images (< min_images {minimum})")
        report["splits"][split] = {
            "dir": str(sdir), "images": len(images), "label_files": len(labels),
            "background_images": unlabeled, "files_with_errors": n_bad,
            "boxes_per_class": {names[k]: v for k, v in sorted(counts.items())},
        }
        if cfg.get("check_leakage", True):
            hashes[split] = {_md5(p) for p in images}

    splits = list(hashes)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = hashes[a] & hashes[b]
            if shared:
                errors.append(f"data leakage: {len(shared)} byte-identical images in both {a} and {b}")
    report["errors"], report["warnings"] = errors, warnings
    return report


def write_resolved_yaml(root: Path, report: dict[str, Any], out_dir: Path) -> Path:
    """Write an Ultralytics data YAML with absolute paths (generated at runtime, not committed)."""
    ensure_dir(out_dir)
    s = report["splits"]
    data = {
        "path": str(root.resolve()),
        "train": str(Path(s["train"]["dir"]).resolve() / "images"),
        "val": str(Path(s["val"]["dir"]).resolve() / "images"),
        "test": str(Path(s["test"]["dir"]).resolve() / "images"),
        "names": {int(k): v for k, v in report["classes"].items()},
    }
    out = out_dir / "data_resolved.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def prepare_dataset(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate and return (resolved data yaml, report). Raises ValueError listing the first errors."""
    root = resolve_dataset_root(cfg)
    report = validate_dataset(root, cfg)
    out_dir = ensure_dir(cfg.get("output_dir", "outputs"))
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["errors"]:
        shown = "\n  ".join(report["errors"][:15])
        raise ValueError(f"Dataset validation failed ({len(report['errors'])} errors):\n  {shown}")
    return write_resolved_yaml(root, report, out_dir), report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the PPE dataset and write outputs/data_resolved.yaml")
    ap.add_argument("--config", default="configs/dataset.yaml", help="dataset config YAML")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set min_images.train=10")
    args = ap.parse_args(argv)
    try:
        cfg = apply_overrides(load_yaml(args.config), args.set)
        resolved, report = prepare_dataset(cfg)
    except (ValueError, FileNotFoundError) as e:
        log.error(str(e))
        return 1
    for split, info in report["splits"].items():
        log.info(f"{split}: {info['images']} images, boxes={info['boxes_per_class']}")
    for w in report["warnings"]:
        log.warning(w)
    log.info(f"OK. Resolved data YAML: {resolved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
