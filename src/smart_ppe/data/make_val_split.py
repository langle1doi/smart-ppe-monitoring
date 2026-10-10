"""Carve a validation split out of `train` when a dataset export only ships train + test.

The Roboflow "Hard Hat Workers" v2 export is a 75/25 train/test split with no validation set, but the protocol
needs one (model selection on val, test reserved for the final report). The original `test` split is copied
untouched; `valid` is a seeded random sample of the original `train`, so splits stay disjoint.

    python -m smart_ppe.data.make_val_split --src /kaggle/input/hard-hat-workers --dst /kaggle/working/hardhat
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

from smart_ppe.utils.logging_utils import get_logger
from smart_ppe.utils.paths import find_split_dir, list_images

log = get_logger(__name__)


def _copy(images: list[Path], label_dir: Path, dst: Path) -> None:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, dst / "images" / img.name)
        lbl = label_dir / f"{img.stem}.txt"
        if lbl.is_file():
            shutil.copy2(lbl, dst / "labels" / lbl.name)


def make_val_split(src: Path, dst: Path, val_frac: float = 0.15, seed: int = 0) -> dict[str, int]:
    """Write dst/{train,valid,test}; returns image counts. If src already has a valid split it is just copied."""
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0,1)")
    if dst.resolve() == src.resolve():
        raise ValueError("--dst must differ from --src (originals are never modified)")
    tr, te = find_split_dir(src, "train"), find_split_dir(src, "test")
    if tr is None or te is None:
        raise FileNotFoundError(f"{src} needs train/ and test/ folders (found train={tr}, test={te})")
    va = find_split_dir(src, "val")
    train_imgs = list_images(tr / "images")
    if va is not None:
        log.warning("source already has a validation split; keeping the original splits")
        parts = {"train": (train_imgs, tr), "valid": (list_images(va / "images"), va)}
    else:
        rng = random.Random(seed)
        shuffled = sorted(train_imgs, key=lambda p: p.name)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_frac))
        parts = {"valid": (sorted(shuffled[:n_val]), tr), "train": (sorted(shuffled[n_val:]), tr)}
    parts["test"] = (list_images(te / "images"), te)
    counts = {}
    for name, (imgs, sdir) in parts.items():
        _copy(imgs, sdir / "labels", dst / name)
        counts[name] = len(imgs)
    names = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))["names"]
    (dst / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}), encoding="utf-8")
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create a seeded validation split from train (test left untouched)")
    ap.add_argument("--src", required=True, help="dataset root with data.yaml, train/, test/")
    ap.add_argument("--dst", required=True, help="output root (must differ from --src)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        counts = make_val_split(Path(args.src), Path(args.dst), args.val_frac, args.seed)
    except (ValueError, FileNotFoundError, KeyError) as e:
        log.error(f"{type(e).__name__}: {e}")
        return 1
    log.info(f"wrote {args.dst}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
