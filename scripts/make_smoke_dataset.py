"""Generate a tiny SYNTHETIC YOLO-format dataset (helmet/head/person) to smoke-test the pipeline.

Images are drawn shapes, not real photos: metrics from them are meaningless and must never be reported as results.

    python scripts/make_smoke_dataset.py --out "smoke data" --train 24 --val 8 --test 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

NAMES = ["helmet", "head", "person"]


def _worker(img: np.ndarray, rng: np.random.Generator, helmet: bool) -> list[str]:
    """Draw one stick-figure worker; return its 3 YOLO label lines (helmet-or-head, person)."""
    h, w = img.shape[:2]
    pw = int(rng.integers(w // 8, w // 5))
    ph = int(pw * rng.uniform(2.2, 3.0))
    x1 = int(rng.integers(5, w - pw - 5))
    y1 = int(rng.integers(5, max(6, h - ph - 5)))
    cv2.rectangle(img, (x1, y1), (x1 + pw, y1 + ph), (90, 90, 160), -1)              # body
    r = pw // 3
    c = (x1 + pw // 2, y1 + r + 2)
    cv2.circle(img, c, r, (150, 190, 230), -1)                                          # head
    if helmet:
        cv2.ellipse(img, c, (r + 2, r + 2), 0, 180, 360, (0, 220, 255), -1)             # yellow cap
    hb = (c[0] - r - 2, c[1] - r - 2, c[0] + r + 2, c[1] + (0 if helmet else r))        # head/helmet box
    if not helmet:
        hb = (c[0] - r, c[1] - r, c[0] + r, c[1] + r)

    def yolo(cls: int, b: tuple[int, int, int, int]) -> str:
        bx1, by1, bx2, by2 = max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)
        return (f"{cls} {(bx1 + bx2) / 2 / w:.6f} {(by1 + by2) / 2 / h:.6f} "
                f"{(bx2 - bx1) / w:.6f} {(by2 - by1) / h:.6f}")

    return [yolo(0 if helmet else 1, hb), yolo(2, (x1, y1, x1 + pw, y1 + ph))]


def make_dataset(root: Path, sizes: dict[str, int], seed: int = 0, size: int = 320) -> Path:
    """Write root/{train,valid,test}/{images,labels} + data.yaml. Distinct RNG streams => no cross-split duplicates."""
    for si, (split, n) in enumerate(sizes.items()):
        rng = np.random.default_rng([seed, si])
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((size, size, 3), rng.integers(100, 200), np.uint8)
            img = cv2.add(img, rng.integers(0, 40, img.shape, dtype=np.uint8))
            lines = []
            for _ in range(int(rng.integers(1, 4))):
                lines += _worker(img, rng, helmet=bool(rng.integers(0, 2)))
            cv2.imwrite(str(root / split / "images" / f"{split}_{i:04d}.jpg"), img)
            (root / split / "labels" / f"{split}_{i:04d}.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"names": NAMES, "nc": len(NAMES)}), encoding="utf-8")
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=24)
    ap.add_argument("--val", type=int, default=8)
    ap.add_argument("--test", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make_dataset(Path(a.out), {"train": a.train, "valid": a.val, "test": a.test}, a.seed)
    print(f"synthetic smoke dataset written to {a.out} (NOT real data)")


if __name__ == "__main__":
    main()
