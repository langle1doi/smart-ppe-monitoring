"""Shared detector interface. Both YOLO and RT-DETR run through Ultralytics, so one wrapper covers both;
subclasses only differ in which Ultralytics class builds the network."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]   # pixels, original image coordinates
    conf: float
    cls_id: int
    cls_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Timing:
    """Per-image times in milliseconds as reported by Ultralytics (letterbox+normalise / network / NMS+scaling)."""
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


def _np(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


class UltralyticsDetector:
    family = ""

    def __init__(self, weights: str | Path, device: str = "cpu", imgsz: int = 640,
                 iou: float = 0.7, half: bool = False, model: Any = None) -> None:
        """`model` lets tests inject a fake; otherwise weights are loaded via the subclass."""
        self.weights = str(weights)
        self.device, self.imgsz, self.iou, self.half = device, imgsz, iou, half
        if model is None:
            if not Path(self.weights).is_file():
                raise FileNotFoundError(f"Weights file not found: {self.weights}")
            model = self._build(self.weights)
        self.model = model

    def _build(self, weights: str) -> Any:  # pragma: no cover - needs ultralytics weights
        raise NotImplementedError

    def predict(self, image: np.ndarray, conf: float = 0.25) -> tuple[list[Detection], Timing]:
        """Run detection on one BGR image. Ultralytics letterboxes to `imgsz` internally (single resize)."""
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"conf must be in [0,1], got {conf}")
        if image is None or image.ndim != 3:
            raise ValueError("image must be an HxWx3 BGR array")
        res = self.model.predict(image, conf=conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, half=self.half, verbose=False)[0]
        names = res.names
        boxes = res.boxes
        xyxy, cf, cl = _np(boxes.xyxy), _np(boxes.conf), _np(boxes.cls).astype(int)
        dets = [Detection(tuple(float(v) for v in b), float(c), int(k), str(names[int(k)]))
                for b, c, k in zip(xyxy, cf, cl)]
        sp = res.speed
        return dets, Timing(float(sp["preprocess"]), float(sp["inference"]), float(sp["postprocess"]))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.model.parameters())
