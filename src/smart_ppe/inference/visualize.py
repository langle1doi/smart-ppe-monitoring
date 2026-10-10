"""Drawing helpers (kept apart from inference and from safety logic)."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from smart_ppe.models.base import Detection
from smart_ppe.safety.compliance import Status, Subject

_PALETTE = [(60, 180, 75), (230, 120, 30), (200, 60, 200), (40, 200, 230), (180, 180, 40)]
_STATUS_STYLE = {Status.COMPLIANT: ((0, 170, 0), "compliant"),
                 Status.POTENTIALLY_NON_COMPLIANT: ((0, 0, 230), "POTENTIAL VIOLATION"),
                 Status.UNCERTAIN: ((0, 190, 230), "uncertain")}


def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 2, y + 2), color, -1)
    cv2.putText(img, text, (x + 1, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw(image: np.ndarray, detections: Sequence[Detection], subjects: Sequence[Subject] | None = None,
         info: str | None = None) -> np.ndarray:
    """Return a copy with class boxes + confidences, optional subject status boxes, optional info banner."""
    out = image.copy()
    for s in subjects or []:
        color, text = _STATUS_STYLE[s.status]
        x1, y1, x2, y2 = (int(v) for v in s.box)
        cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 3)
        _label(out, text, x1 - 3, y2 + 18, color)
    for d in detections:
        color = _PALETTE[d.cls_id % len(_PALETTE)]
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _label(out, f"{d.cls_name} {d.conf:.2f}", x1, y1, color)
    if info:
        _label(out, info, 4, 18, (40, 40, 40))
    return out
