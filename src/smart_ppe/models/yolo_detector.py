"""YOLO baseline (CNN backbone + neck + anchor-free head, NMS post-processing)."""
from __future__ import annotations

from typing import Any

from smart_ppe.models.base import UltralyticsDetector


class YoloDetector(UltralyticsDetector):
    family = "yolo"

    def _build(self, weights: str) -> Any:
        from ultralytics import YOLO

        return YOLO(weights)
