"""RT-DETR: real-time DETR object detector (Ultralytics implementation).

Architecture in brief: a CNN backbone produces multi-scale features; an efficient *hybrid encoder* applies
self-attention (a Transformer) only to the lowest-resolution scale and fuses scales with cross-scale CNN-style
blocks; a small set of top-scoring encoder features initialises the object queries of a Transformer decoder,
which attends to the image features and directly predicts a fixed set of boxes + classes. Training uses
bipartite (Hungarian) matching, so predictions are one-to-one with objects and **no NMS is needed** at inference.
This is an object *detection* architecture - not a Vision Transformer image classifier.
"""
from __future__ import annotations

from typing import Any

from smart_ppe.models.base import UltralyticsDetector


class RtdetrDetector(UltralyticsDetector):
    family = "rtdetr"

    def _build(self, weights: str) -> Any:
        from ultralytics import RTDETR

        return RTDETR(weights)
