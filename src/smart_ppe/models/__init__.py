from __future__ import annotations

from smart_ppe.models.base import Detection, Timing, UltralyticsDetector

MODEL_FAMILIES = ("yolo", "rtdetr")


def load_detector(kind: str, weights: str, device: str = "cpu", imgsz: int = 640,
                  iou: float = 0.7, half: bool = False) -> UltralyticsDetector:
    """Build a detector by family name ('yolo' or 'rtdetr')."""
    if kind == "yolo":
        from smart_ppe.models.yolo_detector import YoloDetector as cls
    elif kind == "rtdetr":
        from smart_ppe.models.rtdetr_detector import RtdetrDetector as cls
    else:
        raise ValueError(f"Unknown model '{kind}'; choose from {MODEL_FAMILIES}")
    return cls(weights, device=device, imgsz=imgsz, iou=iou, half=half)


__all__ = ["Detection", "Timing", "UltralyticsDetector", "load_detector", "MODEL_FAMILIES"]
