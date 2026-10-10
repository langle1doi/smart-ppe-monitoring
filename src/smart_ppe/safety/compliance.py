"""EXPERIMENTAL helmet-compliance analysis on top of detections. Independent of any model code.

Heuristic (bounding boxes only, so it cannot see occlusion or pose):
  * A helmet/head detection is *associated* with a person when >= `min_overlap` of its box area lies inside that
    person's head region (top `head_region_frac` of the person box, extended upward by `head_region_margin`).
    Helmets are NOT required to lie inside the full person box.
  * Unassociated helmet/head detections become standalone subjects (the dataset has no guarantee of a person box).
  * Evidence tiers: conf >= evidence_conf is evidence; weak_conf <= conf < evidence_conf is weak; below is ignored.
Verdicts:
  COMPLIANT                  strong helmet, no strong head.
  POTENTIALLY_NON_COMPLIANT  strong bare-head detection, no helmet evidence (weak or strong), subject large enough.
  UNCERTAIN                  anything else: no/weak evidence, conflicting helmet+head, tiny or top-cropped subject.
A missing helmet alone is never a violation; a bare head must be positively detected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from smart_ppe.models.base import Detection


class Status(str, Enum):
    COMPLIANT = "compliant"
    POTENTIALLY_NON_COMPLIANT = "potentially_non_compliant"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ComplianceConfig:
    helmet_class: str = "helmet"
    head_class: str = "head"
    person_class: str = "person"
    evidence_conf: float = 0.5
    weak_conf: float = 0.25
    head_region_frac: float = 0.4
    head_region_margin: float = 0.1
    min_overlap: float = 0.5
    min_person_height_px: float = 48
    min_head_height_px: float = 16

    def __post_init__(self) -> None:
        if not 0 <= self.weak_conf <= self.evidence_conf <= 1:
            raise ValueError("require 0 <= weak_conf <= evidence_conf <= 1")
        if not 0 < self.head_region_frac <= 1:
            raise ValueError("head_region_frac must be in (0,1]")
        if not 0 < self.min_overlap <= 1:
            raise ValueError("min_overlap must be in (0,1]")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ComplianceConfig":
        return cls(**(d or {}))


@dataclass
class Subject:
    status: Status
    reason: str
    box: tuple[float, float, float, float]       # person box if any, else the helmet/head box
    person: Detection | None = None
    helmets: list[Detection] = field(default_factory=list)
    heads: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "reason": self.reason, "box": list(self.box)}


def _inside_fraction(box: Sequence[float], region: Sequence[float]) -> float:
    """Fraction of `box` area that lies inside `region`."""
    ix = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return (ix * iy) / area if area > 0 else 0.0


def _head_region(person: Sequence[float], cfg: ComplianceConfig) -> tuple[float, float, float, float]:
    h = person[3] - person[1]
    return (person[0], person[1] - cfg.head_region_margin * h, person[2], person[1] + cfg.head_region_frac * h)


def _judge(helmets: list[Detection], heads: list[Detection], person: Detection | None,
           cfg: ComplianceConfig) -> tuple[Status, str]:
    strong_h = [d for d in helmets if d.conf >= cfg.evidence_conf]
    strong_b = [d for d in heads if d.conf >= cfg.evidence_conf]
    if strong_h and strong_b:
        return Status.UNCERTAIN, "conflicting helmet and bare-head detections"
    if strong_h:
        return Status.COMPLIANT, "helmet detected"
    if strong_b:
        if helmets:
            return Status.UNCERTAIN, "bare head detected but weak helmet evidence"
        head = max(strong_b, key=lambda d: d.conf)
        if head.xyxy[3] - head.xyxy[1] < cfg.min_head_height_px:
            return Status.UNCERTAIN, "bare head too small to judge"
        if person is not None and person.xyxy[3] - person.xyxy[1] < cfg.min_person_height_px:
            return Status.UNCERTAIN, "person too small to judge"
        if person is not None and person.xyxy[1] <= 1:
            return Status.UNCERTAIN, "person cropped at top of frame"
        return Status.POTENTIALLY_NON_COMPLIANT, "bare head detected, no helmet evidence"
    if helmets or heads:
        return Status.UNCERTAIN, "only low-confidence helmet/head evidence"
    return Status.UNCERTAIN, "no helmet/head evidence for this person"


def analyze_compliance(detections: Sequence[Detection], cfg: ComplianceConfig | None = None) -> list[Subject]:
    """Group detections into subjects and assign a cautious status to each."""
    cfg = cfg or ComplianceConfig()
    keep = [d for d in detections if d.conf >= cfg.weak_conf]
    persons = [d for d in keep if d.cls_name == cfg.person_class]
    gear = [d for d in keep if d.cls_name in (cfg.helmet_class, cfg.head_class)]

    assigned: dict[int, list[Detection]] = {i: [] for i in range(len(persons))}
    standalone: list[Detection] = []
    for g in gear:
        best, best_frac = None, cfg.min_overlap
        for i, p in enumerate(persons):
            frac = _inside_fraction(g.xyxy, _head_region(p.xyxy, cfg))
            if frac >= best_frac:
                best, best_frac = i, frac
        (assigned[best] if best is not None else standalone).append(g)

    subjects: list[Subject] = []
    for i, p in enumerate(persons):
        hel = [d for d in assigned[i] if d.cls_name == cfg.helmet_class]
        hd = [d for d in assigned[i] if d.cls_name == cfg.head_class]
        status, why = _judge(hel, hd, p, cfg)
        subjects.append(Subject(status, why, p.xyxy, p, hel, hd))
    for g in standalone:
        hel = [g] if g.cls_name == cfg.helmet_class else []
        hd = [g] if g.cls_name == cfg.head_class else []
        status, why = _judge(hel, hd, None, cfg)
        subjects.append(Subject(status, why + " (no person box)", g.xyxy, None, hel, hd))
    return subjects


def summarize(subjects: Sequence[Subject]) -> dict[str, int]:
    out = {s.value: 0 for s in Status}
    for s in subjects:
        out[s.status.value] += 1
    return out
