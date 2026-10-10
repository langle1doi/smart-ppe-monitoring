import pytest

from smart_ppe.models.base import Detection
from smart_ppe.safety.compliance import ComplianceConfig, Status, analyze_compliance, summarize

CLS = {"helmet": 0, "head": 1, "person": 2}


def D(name, box, conf=0.9):
    return Detection(tuple(float(v) for v in box), conf, CLS[name], name)


PERSON = (100, 100, 200, 400)          # head region: y in [70, 220]
HEAD_BOX = (130, 105, 170, 145)


def one(dets, **cfg):
    out = analyze_compliance(dets, ComplianceConfig(**cfg))
    assert len(out) == 1
    return out[0]


def test_helmet_in_head_region_is_compliant():
    assert one([D("person", PERSON), D("helmet", HEAD_BOX)]).status == Status.COMPLIANT


def test_helmet_slightly_above_person_box_still_associated():
    s = one([D("person", PERSON), D("helmet", (130, 80, 170, 120))])
    assert s.status == Status.COMPLIANT and s.person is not None


def test_bare_head_is_potentially_non_compliant():
    assert one([D("person", PERSON), D("head", HEAD_BOX)]).status == Status.POTENTIALLY_NON_COMPLIANT


def test_person_alone_is_uncertain_not_violation():
    s = one([D("person", PERSON)])
    assert s.status == Status.UNCERTAIN and "no helmet/head evidence" in s.reason


def test_low_confidence_head_is_uncertain():
    assert one([D("person", PERSON), D("head", HEAD_BOX, 0.35)]).status == Status.UNCERTAIN


def test_below_weak_conf_ignored():
    assert analyze_compliance([D("head", HEAD_BOX, 0.1)]) == []


def test_conflicting_evidence_uncertain():
    s = one([D("person", PERSON), D("head", HEAD_BOX), D("helmet", HEAD_BOX)])
    assert s.status == Status.UNCERTAIN and "conflicting" in s.reason


def test_bare_head_with_weak_helmet_uncertain():
    assert one([D("person", PERSON), D("head", HEAD_BOX), D("helmet", HEAD_BOX, 0.3)]).status == Status.UNCERTAIN


def test_tiny_person_not_judged_non_compliant():
    s = one([D("person", (100, 100, 120, 130)), D("head", (102, 100, 118, 112))])
    assert s.status == Status.UNCERTAIN


def test_person_cropped_at_top_uncertain():
    s = one([D("person", (100, 0, 200, 300)), D("head", (130, 2, 170, 42))])
    assert s.status == Status.UNCERTAIN and "cropped" in s.reason


def test_standalone_helmet_and_head_without_person():
    out = analyze_compliance([D("helmet", (10, 10, 50, 50)), D("head", (300, 10, 340, 50))])
    assert {s.status for s in out} == {Status.COMPLIANT, Status.POTENTIALLY_NON_COMPLIANT}
    assert all("no person box" in s.reason for s in out)


def test_helmet_far_from_head_region_not_associated():
    out = analyze_compliance([D("person", PERSON), D("helmet", (130, 330, 170, 370))])
    assert len(out) == 2 and out[0].status == Status.UNCERTAIN       # person has no evidence


def test_two_workers_each_get_own_helmet():
    p2 = (400, 100, 500, 400)
    out = analyze_compliance([D("person", PERSON), D("person", p2), D("helmet", HEAD_BOX),
                              D("head", (430, 105, 470, 145))])
    assert summarize(out) == {"compliant": 1, "potentially_non_compliant": 1, "uncertain": 0}


def test_thresholds_configurable():
    dets = [D("person", PERSON), D("head", HEAD_BOX, 0.6)]
    assert one(dets).status == Status.POTENTIALLY_NON_COMPLIANT
    assert one(dets, evidence_conf=0.7).status == Status.UNCERTAIN


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ComplianceConfig(weak_conf=0.8, evidence_conf=0.5)
