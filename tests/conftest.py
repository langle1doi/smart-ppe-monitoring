import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "make_smoke_dataset", Path(__file__).resolve().parent.parent / "scripts" / "make_smoke_dataset.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)

DS_CFG = {"data_yaml": "data.yaml", "roles": {"helmet": "helmet", "head": "head", "person": "person"},
          "check_leakage": True, "min_images": {"train": 4, "val": 2, "test": 2}}


@pytest.fixture
def ds_cfg():
    return copy.deepcopy(DS_CFG)


@pytest.fixture
def dataset(tmp_path):
    """Tiny synthetic dataset in a folder whose name contains a space."""
    return smoke.make_dataset(tmp_path / "my data", {"train": 6, "valid": 3, "test": 3}, seed=1)


class FakeUltralytics:
    """Stands in for an Ultralytics model: returns one fixed helmet and one head detection."""
    names = {0: "helmet", 1: "head", 2: "person"}

    def predict(self, image, **kw):
        self.last_kwargs = kw
        boxes = SimpleNamespace(xyxy=np.array([[10, 10, 40, 40], [60, 12, 90, 44]], float),
                                conf=np.array([0.9, 0.6]), cls=np.array([0, 1]))
        return [SimpleNamespace(names=self.names, boxes=boxes,
                                speed={"preprocess": 1.0, "inference": 5.0, "postprocess": 0.5})]


@pytest.fixture
def fake_detector():
    from smart_ppe.models.base import UltralyticsDetector
    return UltralyticsDetector("fake.pt", model=FakeUltralytics(), imgsz=320)
