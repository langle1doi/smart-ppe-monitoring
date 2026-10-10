import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from smart_ppe.evaluation.benchmark import benchmark_detector
from smart_ppe.evaluation.compare_models import build_rows, to_markdown
from smart_ppe.evaluation.evaluate import (error_analysis, iou_matrix, match_detections,
                                           parse_ultralytics_metrics, read_yolo_labels)
from smart_ppe.inference.predict_image import collect_images, predict_array
from smart_ppe.inference.predict_video import open_video, process_video
from smart_ppe.models import load_detector
from smart_ppe.models.base import Detection
from smart_ppe.safety.compliance import ComplianceConfig
from smart_ppe.utils.device import select_device
from smart_ppe.utils.paths import InputError, load_yaml

IMG = np.full((100, 120, 3), 128, np.uint8)


# ---- model config / detector wrapper -------------------------------------------------------
@pytest.mark.parametrize("name,family,model", [("yolo", "yolo", "yolo11"), ("rtdetr", "rtdetr", "rtdetr")])
def test_training_configs_consistent(name, family, model):
    cfg = load_yaml(f"configs/{name}.yaml")
    assert cfg["family"] == family and cfg["model"].startswith(model)
    base = load_yaml("configs/yolo.yaml")["train"]
    for k in ("imgsz", "epochs", "batch", "optimizer"):          # fairness: same budget/resolution
        assert cfg["train"][k] == base[k]
    assert load_yaml("configs/yolo.yaml")["seed"] == cfg["seed"]


def test_unknown_model_and_missing_weights():
    with pytest.raises(ValueError, match="Unknown model"):
        load_detector("resnet", "x.pt")
    with pytest.raises(FileNotFoundError):
        load_detector("yolo", "does_not_exist.pt")


def test_predict_structure_and_threshold(fake_detector):
    dets, t = fake_detector.predict(IMG, conf=0.3)
    assert [d.cls_name for d in dets] == ["helmet", "head"]
    assert all(isinstance(d, Detection) and len(d.xyxy) == 4 for d in dets)
    assert t.total_ms == pytest.approx(6.5)
    assert fake_detector.model.last_kwargs["conf"] == 0.3 and fake_detector.model.last_kwargs["imgsz"] == 320
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            fake_detector.predict(IMG, conf=bad)
    with pytest.raises(ValueError):
        fake_detector.predict(np.zeros((10, 10)), conf=0.5)


def test_select_device_cpu_and_invalid():
    assert select_device("cpu") == "cpu"
    assert select_device("auto") in ("cpu", "cuda:0")
    with pytest.raises(ValueError):
        select_device("tpu")


# ---- image / video inference ---------------------------------------------------------------
def test_predict_array_output_structure(fake_detector):
    r = predict_array(fake_detector, IMG, 0.25, ComplianceConfig())
    json.dumps({k: v for k, v in r.items() if k not in ("annotated", "_draw_args")})   # JSON-serialisable
    assert r["annotated"].shape == IMG.shape and set(r["timing_ms"]) == {"preprocess", "inference", "postprocess"}
    assert r["compliance"]["experimental"] is True
    assert predict_array(fake_detector, IMG, 0.25, None)["compliance"] is None


def test_collect_images_errors(tmp_path):
    with pytest.raises(InputError, match="not found"):
        collect_images(tmp_path / "nope.jpg")
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(InputError, match="Unsupported"):
        collect_images(tmp_path / "a.txt")
    with pytest.raises(InputError, match="No supported images"):
        collect_images(tmp_path)
    cv2.imwrite(str(tmp_path / "my img.png"), IMG)
    assert [p.name for p in collect_images(tmp_path)] == ["my img.png"]


def _make_video(path, n=7):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (120, 100))
    for i in range(n):
        w.write(np.full((100, 120, 3), i * 20, np.uint8))
    w.release()


def test_video_frame_handling(tmp_path, fake_detector):
    src = tmp_path / "in clip.avi"
    _make_video(src)
    out = tmp_path / "out.avi"
    s = process_video(fake_detector, src, out, comp_cfg=ComplianceConfig(), stride=2, codec="MJPG")
    assert s["frames"] == 7 and s["frames_detected"] == 4          # frames 0,2,4,6
    assert s["model_fps"] == pytest.approx(1000 / 6.5, rel=1e-3) and s["end_to_end_fps"] > 0
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
    s2 = process_video(fake_detector, src, out, max_frames=3, codec="MJPG")
    assert s2["frames"] == 3


def test_video_errors(tmp_path, fake_detector):
    with pytest.raises(InputError, match="not found"):
        open_video(tmp_path / "x.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(InputError):
        open_video(bad)
    (tmp_path / "v.xyz").write_text("")
    with pytest.raises(InputError, match="Unsupported"):
        open_video(tmp_path / "v.xyz")
    with pytest.raises(ValueError):
        process_video(fake_detector, bad, tmp_path / "o.avi", stride=0)


# ---- metrics / matching / reports ----------------------------------------------------------
def test_parse_ultralytics_metrics():
    box = SimpleNamespace(mp=0.8, mr=0.7, map50=0.75, map=0.5, ap_class_index=np.array([0, 2]),
                          p=np.array([0.9, 0.7]), r=np.array([0.8, 0.6]), ap50=np.array([0.85, 0.65]),
                          ap=np.array([0.55, 0.45]))
    m = parse_ultralytics_metrics(SimpleNamespace(box=box, speed={"inference": 3.0}), {0: "helmet", 2: "person"})
    assert m["map50_95"] == 0.5 and [c["class"] for c in m["per_class"]] == ["helmet", "person"]
    assert m["per_class"][1]["recall"] == 0.6 and m["speed_ms"]["inference"] == 3.0
    json.dumps(m)


def test_iou_and_matching():
    a = np.array([[0, 0, 10, 10.0]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[20, 20, 30, 30.0]]))[0, 0] == 0
    gt_b, gt_c = np.array([[0, 0, 10, 10.0], [50, 50, 80, 80.0]]), np.array([0, 1])
    preds = [Detection((0, 0, 10, 10), 0.9, 0, "helmet"),      # TP
             Detection((0, 0, 10, 10), 0.8, 0, "helmet"),      # duplicate -> FP
             Detection((50, 50, 80, 80), 0.7, 0, "helmet")]    # wrong class -> FP, and gt 1 is FN
    m = match_detections(gt_b, gt_c, preds, 0.5)
    assert (m["tp"], sorted(m["fp"]), m["fn"]) == ([0], [1, 2], [1])


def test_read_labels(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("1 0.5 0.5 0.5 0.5\n")
    b, c = read_yolo_labels(p, 100, 200)
    assert b.tolist() == [[25, 50, 75, 150]] and c.tolist() == [1]
    assert len(read_yolo_labels(tmp_path / "none.txt", 10, 10)[0]) == 0


def test_error_analysis_on_synthetic_split(dataset, fake_detector, tmp_path):
    names = {0: "helmet", 1: "head", 2: "person"}
    r = error_analysis(fake_detector, dataset / "valid" / "images", names, conf=0.25, out_dir=tmp_path / "ex",
                       max_examples=2)
    assert r["images"] == 3 and r["total"]["tp"] + r["total"]["fp"] == 6      # fake always emits 2 boxes
    assert r["total"]["fn"] > 0 and list((tmp_path / "ex").glob("fn_*.jpg"))


def test_benchmark_structure(fake_detector, tmp_path):
    w = tmp_path / "f.pt"
    w.write_bytes(b"0" * 1024)
    fake_detector.weights = str(w)
    fake_detector.model.model = SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 10)])
    r = benchmark_detector(fake_detector, [IMG] * 4, warmup=2, conf=0.25, device="cpu")
    assert r["images"] == 4 and r["fps_pipeline"] == pytest.approx(1000 / 6.5)
    assert r["includes_video_decoding"] is False and r["peak_gpu_mem_mb"] is None
    assert r["latency_ms"]["inference"]["mean"] == 5.0 and r["params_m"] == 0.0
    with pytest.raises(ValueError):
        benchmark_detector(fake_detector, [])


def test_comparison_report_from_json_with_pending(tmp_path):
    (tmp_path / "yolo_test.json").write_text(json.dumps(
        {"precision": 0.9, "recall": 0.8, "map50": 0.85, "map50_95": 0.55}))
    (tmp_path / "yolo_benchmark.json").write_text(json.dumps(
        {"latency_ms": {"pipeline_total": {"mean": 10.0}}, "fps_pipeline": 100.0, "fps_wall": 90.0,
         "model_size_mb": 50.0, "params_m": 25.0, "peak_gpu_mem_mb": None}))
    rows = build_rows(tmp_path, ["yolo", "rtdetr"])
    assert rows[0]["map50"] == 0.85 and rows[1]["map50"] is None
    md = to_markdown(rows)
    assert "0.850" in md and "100.0" in md and md.count("pending") >= 10      # rtdetr fully pending + yolo GPU mem


def test_inference_config_valid():
    cfg = load_yaml("configs/inference.yaml")
    ComplianceConfig.from_dict(cfg["compliance"])
    assert yaml.safe_load(open("configs/dataset.yaml"))["roles"]["helmet"] == "helmet"
