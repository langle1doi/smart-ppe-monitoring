"""Streamlit demo: run YOLO and/or RT-DETR on an example or uploaded image, with experimental compliance analysis.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))      # lets the app run without `pip install -e .`

from smart_ppe.inference.predict_image import predict_array  # noqa: E402
from smart_ppe.models import load_detector  # noqa: E402
from smart_ppe.safety.compliance import ComplianceConfig  # noqa: E402
from smart_ppe.utils.device import select_device  # noqa: E402
from smart_ppe.utils.paths import IMAGE_EXTS, load_yaml  # noqa: E402

MODELS = {"YOLO11-L": ("yolo", "outputs/runs/yolo/weights/best.pt"),
          "RT-DETR-L": ("rtdetr", "outputs/runs/rtdetr/weights/best.pt")}
STATUS_LABEL = {"compliant": "compliant", "potentially_non_compliant": "POTENTIAL VIOLATION", "uncertain": "uncertain"}

st.set_page_config(page_title="Smart PPE Monitoring", page_icon="🦺", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_detector(kind: str, weights: str, device: str):
    return load_detector(kind, weights, device, imgsz=640, iou=0.7)


@st.cache_data
def inference_cfg() -> dict:
    return load_yaml(ROOT / "configs" / "inference.yaml")


def decode_upload(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------------- sidebar --
st.title("🦺 Smart PPE Monitoring — YOLO vs RT-DETR")
st.caption("Helmet / bare-head detection on the Roboflow *Hard Hat Workers* dataset. "
           "The compliance layer is **experimental** and has not been validated against labelled compliance data.")

with st.sidebar:
    st.header("Settings")
    chosen = st.multiselect("Models", list(MODELS), default=list(MODELS))
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    use_comp = st.checkbox("Compliance analysis (experimental)", value=True)
    device = select_device("auto")
    st.write(f"Device: `{device}`")
    st.divider()
    source = st.radio("Image source", ["Example", "Upload"], horizontal=True)

samples = sorted(p for p in (ROOT / "demo_samples").glob("*") if p.suffix.lower() in IMAGE_EXTS)
image: np.ndarray | None = None
caption = ""
if source == "Example":
    if not samples:
        st.warning("No images in demo_samples/.")
    else:
        default = next((i for i, p in enumerate(samples) if p.name.startswith("005705")), 0)
        pick = st.sidebar.selectbox("Example image", samples, index=default, format_func=lambda p: f"test image {p.name.split('_')[0]}")
        image, caption = cv2.imread(str(pick)), pick.name
else:
    up = st.sidebar.file_uploader("Image", type=[e.lstrip(".") for e in sorted(IMAGE_EXTS)])
    if up is not None:
        image, caption = decode_upload(up.getvalue()), up.name
        if image is None:
            st.error("Could not decode that file as an image.")

# ---------------------------------------------------------------------------------- results ---
if image is None:
    st.info("Pick an example or upload an image in the sidebar.")
elif not chosen:
    st.info("Select at least one model.")
else:
    st.subheader(caption)
    comp = ComplianceConfig.from_dict(inference_cfg().get("compliance")) if use_comp else None
    cols = st.columns(len(chosen))
    for col, label in zip(cols, chosen):
        kind, rel = MODELS[label]
        weights = ROOT / rel
        with col:
            st.markdown(f"#### {label}")
            if not weights.is_file():
                st.error(f"Weights not found: `{rel}`. See README ('Get the trained weights').")
                continue
            det = get_detector(kind, str(weights), device)
            for _ in range(2):                                  # warm-up so the displayed latency is steady-state
                det.predict(image, conf=conf)
            res = predict_array(det, image, conf, comp)
            st.image(cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB), width="stretch")
            t = res["timing_ms"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Detections", len(res["detections"]))
            c2.metric("Model latency", f"{sum(t.values()):.1f} ms")
            c3.metric("≈ FPS", f"{1000 / max(sum(t.values()), 1e-6):.0f}")
            if res["detections"]:
                df = pd.DataFrame(res["detections"])
                st.dataframe(df.groupby("cls_name")["conf"].agg(count="count", mean_conf="mean").round(3),
                             width="stretch")
            if res["compliance"]:
                s = res["compliance"]["summary"]
                st.markdown("**Compliance (experimental):** " +
                            " · ".join(f"{STATUS_LABEL[k]}: {v}" for k, v in s.items()))
                with st.expander("Per-subject reasons"):
                    st.dataframe(pd.DataFrame(res["compliance"]["subjects"])[["status", "reason"]],
                                 width="stretch")

# ---------------------------------------------------------------------------------- about -----
with st.expander("Measured results (test split, 20 epochs each, single seed)"):
    f = ROOT / "outputs" / "metrics" / "comparison.json"
    if f.is_file():
        rows = pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
        st.dataframe(rows.round(3), width="stretch")
    else:
        st.write("No metrics found - run the Kaggle notebook and extract `ppe_results.zip` into `outputs/`.")
    st.markdown(
        "* `head` and `helmet` are detected well; the **`person` class is essentially undetected** by both models "
        "(very few training boxes), which lowers the all-class mAP and distorts the all-class precision.\n"
        "* Speeds are batch-1 model time on a Kaggle T4 (no decoding/drawing); the latency shown above is on *this* machine.\n"
        "* A bare head is only flagged as a *potential* violation; a missing helmet detection alone is reported as *uncertain*.")
