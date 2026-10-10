"""Generate notebooks/kaggle_training.ipynb (re-run after editing the cells below).

    python scripts/make_notebook.py
"""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bundle() -> str:
    """Zip the project source (small) and return it base64-encoded, so the notebook is self-contained."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["pyproject.toml", "requirements.txt", *map(str, (p.relative_to(ROOT) for d in ("configs", "src", "scripts")
                  for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts))]:
            z.write(ROOT / f, f)
    return base64.b64encode(buf.getvalue()).decode()


MD = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
CODE = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip().splitlines(True)}

CELLS = [
    MD("""
# Smart PPE Monitoring - YOLO vs RT-DETR on Kaggle (T4)

**Before running:** Settings -> Accelerator -> *GPU T4 x1*, Internet -> *On*.
Dataset: attach a YOLOv8-format Hard Hat Workers dataset (the notebook finds `data.yaml` under `/kaggle/input`;
tested listing: `hebaomer/hard-hat-workers`, Roboflow v2, train+test only). Fallback: Kaggle Secret `ROBOFLOW_API_KEY` + `RF_VERSION`.

The project source is **embedded in this notebook** - nothing else to upload. Everything (validation, loading,
training, evaluation, benchmarking) runs here on Kaggle. Set `SMOKE = True` in the setup cell for a quick dry run.
"""),
    CODE("""
import sys, subprocess, torch
print(sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30, 1), "GiB")
subprocess.run(["nvidia-smi"])
"""),
    CODE("""
import os, io, base64, zipfile, shutil
WORK = "/kaggle/working/smart-ppe-monitoring"
BUNDLE = "@@BUNDLE@@"
os.makedirs(WORK, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(WORK)   # project source is embedded in this notebook
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)   # torch is preinstalled
SMOKE = False   # True = 1 epoch on 10% of train: tests the whole pipeline and measures time/epoch
EPOCHS = None   # None = config default (50); set an int to give BOTH models the same shorter budget
EXTRA = ("--set train.epochs=1 --set train.fraction=0.1" if SMOKE else
         f"--set train.epochs={EPOCHS}" if EPOCHS else "")


def run(cmd):
    # stops the notebook if the command fails (a plain !cmd would silently continue)
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
""".replace("@@BUNDLE@@", _bundle())),
    CODE("""
# --- dataset: prefer an attached Kaggle dataset (e.g. hebaomer/hard-hat-workers), else Roboflow API ---
import glob
RF_VERSION = None       # fallback only: Roboflow version number, needs Kaggle Secret ROBOFLOW_API_KEY
hits = glob.glob("/kaggle/input/**/data.yaml", recursive=True)
if hits:
    SRC = os.path.dirname(hits[0])
else:
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "roboflow"], check=True)
    from roboflow import Roboflow
    if RF_VERSION is None:
        raise RuntimeError("No dataset attached under /kaggle/input; attach one or set RF_VERSION")
    SRC = Roboflow(api_key=key).workspace("joseph-nelson").project("hard-hat-workers").version(RF_VERSION)\
        .download("yolov8", location="/kaggle/working/hardhat_raw").location
print("source dataset:", SRC, os.listdir(SRC))
"""),
    MD("The Roboflow v2 export has only train+test. This carves a seeded 15% validation split out of *train*; test is copied untouched."),
    CODE("""
DST = "/tmp/hardhat"     # outside /kaggle/working so the Output panel only holds results
run(f"python -m smart_ppe.data.make_val_split --src {SRC} --dst {DST}")
os.environ["PPE_DATASET_ROOT"] = DST
"""),
    CODE('run("python -m smart_ppe.data.validate_dataset --config configs/dataset.yaml")'),
    MD("## Train sequentially (one model on the GPU at a time)\nSmoke test first: set `SMOKE = True` in the setup cell."),
    CODE('run(f"python -m smart_ppe.training.train_yolo --config configs/yolo.yaml {EXTRA}")'),
    CODE('run(f"python -m smart_ppe.training.train_rtdetr --config configs/rtdetr.yaml {EXTRA}")'),
    MD("## Evaluate\nValidation split = model selection. **Run the test split once, after choosing configs.**"),
    CODE("""
for m in ("yolo", "rtdetr"):
    run(f"python -m smart_ppe.evaluation.evaluate --model {m} --weights outputs/runs/{m}/weights/best.pt --split val --name {m}_val_run")
"""),
    CODE("""
for m in ("yolo", "rtdetr"):
    run(f"python -m smart_ppe.evaluation.evaluate --model {m} --weights outputs/runs/{m}/weights/best.pt --split test --name {m}")
"""),
    CODE('run("python -m smart_ppe.evaluation.compare_models --config configs/inference.yaml")'),
    CODE("""
from IPython.display import Image, display
for f in ("detection_metrics", "speed", "per_class_map", "yolo_training_curves", "rtdetr_training_curves"):
    p = f"outputs/metrics/{f}.png"
    if os.path.exists(p): display(Image(p))
"""),
    CODE("""
shutil.make_archive("/kaggle/working/ppe_results", "zip", "outputs")   # appears in the Output panel
print(os.path.getsize("/kaggle/working/ppe_results.zip") / 2**20, "MiB")
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_training.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
