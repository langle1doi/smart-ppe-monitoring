"""python -m smart_ppe.training.train_yolo --config configs/yolo.yaml"""
import sys

from smart_ppe.training.common import cli

if __name__ == "__main__":
    sys.exit(cli("yolo"))
