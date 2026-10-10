"""python -m smart_ppe.training.train_rtdetr --config configs/rtdetr.yaml"""
import sys

from smart_ppe.training.common import cli

if __name__ == "__main__":
    sys.exit(cli("rtdetr"))
