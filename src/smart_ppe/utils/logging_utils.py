"""Logging: one line per event, with `key=value` extras via kv()."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "smart_ppe", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(level)
        log.propagate = False
    return log


def kv(**fields: object) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items())
