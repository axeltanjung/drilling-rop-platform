"""Project-wide logging helper."""
from __future__ import annotations

import logging
import sys

from backend.utils.config import settings

_CONFIGURED = False


def get_logger(name: str = "drilling") -> logging.Logger:
    """Return a configured logger. Idempotent."""
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        _CONFIGURED = True

    return logger
