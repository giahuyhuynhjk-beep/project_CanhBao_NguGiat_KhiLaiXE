"""
Logging utility for Driver Monitoring System (DOAN_CNPM_NHANDIENTAIXE).
Provides standardized, structured console and file logging.
"""

import os
import sys
import logging
from typing import Optional


def get_logger(name: str = "DMS", level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a configured logger with consistent formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler with UTF-8 safe stream
        stream = sys.stdout
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = get_logger("DMS")
