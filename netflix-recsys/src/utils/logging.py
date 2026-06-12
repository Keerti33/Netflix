"""Logging utilities for Netflix recommendation system.

Provides consistent logging configuration for all modules,
with both console and optional file output.

Usage::

    from src.utils.logging import get_logger
    
    logger = get_logger("model_training")
    logger.info("Starting model training...")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically module name).
        level: Logging level (default: INFO).
        log_file: Optional file path for file logging.

    Returns:
        Configured Logger instance.

    Example::

        logger = get_logger("svd_model", log_file=Path("logs/model.log"))
        logger.info("Model training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(name)s] %(levelname)s: %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
