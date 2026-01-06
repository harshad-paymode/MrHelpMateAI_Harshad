"""
logging_config.py
Simple logging configuration helper.
"""
import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def configure_logging(name: str = "MrHelpMateAI") -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    return logger

logger = configure_logging()