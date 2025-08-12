import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from backend.config.config import LOG_LEVEL, ROOT_DIR

def _ensure_log_dir() -> Path:
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    log_dir = _ensure_log_dir()
    file_handler = RotatingFileHandler(log_dir / "backend.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    file_handler.setLevel(level)
    file_fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(file_fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_fmt = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler.setFormatter(stream_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
