import logging, os
from logging.handlers import RotatingFileHandler

def get_logger(name: str):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    os.makedirs("data/logs", exist_ok=True)
    fh = RotatingFileHandler("data/logs/app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger
