# -*- coding: utf-8 -*-
# core/logger.py
from __future__ import annotations
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def get_logger(name: str):
    return logging.getLogger(name)

def log_info(msg: str):
    logging.getLogger("ei3").info(msg)
