# -*- coding: utf-8 -*-
"""
core/untils.py
Estado global sencillo para sesiones del Totem.
"""

from typing import Dict, Any

# estado_global[session_id] = {"slots": {...}, "history": [...]}
estado_global: Dict[str, Dict[str, Any]] = {}


def reiniciar_estado(session_id: str) -> None:
    """
    Inicializa o limpia el estado de una sesión.
    """
    estado_global[session_id] = {
        "slots": {},
        "history": [],
    }
