# -*- coding: utf-8 -*-
"""
core.proposal_trigger

Módulo del Totem para disparar la generación de propuesta en Catalystic.

- Toma los slots acumulados de la conversación con Nacho.
- Construye el JSON que espera tu endpoint /open-proposal-fields.
- Hace POST a Catalyst (CATA_PROPOSAL_URL) y registra el resultado en logs.

No devuelve nada al Totem; la propuesta se arma en el backend (Catalyst/Writer/WorkDrive).
"""

from __future__ import annotations

import os
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------
# Cargar .env y URL de Catalyst
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# ⚠️ En tu .env pon algo como:
# CATA_PROPOSAL_URL=https://TU-FUNCION-CATALYST/open-proposal-fields
CATALYST_PROPOSAL_URL = os.getenv("CATA_PROPOSAL_URL", "").strip()
if not CATALYST_PROPOSAL_URL:
    logger.warning(
        "core.proposal_trigger: CATA_PROPOSAL_URL no definida en .env; "
        "no se podrán disparar propuestas."
    )


# ---------------------------------------------------------
# Utilidad: inferir módulos en base a los slots detalle_*
# ---------------------------------------------------------

def _inferir_modulos_desde_slots(slots: Dict[str, Any]) -> List[str]:
    """
    A partir de los slots detalle_* decidimos qué módulos enviar a Catalyst.

    BIBLIOTECA_OFICIAL en Catalyst maneja:
    - zoho_crm
    - zoho_desk
    - zoho_books
    - zoho_inventory
    - zoho_sign
    """
    mods: List[str] = []

    # CRM
    if slots.get("detalle_zoho_crm"):
        mods.append("zoho_crm")

    # Desk (soporte)
    if slots.get("detalle_desk"):
        mods.append("zoho_desk")

    # Books
    if slots.get("detalle_zoho_books") or slots.get("detalle_books"):
        mods.append("zoho_books")

    # Inventory
    if slots.get("detalle_zoho_inventory") or slots.get("detalle_inventory"):
        mods.append("zoho_inventory")

    # Sign
    if slots.get("detalle_zoho_sign") or slots.get("detalle_sign"):
        mods.append("zoho_sign")

    # Si no detectamos nada, por defecto proponemos Zoho CRM
    if not mods:
        mods.append("zoho_crm")

    return mods


# ---------------------------------------------------------
# Construir el payload que se envía a Catalyst
# ---------------------------------------------------------

def _build_payload_from_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mapea los slots del Totem al JSON que espera tu función Catalyst
    (/open-proposal-fields).

    Campos esperados por Catalyst en main.py:
      Name, Company, Objetivo, Duracion, 'Semanas piloto', Precio,
      Modulos, Integracion_SAP, Integracion_Oracle, Requiere_SAT,
      Factura_electronica, CFDI, Complementos, Usara_Zoho_Books, etc.

    Además mandamos:
      Correo, Telefono, Detalle_Zoho_CRM, Detalle_SalesIQ, Detalle_Desk, Diagnostico,
      y una copia del dict de slots como SlotsOriginales.
    """
    nombre = str(slots.get("nombre") or "").strip()
    empresa = str(slots.get("empresa") or "").strip()
    correo = str(slots.get("correo") or "").strip()
    telefono = str(slots.get("telefono") or "").strip()

    # Objetivo / solución principal:
    solucion = str(slots.get("solucion_a_implementar") or "").strip()
    diagnostico = str(slots.get("diagnostico") or "").strip()

    # Detalles por módulo
    detalle_zoho_crm = str(slots.get("detalle_zoho_crm") or "").strip()
    detalle_salesiq = str(slots.get("detalle_salesiq") or "").strip()
    detalle_desk = str(slots.get("detalle_desk") or "").strip()

    # Objetivo final que verá la IA de Catalyst
    objetivo = solucion or diagnostico or detalle_zoho_crm

    # Por ahora no estamos capturando duración ni precio desde el Totem;
    # lo dejamos que lo calcule tu lógica de IA o lo ajustamos después.
    duracion = ""            # deja a la IA decidir (según tu main.py)
    semanas_piloto = "1"
    precio = str(slots.get("precio") or "0").strip()  # si algún día lo capturas en el diálogo

    modulos = _inferir_modulos_desde_slots(slots)

    payload: Dict[str, Any] = {
        # Campos principales esperados por main.py
        "Name": nombre,
        "Company": empresa,
        "Objetivo": objetivo,
        # Compatibilidad con: data.get("Objetivo") or data.get("Solucion a implementar")
        "Solucion a implementar": solucion,
        "Duracion": duracion,
        "Semanas piloto": semanas_piloto,
        "Precio": precio,
        "Modulos": modulos,

        # 📨 Datos de contacto
        "Correo": correo,
        "Telefono": telefono,

        # 🔍 Detalles de diagnóstico por módulo
        "Diagnostico": diagnostico,
        "Detalle_Zoho_CRM": detalle_zoho_crm,
        "Detalle_SalesIQ": detalle_salesiq,
        "Detalle_Desk": detalle_desk,

        # Flags opcionales (de momento todos en False / vacío, se pueden usar después)
        "Integracion_SAP": False,
        "Integracion_Oracle": False,
        "Requiere_SAT": False,
        "Factura_electronica": False,
        "CFDI": False,
        "Complementos": "",
        "Usara_Zoho_Books": False,

        # Copia completa de los slots por si quieres usarlos en Catalyst
        "SlotsOriginales": slots,
    }

    return payload


# ---------------------------------------------------------
# Función principal llamada desde amain.py
# ---------------------------------------------------------

async def generar_propuesta_pdf(slots: Dict[str, Any]) -> None:
    """
    Llamada desde amain.py cuando el Totem ya tiene
    los mínimos (por ahora: nombre y empresa).

    - Construye el payload a partir de los slots.
    - Hace POST a CATALYST_PROPOSAL_URL (/open-proposal-fields).
    - No devuelve nada; solo registra logs para depurar.
    """
    if not CATALYST_PROPOSAL_URL:
        logger.warning(
            "generar_propuesta_pdf: CATA_PROPOSAL_URL no está configurada; "
            "NO se envía nada a Catalyst."
        )
        return

    payload = _build_payload_from_slots(slots)

    def _do_post() -> requests.Response:
        logger.info(
            "[proposal_trigger] Enviando a Catalyst (%s) payload=%s",
            CATALYST_PROPOSAL_URL,
            json.dumps(payload, ensure_ascii=False),
        )
        return requests.post(
            CATALYST_PROPOSAL_URL,
            json=payload,
            timeout=60,
        )

    try:
        resp: requests.Response = await asyncio.to_thread(_do_post)
    except Exception as e:
        logger.exception("[proposal_trigger] Error al contactar Catalyst: %r", e)
        return

    texto = resp.text[:600]
    logger.info(
        "[proposal_trigger] Respuesta Catalyst HTTP %s body=%s",
        resp.status_code,
        texto,
    )

    if resp.status_code != 200:
        logger.warning(
            "[proposal_trigger] Catalyst devolvió código %s; revisar logs de la función.",
            resp.status_code,
        )
        return

    try:
        data = resp.json()
    except Exception:
        logger.exception(
            "[proposal_trigger] No se pudo parsear JSON de Catalyst; body=%s",
            texto,
        )
        return

    if not isinstance(data, dict):
        logger.warning(
            "[proposal_trigger] Respuesta inesperada de Catalyst (no es dict): %r",
            type(data),
        )
        return

    ok = data.get("ok")
    if ok:
        logger.info(
            "[proposal_trigger] Catalyst generó campos de propuesta correctamente (ok=true)."
        )
    else:
        logger.warning(
            "[proposal_trigger] Catalyst respondió ok=%r, detalle=%r",
            ok,
            data,
        )


# ---------------------------------------------------------
# Router opcional (para disparar vía HTTP si se requiere)
# ---------------------------------------------------------

router = APIRouter()


class ProposalTriggerRequest(BaseModel):
    slots: Dict[str, Any]


@router.post("/trigger")
async def trigger_proposal(req: ProposalTriggerRequest):
    """
    Endpoint opcional:

    POST /proposal/trigger
    {
      "slots": { ... }
    }

    → Llama internamente a generar_propuesta_pdf(slots).
    """
    try:
        await generar_propuesta_pdf(req.slots)
    except Exception as e:
        logger.exception("Error en /proposal/trigger: %r", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True}
