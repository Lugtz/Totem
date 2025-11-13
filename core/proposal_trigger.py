# -*- coding: utf-8 -*-
"""
core/proposal_trigger.py

Módulo para disparar la generación de propuestas desde el Totem Evolución IA3.

- Recibe un diccionario de campos (slots) desde el diálogo.
- Normaliza nombres de campos.
- Envía los datos a Zoho Flow (FLOW_URL) en el formato:
    { "payload": { ... } }
- Si FLOW_URL NO está definido, guarda el payload en core/outputs/proposal_*.json
  para poder revisar qué se habría enviado.
"""

from __future__ import annotations

import os
import json
import asyncio
import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import requests
from dotenv import load_dotenv


# ----------------------------------------------------------------------
# 1. ENTORNO Y RUTAS
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent.resolve()
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# Carpeta de salidas locales (propuestas de respaldo)
CORE_DIR = Path(__file__).parent  # .../Totem/core
OUTPUT_DIR = CORE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOW_URL: Optional[str] = os.getenv("FLOW_URL", "").strip() or None
WRITER_DOC_ID: Optional[str] = os.getenv("WRITER_DOC_ID", "").strip() or None


# ----------------------------------------------------------------------
# 2. MAPEO DE CAMPOS
# ----------------------------------------------------------------------
def _map_campos(campos: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza los campos que vienen del Totem y los deja listos
    para Zoho Flow / Writer.

    Importante:
    - 'objetivo' y 'solucion' se llenan con el mismo valor
      (lo que el Totem capturó como solución/objetivo).
    - Acepta tanto 'semana_piloto' como 'semanas_piloto' y variantes.
    - No obliga a tener IVA ni precios por usuario; si no vienen, se omiten.
    """

    # Texto de objetivo / solución (puede venir como 'solucion' o 'objetivo')
    objetivo_o_solucion = (
        campos.get("solucion")
        or campos.get("objetivo")
    )

    out: Dict[str, Any] = {
        # Datos base
        "nombre": campos.get("nombre"),
        "empresa": campos.get("empresa"),
        "email": campos.get("email"),

        # Objetivo / solución del proyecto
        "objetivo": objetivo_o_solucion,
        "solucion": objetivo_o_solucion,

        # Datos de proyecto
        "duracion": campos.get("duracion"),
        "precio": campos.get("precio"),
        "moneda": campos.get("moneda"),

        # Semana / mes de piloto (acepta varias variantes)
        "semanas_piloto": (
            campos.get("semana_piloto")
            or campos.get("semanas_piloto")
            or campos.get("Semanas piloto")
        ),

        # IVA (opcional)
        "iva": campos.get("iva") or campos.get("IVA"),

        # Porcentajes (opcional)
        "pago_inicio": (
            campos.get("porcentaje_inicio")
            or campos.get("pago_inicio")
            or campos.get("Porcentaje de inicio")
        ),
        "pago_cierre": (
            campos.get("porcentaje_cierre")
            or campos.get("pago_cierre")
            or campos.get("Porcentaje al cierre")
        ),

        # Licencia / versión (opcional)
        "licencia": campos.get("licencia") or campos.get("Licencia a utilizar"),
        "version": campos.get("version") or campos.get("Versión"),

        # Precios por usuario (opcionales)
        "precio_mensual_usuario": (
            campos.get("precio_mensual_usuario")
            or campos.get("Precio mensual por Usuario")
        ),
        "precio_anual_usuario": (
            campos.get("precio_anual_usuario")
            or campos.get("Precio Anual por usuario")
        ),
        "moneda_licencias": (
            campos.get("moneda_licencias")
            or campos.get("Moneda de las Licencias")
        ),
    }

    # ID del Writer Doc, si está configurado
    if WRITER_DOC_ID:
        out["writer_doc_id"] = WRITER_DOC_ID

    # Limpia claves vacías (None o "")
    return {k: v for k, v in out.items() if v not in (None, "")}


# ----------------------------------------------------------------------
# 3. FUNCIÓN PRINCIPAL: generar_propuesta_pdf
# ----------------------------------------------------------------------
async def generar_propuesta_pdf(campos: Dict[str, Any]) -> Any:
    """
    Recibe los campos crudos del Totem (slots) y:

    1. Los normaliza con _map_campos().
    2. Si FLOW_URL está configurado → POST a Zoho Flow:
          { "payload": { ...campos_normalizados... } }
    3. Si NO hay FLOW_URL → guarda el JSON en core/outputs/proposal_*.json

    Devuelve:
    - El resultado de Flow (dict) si la llamada fue exitosa, o
    - La ruta del archivo JSON local de respaldo, o
    - None si algo sale muy mal (ya se imprime el error).
    """
    campos_normalizados = _map_campos(campos)
    payload = {"payload": campos_normalizados}

    # Log rápido para depuración
    print("▶ generar_propuesta_pdf() — campos_normalizados:")
    print(json.dumps(campos_normalizados, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    # CASO 1: Hay FLOW_URL → enviar a Zoho Flow
    # ------------------------------------------------------------------
    if FLOW_URL:
        def _post() -> Any:
            r = requests.post(FLOW_URL, json=payload, timeout=30)
            r.raise_for_status()

            ct = (r.headers.get("Content-Type") or "").lower()
            if "application/json" in ct:
                try:
                    return r.json()
                except Exception:
                    return {"status": r.status_code, "raw": r.text[:500]}

            return {"status": r.status_code, "text": r.text[:500]}

        try:
            result = await asyncio.to_thread(_post)
            print("✅ Zoho Flow OK:", result)
            # Guardamos copia de lo que mandamos para depurar si hace falta
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = OUTPUT_DIR / f"proposal_{ts}.json"
            with backup_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {"flow_url": FLOW_URL, "payload": payload, "result": result},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"💾 Respaldo de propuesta guardado en: {backup_path}")
            return result
        except Exception as e:
            print(f"❌ Error en generar_propuesta_pdf (FLOW_URL): {e}")
            # También dejamos respaldo local del error y del payload
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            error_path = OUTPUT_DIR / f"proposal_error_{ts}.json"
            with error_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "flow_url": FLOW_URL,
                        "payload": payload,
                        "error": repr(e),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"⚠ Respaldo de error guardado en: {error_path}")
            return None

    # ------------------------------------------------------------------
    # CASO 2: No hay FLOW_URL → solo guardar el JSON localmente
    # ------------------------------------------------------------------
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_json = OUTPUT_DIR / f"proposal_{ts}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"⚠ FLOW_URL no definido. Se guardó payload en: {out_json}")
        return str(out_json)
    except Exception as e:
        print(f"❌ Error guardando propuesta localmente: {e}")
        return None
