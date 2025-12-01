# -*- coding: utf-8 -*-
"""
amain.py — Punto de entrada FastAPI para Totem Evolución IA3
"""

import asyncio
from uuid import uuid4
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx # Cliente HTTP asíncrono

from core.logger import get_logger
from core.dialog_engine import procesar_turno_dialogo, CAMPOS_REQUERIDOS
from core.camera_agent import iniciar_detector  # Detector de personas (YOLO + cámara)
import urllib.parse
import os

# ❌ Ya NO usamos TTS aquí para evitar duplicados.
# from core.tts_engine import speak  # 🔊 TTS (se usa solo desde core.camera_agent)

logger = get_logger(__name__)
NACHO_BASE_URL = os.getenv("NACHO_BASE_URL", "http://localhost:7000").rstrip("/")

# ----------------------------------------------------------
# Import opcional de funciones de propuesta / infografía
# ----------------------------------------------------------
try:
    from core.proposal_trigger import generar_propuesta_pdf  # type: ignore
    logger.info("amain: generar_propuesta_pdf importado correctamente.")
except Exception as e:
    generar_propuesta_pdf = None  # type: ignore
    logger.warning(
        "amain: NO se pudo importar generar_propuesta_pdf desde core.proposal_trigger: %r",
        e,
    )

try:
    from core.infographic_engine import generar_infografia_png  # type: ignore
    logger.info("amain: generar_infografia_generada importado correctamente.")
except Exception as e:
    generar_infografia_png = None  # type: ignore
    logger.warning(
        "amain: NO se pudo importar generar_infografia_png desde core.infographic_engine: %r",
        e,
    )


# ----------------------------------------------------------
# Modelos Pydantic
# ----------------------------------------------------------
class ChatTurnRequest(BaseModel):
    """
    Request para /chat/turn.

    Para ser 100% compatible con el código de la cámara ACEPTA:
    - session_id: str
    - texto_usuario: str (nombre nuevo)
    - texto: str (nombre antiguo)

    En el endpoint normalizamos al nombre interno texto_usuario.
    """
    session_id: str
    texto_usuario: Optional[str] = None
    texto: Optional[str] = None


class SessionStartRequest(BaseModel):
    modo: Optional[str] = "voz"


# ----------------------------------------------------------
# Inicialización de FastAPI
# ----------------------------------------------------------
app = FastAPI(
    title="Totem Evolución IA3",
    version="0.1.0",
    description="Backend principal del Totem de bienvenida de Evolución i3.",
)


# ----------------------------------------------------------
# Routers opcionales (proposal / infographic / health)
# ----------------------------------------------------------
# Health
try:
    from core.health import router as health_router  # type: ignore

    app.include_router(health_router, prefix="/health", tags=["health"])
except Exception:
    # Si no existe, no pasa nada
    pass

# Proposal (router HTTP, opcional)
try:
    from core.proposal_trigger import router as proposal_router  # type: ignore

    app.include_router(proposal_router, prefix="/proposal", tags=["proposal"])
except Exception:
    logger.warning(
        "core.proposal_trigger.router no disponible; /proposal/* no se registra aquí."
    )

# Infographic (router HTTP, opcional)
try:
    from core.infographic_engine import router as infographic_router  # type: ignore

    app.include_router(infographic_router, prefix="/infographic", tags=["infographic"])
except Exception:
    logger.warning(
        "core.infographic_engine.router no disponible; /infographic/* no se registra aquí."
    )


# ----------------------------------------------------------
# Estado simple de sesiones (en memoria)
# ----------------------------------------------------------
# Aquí guardamos si ya se envió propuesta / infografía para esa sesión
SESIONES: dict[str, Dict[str, Any]] = {}


def _get_sesion_meta(session_id: str) -> Dict[str, Any]:
    """
    Devuelve/crea la metadata de la sesión.
    - proposal_enviada: bool
    - infografia_generada: bool
    """
    if session_id not in SESIONES:
        SESIONES[session_id] = {
            "proposal_enviada": False,
            "infografia_generada": False,
        }
    else:
        SESIONES[session_id].setdefault("proposal_enviada", False)
        SESIONES[session_id].setdefault("infografia_generada", False)
    return SESIONES[session_id]


def _tiene_minimos_para_propuesta(slots: Dict[str, Any]) -> bool:
    """
    Condición mínima para disparar proposal / infografía,
    AUNQUE haya campos pendientes (como pediste).

    Por ahora: tener al menos nombre y empresa.
    """
    return bool(slots.get("nombre")) and bool(slots.get("empresa"))


def _normalizar_respuesta_dialog_engine(
    raw_resp: Any,
) -> tuple[str, Dict[str, Any], List[str], bool]:
    """
    Adapta lo que devuelva procesar_turno_dialogo a:
    (assistant_text, slots, campos_pendientes, campos_completos)

    Soporta dos formas:
    1) dict con llaves: assistant_text/reply, slots, campos_pendientes, ready_for_proposal
    2) tupla/lista: (assistant_text, slots, campos_pendientes, campos_completos)
    """
    # Caso 1: dict
    if isinstance(raw_resp, dict):
        assistant_text = (
            raw_resp.get("assistant_text")
            or raw_resp.get("respuesta")
            or raw_resp.get("reply")
            or ""
        )

        slots = raw_resp.get("slots") or raw_resp.get("slots_detectados") or {}
        if not isinstance(slots, dict):
            logger.warning(
                "slots en respuesta de dialog_engine (dict) no es dict: %r",
                type(slots),
            )
            slots = {}

        campos_pendientes = raw_resp.get("campos_pendientes") or raw_resp.get(
            "pending_fields"
        ) or []
        if isinstance(campos_pendientes, str):
            campos_pendientes = [campos_pendientes]
        if not isinstance(campos_pendientes, list):
            campos_pendientes = []

        campos_completos = raw_resp.get("campos_completos")
        if campos_completos is None:
            campos_completos = raw_resp.get("ready_for_proposal")

        if isinstance(campos_completos, str):
            campos_completos = campos_completos.lower() in (
                "true",
                "1",
                "yes",
                "si",
                "sí",
            )

        campos_completos = bool(campos_completos)

        return assistant_text, slots, campos_pendientes, campos_completos

    # Caso 2: tupla/lista clásica (assistant_text, slots, campos_pendientes, campos_completos)
    if isinstance(raw_resp, (list, tuple)) and len(raw_resp) >= 4:
        assistant_text, slots, campos_pendientes, campos_completos = raw_resp[:4]

        if not isinstance(slots, dict):
            logger.warning(
                "slots en tupla de dialog_engine no es dict: %r",
                type(slots),
            )
            slots = {}

        if not isinstance(campos_pendientes, list):
            campos_pendientes = []

        campos_completos = bool(campos_completos)
        return assistant_text, slots, campos_pendientes, campos_completos

    # Cualquier otra cosa es inesperada
    logger.error(
        "Formato inesperado de respuesta de dialog_engine: %r (%s)",
        raw_resp,
        type(raw_resp),
    )
    raise RuntimeError("Formato inesperado de respuesta de dialog_engine")


# ----------------------------------------------------------
# Eventos de arranque y apagado
# ----------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("🚀 Totem Evolución IA3 iniciado correctamente.")
    logger.info("🎥 Activando detector de personas...")
    try:
        # Nota: Si iniciar_detector() es una función síncrona que bloquea
        # el hilo por mucho tiempo, idealmente debe ser envuelta en asyncio.to_thread().
        iniciar_detector()
    except Exception:
        logger.exception("Error al iniciar el detector de personas.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("🛑 Totem Evolución IA3 apagándose.")


# ----------------------------------------------------------
# Endpoints
# ----------------------------------------------------------
@app.get("/")
async def root():
    """Ping rápido para comprobar que el backend está vivo."""
    return {
        "status": "ok",
        "message": "Totem Evolución IA3 backend activo.",
    }


@app.post("/session/start")
async def session_start():
    """
    Crea una nueva sesión de diálogo.
    La cámara / YOLO llama a este endpoint antes de iniciar la conversación por voz.
    """
    session_id = str(uuid4())
    _get_sesion_meta(session_id)  # inicializa flags

    logger.info("Nueva sesión creada: %s", session_id)

    return {
        "session_id": session_id,
        "campos_requeridos": list(CAMPOS_REQUERIDOS),
    }


async def _enviar_slots_al_ui(respuesta: dict) -> None:
    """
    Empuja al visor (ui.py) la info básica del lead para el panel CRM.
    
    ¡IMPORTANTE! Ahora es una función asíncrona usando httpx para evitar bloqueos.

    Usa el servidor HTTP de Nacho en NACHO_BASE_URL (por defecto http://localhost:7000).
    NO lanza excepciones hacia afuera (solo loguea).
    """
    try:
        slots = respuesta.get("slots") or {}
        if not isinstance(slots, dict):
            return

        email = slots.get("correo") or slots.get("email") or ""
        nombre = slots.get("nombre") or slots.get("name") or ""
        empresa = slots.get("empresa") or slots.get("company") or ""

        pendientes = respuesta.get("campos_pendientes") or []
        progreso = respuesta.get("progreso", None)

        partes_estado = []
        if pendientes:
            partes_estado.append("Pendientes: " + ", ".join(pendientes))
        if isinstance(progreso, (int, float)):
            partes_estado.append(f"Progreso: {int(round(progreso * 100))}%")

        proposal = " | ".join(partes_estado)

        data = {
            "email": email,
            "name": nombre,
            "company": empresa,
            "proposal": proposal,
        }

        query = urllib.parse.urlencode(data, doseq=False, safe="")
        url = f"{NACHO_BASE_URL}/crm?{query}"

        logger.info("[UI] Enviando datos CRM al visor: %s", url)

        # GET rápido ASÍNCRONO y si falla NO rompemos el backend
        async with httpx.AsyncClient(timeout=0.5) as client:
            await client.get(url)

    except Exception as e:
        logger.warning("[UI] No se pudo notificar CRM al visor: %s", e)


@app.post("/chat/turn")
async def chat_turn(payload: ChatTurnRequest):
    """
    Turno de diálogo:
    - Recibe el texto del usuario (ya transcrito por ASR).
    - Llama a core.dialog_engine.procesar_turno_dialogo(...).
    - Regresa la respuesta en texto y banderas de control.
    - Si ya tenemos nombre y empresa, DISPARA LA PROPUESTA y la INFOGRAFÍA
      (una sola vez cada una por sesión), AUNQUE haya campos pendientes.

    IMPORTANTE:
    - Acepta tanto "texto_usuario" como "texto" en el body.
      Esto evita el error 422 con la versión actual de core.camera_agent.
    - NO hace TTS aquí. El TTS lo maneja core.camera_agent para evitar duplicados.
    """
    session_id = payload.session_id
    # El texto limpio, que se usa para llamar a procesar_turno_dialogo
    texto_usuario_limpio = (payload.texto_usuario or payload.texto or "").strip()

    if not texto_usuario_limpio:
        raise HTTPException(
            status_code=400,
            detail="El cuerpo debe incluir 'texto_usuario' o 'texto' con contenido.",
        )

    logger.info(
        "Turno de diálogo recibido. session_id=%s, texto='%s'",
        session_id,
        texto_usuario_limpio,
    )

    # ------------------------------------------------------
    # 1) Llamar al motor de diálogo y normalizar respuesta
    # ------------------------------------------------------
    try:
        raw_resp = procesar_turno_dialogo(session_id, texto_usuario_limpio)
        assistant_text, slots, campos_pendientes, campos_completos = (
            _normalizar_respuesta_dialog_engine(raw_resp)
        )

        logger.info(
            "Resultado normalizado de dialog_engine: respuesta='%s', campos_completos=%s",
            assistant_text,
            campos_completos,
        )
        logger.info(
            "[%s] slots=%r | pendientes=%r | ready_minimos=%s",
            session_id,
            slots,
            campos_pendientes,
            _tiene_minimos_para_propuesta(slots),
        )
    except Exception as e:
        logger.exception("Error procesando el diálogo en /chat/turn.")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando el diálogo: {e}",
        )

    # ------------------------------------------------------
    # 2) Disparo de propuesta e infografía
    #    (aquí YA se tienen los slots actualizados)
    # ------------------------------------------------------
    meta = _get_sesion_meta(session_id)
    ready_minimos = _tiene_minimos_para_propuesta(slots)

    logger.info(
        "[%s] meta_inicio: proposal_enviada=%s, infografia_generada=%s, ready_minimos=%s",
        session_id,
        meta["proposal_enviada"],
        meta["infografia_generada"],
        ready_minimos,
    )

    if ready_minimos:
        # -------- Propuesta (Zoho Flow) --------
        if generar_propuesta_pdf is not None and not meta["proposal_enviada"]:
            meta["proposal_enviada"] = True
            logger.info(
                "[%s] Disparando propuesta (Zoho Flow) con nombre=%r, empresa=%r",
                session_id,
                slots.get("nombre"),
                slots.get("empresa"),
            )
            try:
                # Tu función en core/proposal_trigger.py es async → la esperamos aquí
                await generar_propuesta_pdf(slots)  # type: ignore[arg-type]
                logger.info(
                    "[%s] generar_propuesta_pdf finalizó (revisa logs de Zoho Flow / archivo JSON).",
                    session_id,
                )
            except Exception:
                logger.exception(
                    "[%s] Error en generar_propuesta_pdf (Zoho Flow / JSON local).",
                    session_id,
                )
        elif generar_propuesta_pdf is None:
            logger.warning(
                "[%s] generar_propuesta_pdf es None; NO se disparó propuesta.",
                session_id,
            )

        # -------- Infografía (backend local) --------
        if generar_infografia_png is not None and not meta["infografia_generada"]:
            meta["infografia_generada"] = True
            logger.info(
                "[%s] Generando infografía con los slots actuales.",
                session_id,
            )
            try:
                # Se usa asyncio.to_thread para no bloquear el bucle de eventos con la tarea síncrona
                await asyncio.to_thread(generar_infografia_png, slots)  # type: ignore[arg-type]
                logger.info(
                    "[%s] generar_infografia_png finalizó (PNG/PDF generados).",
                    session_id,
                )
            except Exception:
                logger.exception(
                    "[%s] Error en generar_infografia_png (PIL / escritura de archivos).",
                    session_id,
                )
        elif generar_infografia_png is None:
            logger.warning(
                "[%s] generar_infografia_png es None; NO se generó infografía.",
                session_id,
            )

    # ------------------------------------------------------
    # 3) Calcular progreso para el panel (opcional)
    #    Usamos CAMPOS_REQUERIDOS como referencia de total
    # ------------------------------------------------------
    try:
        campos_totales = len(CAMPOS_REQUERIDOS) or 1
        campos_llenos = sum(
            1 for campo in CAMPOS_REQUERIDOS if slots.get(campo)
        )
        progreso = campos_llenos / campos_totales
    except Exception:
        campos_totales = 1
        campos_llenos = 0
        progreso = 0.0

    # ------------------------------------------------------
    # 4) Respuesta al front / cámara
    # ------------------------------------------------------
    # FIX: Se usa 'payload' en lugar de la variable indefinida 'req'.
    # Usamos el valor original del campo texto_usuario o texto (si texto_usuario es None)
    input_text_for_response = payload.texto_usuario if payload.texto_usuario is not None else payload.texto
    
    respuesta = {
    "session_id": session_id,
    "texto_usuario": input_text_for_response,
    "respuesta": assistant_text,
    "slots": slots,
    "campos_pendientes": campos_pendientes,
    "campos_completos": bool(campos_completos),
    "campos_totales": campos_totales,
    "campos_llenos": campos_llenos,
    "progreso": progreso,  # 0.0–1.0
    # Por ahora el flujo de voz no usa 'terminar', lo dejamos siempre False
    "terminar": False,
    "resultado_bruto": [
        assistant_text,
        slots,
        campos_pendientes,
        bool(campos_completos),
    ],
    }
    # 5) Empujar estado al visor (panel CRM abajo del UI)
    try:
        # La función _enviar_slots_al_ui es asíncrona (usa httpx), por eso requiere await
        await _enviar_slots_al_ui(respuesta)
    except Exception:
        # Nunca queremos tumbar el backend solo por un problema visual
        logger.exception("Error al enviar datos al panel CRM del visor")

    logger.info("Respuesta normalizada para /chat/turn: %r", respuesta)

    return respuesta


# ----------------------------------------------------------
# Punto de entrada opcional (por si ejecutas `python amain.py`)
# ----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "amain:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )