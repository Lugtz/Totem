# -*- coding: utf-8 -*- 
"""
amain.py — Punto de entrada FastAPI para Totem Evolución IA3
"""

import asyncio
from uuid import uuid4
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx  # Cliente HTTP asíncrono
import urllib.parse
import os
import json  # 👈 NUEVO: para guardar estado en JSON

from core.logger import get_logger
from core.dialog_engine import procesar_turno_dialogo, CAMPOS_REQUERIDOS
from core.camera_agent import iniciar_detector  # Detector de personas (YOLO + cámara)

logger = get_logger(__name__)
NACHO_BASE_URL = os.getenv("NACHO_BASE_URL", "http://localhost:7000").rstrip("/")

# Archivo JSON donde se guardará el estado normalizado de sesiones
JSON_STATE_FILE = os.getenv("TOTEM_STATE_JSON", "totem_sessions_state.json")

# ----------------------------------------------------------
# PALABRAS CLAVE DE CIERRE RÁPIDO
# ----------------------------------------------------------
END_KEYWORDS = [
    "gracias, nacho",
    "ya es todo",
    "ya termine",
    "ya terminé",
    "eso es todo",
    "listo gracias",
    "listo, gracias",
    "adios",
    "adiós",
]

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
    logger.info("amain: generar_infografia_png importado correctamente.")
except Exception as e:
    generar_infografia_png = None  # type: ignore
    logger.warning(
        "amain: NO se pudo importar generar_infografia_png desde core.infographic_engine: %r",
        e,
    )

# ----------------------------------------------------------
# Import opcional del cliente de CRM (Zoho)
# ----------------------------------------------------------
try:
    from core.crm_client import crear_lead_en_crm  # type: ignore
    logger.info("amain: crear_lead_en_crm importado correctamente.")
except Exception as e:
    crear_lead_en_crm = None  # type: ignore
    logger.warning(
        "amain: NO se pudo importar crear_lead_en_crm desde core.crm_client: %r",
        e,
    )

# ----------------------------------------------------------
# Import opcional del flujo WhatsApp + Email + Infografía
# ----------------------------------------------------------
try:
    from core.whatsapp_email_infografia import flujo_infografia_whatsapp_email  # type: ignore
    logger.info("amain: flujo_infografia_whatsapp_email importado correctamente.")
except Exception:
    flujo_infografia_whatsapp_email = None  # type: ignore
    logger.exception(
        "amain: ERROR importando flujo_infografia_whatsapp_email desde core.whatsapp_email_infografia"
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
# Aquí guardamos si ya se envió propuesta / infografía / lead CRM para esa sesión
SESIONES: dict[str, Dict[str, Any]] = {}


def _get_sesion_meta(session_id: str) -> Dict[str, Any]:
    """
    Devuelve/crea la metadata de la sesión.
    - proposal_enviada: bool
    - infografia_generada: bool
    - lead_creado: bool
    """
    if session_id not in SESIONES:
        SESIONES[session_id] = {
            "proposal_enviada": False,
            "infografia_generada": False,
            "lead_creado": False,
        }
    else:
        SESIONES[session_id].setdefault("proposal_enviada", False)
        SESIONES[session_id].setdefault("infografia_generada", False)
        SESIONES[session_id].setdefault("lead_creado", False)
    return SESIONES[session_id]


def _tiene_minimos_para_propuesta(slots: Dict[str, Any]) -> bool:
    """
    Condición mínima para disparar proposal / infografía,
    AUNQUE haya campos pendientes.

    Por ahora: tener al menos nombre y empresa.
    """
    return bool(slots.get("nombre")) and bool(slots.get("empresa"))


def _mapear_slots_para_crm(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mapea los slots del Totem a los nombres esperados por core.crm_client.crear_lead_en_crm.

    Resultado esperado:
    {
        "Name": "...",
        "Company": "...",
        "Email": "...",
        "Phone": "...",
        "OBJETIVO": "..."   # tomado de solucion_a_implementar / objetivo / diagnostico
    }
    """
    payload = {
        "Name": (
            slots.get("nombre")
            or slots.get("Name")
            or "Visitante Totem"
        ),
        "Company": (
            slots.get("empresa")
            or slots.get("Company")
            or "Visitante Totem"
        ),
        "Email": (
            slots.get("correo")
            or slots.get("email")
            or slots.get("Email")
        ),
        "Phone": (
            slots.get("telefono")
            or slots.get("phone")
            or slots.get("Phone")
        ),
        "OBJETIVO": (
            slots.get("solucion_a_implementar")
            or slots.get("objetivo")
            or slots.get("OBJETIVO")
            or slots.get("diagnostico")
        ),
    }

    # Log para ver qué está pasando
    logger.info(
        "[_mapear_slots_para_crm] slots_raw=%r | payload_crm=%r",
        slots,
        payload,
    )

    # Print para verlo directo en consola
    print(">>> [_mapear_slots_para_crm] slots_raw =", slots)
    print(">>> [_mapear_slots_para_crm] payload_crm =", payload)

    return payload


def _normalizar_respuesta_dialog_engine(
    raw_resp: Any,
) -> tuple[str, Dict[str, Any], List[str], bool, bool]:
    """
    Adapta lo que devuelva procesar_turno_dialogo a:
    (assistant_text, slots, campos_pendientes, campos_completos, es_despedida)

    Soporta dos formas:
    1) dict con llaves:
       - assistant_text / respuesta / reply
       - slots / slots_detectados
       - campos_pendientes / pending_fields
       - campos_completos / ready_for_proposal
       - es_despedida_model / end_of_conversation (opcional)
    2) tupla/lista:
       - (assistant_text, slots, campos_pendientes, campos_completos)
       - o (assistant_text, slots, campos_pendientes, campos_completos, es_despedida)
    """
    es_despedida = False

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

        # Bandera de despedida (si el modelo la manda)
        es_despedida_val = (
            raw_resp.get("es_despedida_model")
            or raw_resp.get("end_of_conversation")
        )
        if isinstance(es_despedida_val, str):
            es_despedida = es_despedida_val.lower() in (
                "true",
                "1",
                "yes",
                "si",
                "sí",
            )
        elif isinstance(es_despedida_val, bool):
            es_despedida = es_despedida_val
        else:
            es_despedida = False

        return assistant_text, slots, campos_pendientes, campos_completos, es_despedida

    # Caso 2: tupla/lista clásica
    if isinstance(raw_resp, (list, tuple)) and len(raw_resp) >= 4:
        # Soportamos opcionalmente un 5º elemento = es_despedida
        if len(raw_resp) >= 5:
            assistant_text, slots, campos_pendientes, campos_completos, es_despedida_val = raw_resp[:5]
        else:
            assistant_text, slots, campos_pendientes, campos_completos = raw_resp[:4]
            es_despedida_val = False

        if not isinstance(slots, dict):
            logger.warning(
                "slots en tupla de dialog_engine no es dict: %r",
                type(slots),
            )
            slots = {}

        if not isinstance(campos_pendientes, list):
            campos_pendientes = []

        campos_completos = bool(campos_completos)

        if isinstance(es_despedida_val, str):
            es_despedida = es_despedida_val.lower() in (
                "true",
                "1",
                "yes",
                "si",
                "sí",
            )
        else:
            es_despedida = bool(es_despedida_val)

        return assistant_text, slots, campos_pendientes, campos_completos, es_despedida

    # Cualquier otra cosa es inesperada
    logger.error(
        "Formato inesperado de respuesta de dialog_engine: %r (%s)",
        raw_resp,
        type(raw_resp),
    )
    raise RuntimeError("Formato inesperado de respuesta de dialog_engine")


# ----------------------------------------------------------
# Helper: guardar estado normalizado en JSON
# ----------------------------------------------------------
def _guardar_estado_json(session_id: str, respuesta: Dict[str, Any]) -> None:
    """
    Guarda/actualiza el estado normalizado de la sesión en un archivo JSON.

    Estructura del JSON:
    {
      "session_id_1": { ...respuesta... },
      "session_id_2": { ... }
    }
    """
    try:
        data: Dict[str, Any] = {}
        if os.path.exists(JSON_STATE_FILE):
            try:
                with open(JSON_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                # Si el archivo está corrupto u otra cosa, lo reseteamos
                logger.warning(
                    "[JSON] No se pudo leer/parsing %s, se reinicia estructura.",
                    JSON_STATE_FILE,
                )
                data = {}

        # Actualizamos solo la entrada de esta sesión
        data[session_id] = respuesta

        with open(JSON_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "[JSON] Estado de sesión %s guardado/actualizado en %s",
            session_id,
            JSON_STATE_FILE,
        )
    except Exception:
        logger.exception(
            "[JSON] Error guardando estado normalizado de la sesión en archivo."
        )


# ----------------------------------------------------------
# Eventos de arranque y apagado
# ----------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("🚀 Totem Evolución IA3 iniciado correctamente.")
    logger.info("🎥 Activando detector de personas...")
    try:
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
        # 👉 Teléfono para el panel CRM del UI
        telefono = (
            slots.get("telefono")
            or slots.get("phone")
            or slots.get("Phone")
            or ""
        )

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
            "phone": telefono,
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

async def _limpiar_panel_crm() -> None:
    """
    Envía al visor un estado vacío para limpiar el panel CRM
    (nombre, empresa, correo, teléfono y texto de propuesta).
    """
    try:
        respuesta_vacia = {
            "slots": {
                "nombre": "",
                "empresa": "",
                "correo": "",
                "email": "",
                "telefono": "",
                "phone": "",
            },
            "campos_pendientes": [],
            "progreso": 0.0,
        }

        await _enviar_slots_al_ui(respuesta_vacia)
        logger.info("[UI] Panel CRM limpiado después del flujo WhatsApp/Email.")
    except Exception:
        logger.exception("[UI] Error al intentar limpiar el panel CRM.")

@app.post("/chat/turn")
async def chat_turn(payload: ChatTurnRequest):
    """
    Turno de diálogo:
    - Recibe el texto del usuario (ya transcrito por ASR).
    - Llama a core.dialog_engine.procesar_turno_dialogo(...).
    - Regresa la respuesta en texto y banderas de control.
    - Si ya tenemos nombre y empresa, DISPARA LA PROPUESTA y la INFOGRAFÍA
      (una sola vez cada una por sesión), AUNQUE haya campos pendientes.
    - Cuando la conversación YA ES UNA DESPEDIDA y hay mínimos (nombre + empresa),
      se manda el lead a Zoho CRM y se dispara también el flujo de Whats/Email.
    - EXTRA: si el usuario dice una PALABRA CLAVE de cierre rápido
      (ej. 'gracias nacho', 'ya es todo'), se fuerza el cierre:
         * es_despedida = True
         * terminar = True
         * se intenta mandar el lead al CRM AUNQUE falten campos.
    """
    session_id = payload.session_id
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
    # 0) Detectar si el usuario dijo la PALABRA CLAVE de cierre rápido
    # ------------------------------------------------------
    texto_lower = texto_usuario_limpio.lower().strip()
    cierre_forzado_por_keyword = texto_lower in END_KEYWORDS
    if cierre_forzado_por_keyword:
        logger.info(
            "[%s] CIERRE RÁPIDO por keyword detectada ('%s').",
            session_id,
            texto_lower,
        )

    # ------------------------------------------------------
    # 1) Llamar al motor de diálogo y normalizar respuesta
    # ------------------------------------------------------
    try:
        raw_resp = procesar_turno_dialogo(session_id, texto_usuario_limpio)
        assistant_text, slots, campos_pendientes, campos_completos, es_despedida = (
            _normalizar_respuesta_dialog_engine(raw_resp)
        )

        # Si entramos por palabra clave, FORZAMOS que sea despedida,
        # aunque el modelo no lo haya marcado.
        if cierre_forzado_por_keyword:
            es_despedida = True

        logger.info(
            "Resultado normalizado de dialog_engine: respuesta='%s', campos_completos=%s, es_despedida=%s",
            assistant_text,
            campos_completos,
            es_despedida,
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
    # 2) Disparo de propuesta, infografía y LEAD en CRM
    # ------------------------------------------------------
    meta = _get_sesion_meta(session_id)
    ready_minimos = _tiene_minimos_para_propuesta(slots)

    logger.info(
        "[%s] meta_inicio: proposal_enviada=%s, infografia_generada=%s, lead_creado=%s, ready_minimos=%s",
        session_id,
        meta["proposal_enviada"],
        meta["infografia_generada"],
        meta["lead_creado"],
        ready_minimos,
    )

    # -------- Propuesta (Zoho Flow) --------
    if ready_minimos:
        if generar_propuesta_pdf is not None and not meta["proposal_enviada"]:
            meta["proposal_enviada"] = True
            logger.info(
                "[%s] Disparando propuesta (Zoho Flow) con nombre=%r, empresa=%r",
                session_id,
                slots.get("nombre"),
                slots.get("empresa"),
            )
            try:
                # generar_propuesta_pdf es async → la esperamos aquí
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
                    "[%s] Error en generar_infografia_png (generación local de archivos).",
                    session_id,
                )
        elif generar_infografia_png is None:
            logger.warning(
                "[%s] generar_infografia_png es None; NO se generó infografía.",
                session_id,
            )

    # -------- Lead en Zoho CRM + flujo Whats/Email --------
    # CASO 1 (normal):
    #   - es_despedida == True (Nacho marcó que la conversación terminó)
    #   - ready_minimos == True (al menos nombre y empresa)
    # CASO 2 (CIERRE RÁPIDO):
    #   - cierre_forzado_por_keyword == True
    #
    # En ambos casos:
    #   - crear_lead_en_crm disponible
    #   - aún no se haya creado lead para esta sesión
    crear_lead_condicion_normal = es_despedida and ready_minimos
    crear_lead_condicion_forzada = cierre_forzado_por_keyword

    logger.info(
        "[%s] FLAGS FIN: es_despedida=%s | cierre_forzado=%s | campos_completos=%s | ready_minimos=%s",
        session_id,
        es_despedida,
        cierre_forzado_por_keyword,
        campos_completos,
        ready_minimos,
    )

    if (
        crear_lead_en_crm is not None
        and not meta["lead_creado"]
        and (crear_lead_condicion_normal or crear_lead_condicion_forzada)
    ):
        # 🔴 Un solo try para CRM + flujo Whats/Email
        try:
            payload_crm = _mapear_slots_para_crm(slots)
            logger.info(
                "[%s] Enviando lead a Zoho CRM con payload: %r",
                session_id,
                payload_crm,
            )

            # Llamada síncrona al CRM
            crear_lead_en_crm(payload_crm)
            meta["lead_creado"] = True
            logger.info("[%s] Lead creado en Zoho CRM correctamente.", session_id)

            # ------------------------------------------------------
            # Flujo WhatsApp + Email + Infografía (si está disponible)
            # ------------------------------------------------------
            if flujo_infografia_whatsapp_email is not None:
                nombre_cliente = (
                    slots.get("nombre")
                    or slots.get("Name")
                    or "Visitante Totem"
                )
                empresa_cliente = (
                    slots.get("empresa")
                    or slots.get("Company")
                    or "Visitante Totem"
                )
                objetivo_cliente = (
                    slots.get("solucion_a_implementar")
                    or slots.get("objetivo")
                    or slots.get("OBJETIVO")
                    or slots.get("diagnostico")
                    or ""
                )

                problemas = slots.get("problemas") or slots.get("retos") or ""
                necesidades = (
                    slots.get("necesidades")
                    or slots.get("necesidades_clave")
                    or ""
                )
                soluciones = (
                    slots.get("soluciones")
                    or slots.get("productos")
                    or slots.get("solucion")
                    or []
                )

                datos_cliente: Dict[str, Any] = {
                    "nombre": nombre_cliente,
                    "empresa": empresa_cliente,
                    "OBJETIVO": objetivo_cliente,
                    "problemas": problemas,
                    "necesidades": necesidades,
                    "soluciones": soluciones,
                    "_slots_raw": slots,
                }

                # Limpiar y formatear teléfono
                telefono_slot = (
                    slots.get("telefono")
                    or slots.get("phone")
                    or slots.get("Phone")
                    or ""
                )
                telefono_limpio = (
                    telefono_slot.replace("-", "")
                    .replace("_", "")
                    .replace(" ", "")
                )
                telefono_cliente = (
                    f"+521{telefono_limpio}" if telefono_limpio else None
                )

                email_cliente = (
                    slots.get("correo")
                    or slots.get("email")
                    or slots.get("Email")
                    or None
                )

                nombre_archivo = (
                    f"infografia_{empresa_cliente}".replace(" ", "_")
                )

                logger.info(
                    "[%s] Disparando flujo_infografia_whatsapp_email (SINCRONO) "
                    "con telefono=%r, email=%r, nombre_archivo=%r",
                    session_id,
                    telefono_cliente,
                    email_cliente,
                    nombre_archivo,
                )

                print(
                    f">>> [{session_id}] EJECUTANDO flujo_infografia_whatsapp_email "
                    f"PARA {nombre_cliente} / {empresa_cliente}"
                )

                flujo_infografia_whatsapp_email(
                    datos_cliente,
                    telefono_cliente,
                    email_cliente,
                    nombre_archivo,
                    True,   # enviar_whatsapp
                    True,   # enviar_email
                )

                # 🔵 Aquí limpiamos el panel CRM del visor DESPUÉS del flujo
                await _limpiar_panel_crm()

                logger.info(
                    "[%s] flujo_infografia_whatsapp_email finalizó correctamente.",
                    session_id,
                )
            else:
                logger.warning(
                    "[%s] flujo_infografia_whatsapp_email es None; "
                    "NO se disparó flujo WhatsApp/Email.",
                    session_id,
                )

        except Exception:
            logger.exception(
                "[%s] Error en creación de lead o en flujo WhatsApp/Email.",
                session_id,
            )


    # ------------------------------------------------------
    # 3) Calcular progreso para el panel (opcional)
    #    Usamos CAMPOS_REQUERIDOS como referencia de total
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
    input_text_for_response = (
        payload.texto_usuario if payload.texto_usuario is not None else payload.texto
    )

    # terminamos conversación si:
    #   - el modelo marcó es_despedida
    #   - O hubo cierre rápido por keyword
    terminar_flag = bool(es_despedida or cierre_forzado_por_keyword)

    respuesta: Dict[str, Any] = {
        "session_id": session_id,
        "texto_usuario": input_text_for_response,
        "respuesta": assistant_text,
        "slots": slots,
        "campos_pendientes": campos_pendientes,
        "campos_completos": bool(campos_completos),
        "campos_totales": campos_totales,
        "campos_llenos": campos_llenos,
        "progreso": progreso,  # 0.0–1.0
        "terminar": terminar_flag,
        "resultado_bruto": [
            assistant_text,
            slots,
            campos_pendientes,
            bool(campos_completos),
        ],
        "es_despedida": bool(es_despedida),
        "cierre_forzado_por_keyword": bool(cierre_forzado_por_keyword),
    }

    # 4.5) Guardar en JSON el estado normalizado de esta sesión
    try:
        _guardar_estado_json(session_id, respuesta)
    except Exception:
        # No rompemos el flujo si falla la escritura
        logger.exception(
            "[JSON] Error al intentar guardar el estado normalizado de la sesión."
        )

    # 5) Empujar estado al visor (panel CRM abajo del UI)
    try:
        await _enviar_slots_al_ui(respuesta)
    except Exception:
        logger.exception("Error al enviar datos al panel CRM del visor")

    logger.info("Respuesta normalizada para /chat/turn: %r", respuesta)

    # Opcional: limpiar meta cuando se termina conversación
    if terminar_flag:
        logger.info(
            "[%s] Conversación marcada como TERMINADA. Limpiando meta de sesión.",
            session_id,
        )
        SESIONES.pop(session_id, None)

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
