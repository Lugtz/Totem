# -*- coding: utf-8 -*-
"""
core.dialog_engine

Motor de diálogo de Nacho (Totem Evolución IA3).

- Lleva estado por sesión (slots, fase, historial).
- Usa OpenAI para generar respuestas naturales.
- Integra knowledge pack de productos Zoho (archivos .txt) desde core/knowledge_pack.
- Usa preguntas de diagnóstico por producto para enriquecer la propuesta.
- SOLO maneja campos requeridos + slots detalle_* (no hay CAMPOS_OPCIONALES globales).
- Expone:
    - CAMPOS_REQUERIDOS
    - procesar_turno_dialogo(session_id, texto_usuario)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

try:
    from core.logger import get_logger  # type: ignore
except Exception:  # fallback simple si aún no tienes core.logger listo
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Rutas y entorno
# ---------------------------------------------------------------------

# ROOT_DIR = raíz del proyecto (Totem/)
ROOT_DIR = Path(__file__).resolve().parent.parent
# CORE_DIR = carpeta core/
CORE_DIR = Path(__file__).resolve().parent

ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# ---------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENAI_APIKEY")
    or os.getenv("AZURE_OPENAI_API_KEY")
)
OPENAI_MODEL = os.getenv("TOTEM_MODEL_NAME", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("TOTEM_TEMPERATURE", "0.4"))

_CLIENT: Any = None
_USE_NEW_CHAT: bool = False

# Intentamos usar el cliente moderno de OpenAI (openai>=1.x)
try:
    from openai import OpenAI  # type: ignore

    if OPENAI_API_KEY:
        _CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    else:
        _CLIENT = OpenAI()
    _USE_NEW_CHAT = True
    logger.info("dialog_engine: usando cliente OpenAI (chat.completions).")
except Exception as e:
    logger.warning("dialog_engine: no se pudo usar OpenAI() moderno: %r", e)
    # Fallback a la librería clásica
    try:
        import openai  # type: ignore

        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        _CLIENT = openai
        _USE_NEW_CHAT = False
        logger.info("dialog_engine: usando openai.ChatCompletion clásico.")
    except Exception as e2:
        logger.error(
            "dialog_engine: no se pudo inicializar ningún cliente OpenAI: %r",
            e2,
        )
        _CLIENT = None

# ---------------------------------------------------------------------
# Campos requeridos del Totem (para la propuesta)
# ---------------------------------------------------------------------

# Campos base obligatorios para poder disparar propuesta / infografía / lead CRM
# ✅ diagnostico DEJA DE SER OBLIGATORIO: ya NO aparece en esta lista
CAMPOS_REQUERIDOS: List[str] = [
    "nombre",
    "empresa",
    "correo",
    "telefono",
    # "diagnostico",  # 👈 diagnostico ahora es opcional
]

CAMPOS_REQUERIDOS_DESC = "\n".join(f"- {c}" for c in CAMPOS_REQUERIDOS)

# 🔴 FORCED EXIT: palabras clave para cortar conversación aunque falten campos
FORCED_EXIT_KEYWORDS: List[str] = [
    "terminar totem",
    # puedes agregar más si quieres:
    # "fin totem",
    # "cortar totem",
]

# ---------------------------------------------------------------------
# Knowledge pack de productos Zoho (core/knowledge_pack)
# ---------------------------------------------------------------------

KNOWLEDGE_DIR = CORE_DIR / "knowledge_pack"
PRODUCT_KNOWLEDGE: Dict[str, str] = {}

# Sinónimos para detectar productos en el texto del usuario
PRODUCT_SYNONYMS: Dict[str, List[str]] = {
    "zoho_crm": ["zoho crm", "crm"],
    "bigin": ["bigin"],
    "salesiq": ["salesiq", "sales iq", "chat en vivo", "chat de ventas"],
    "desk": ["zoho desk", "desk", "mesa de ayuda", "helpdesk"],
    "voice": ["zoho voice", "voz", "telefonia", "telefonía"],
    "assist": ["assist", "remote support", "soporte remoto"],
    "creator": ["zoho creator", "creator", "low code"],
    "campaigns": ["zoho campaigns", "campaigns", "email marketing"],
    "social": ["zoho social", "social media", "redes sociales"],
    "mail": ["zoho mail", "mail", "correo zoho"],
    "workdrive": ["workdrive", "drive", "archivos en la nube"],
    "sign": ["zoho sign", "firma electrónica", "firma digital"],
    "forms": ["zoho forms", "forms", "formularios"],
    "survey": ["zoho survey", "encuestas", "survey"],
    "pagesense": ["pagesense", "experimentos web", "heatmaps"],
    "projects": ["zoho projects", "projects", "gestión de proyectos"],
    "sprints": ["zoho sprints", "agile", "scrum"],
    "zoho_one": ["zoho one"],
    "zoho_crm_plus": ["crm plus", "zoho crm plus"],
    "zoho_workplace": ["workplace", "zoho workplace"],
    "zoho_finance_plus": ["finance plus", "zoho finance plus"],
    "zoho_one_essentials": ["one essentials", "zoho one essentials"],
}

# Config de preguntas de diagnóstico por producto y el nombre de slot detalle_*
PRODUCT_DIAG_CONFIG: Dict[str, Dict[str, str]] = {
    "assist": {
        "slot_detalle": "detalle_assist",
        "pregunta": "¿Su equipo da soporte técnico o ayuda a usuarios remotamente?"
    },
    "backstage": {
        "slot_detalle": "detalle_backstage",
        "pregunta": "¿Organizan eventos o webinars en su empresa?"
    },
    "bigin": {
        "slot_detalle": "detalle_bigin",
        "pregunta": "¿Tu equipo necesita algo sencillo para empezar o buscan algo más completo como Zoho CRM?"
    },
    "billing": {
        "slot_detalle": "detalle_billing",
        "pregunta": "¿Manejan algún servicio por suscripción o planes recurrentes?"
    },
    "campaigns": {
        "slot_detalle": "detalle_campaigns",
        "pregunta": "¿Cómo están haciendo hoy sus envíos de correo masivo?"
    },
    "cliq": {
        "slot_detalle": "detalle_cliq",
        "pregunta": "¿Su equipo usa algún chat interno hoy para coordinarse?"
    },
    "commerce": {
        "slot_detalle": "detalle_commerce",
        "pregunta": "¿Están vendiendo en línea o planean abrir una tienda propia?"
    },
    "connect": {
        "slot_detalle": "detalle_connect",
        "pregunta": "¿Tienen hoy algún espacio interno para compartir información?"
    },
    "creator": {
        "slot_detalle": "detalle_creator",
        "pregunta": "¿Tienen algún proceso que les gustaría digitalizar a su medida?"
    },
    "desk": {
        "slot_detalle": "detalle_desk",
        "pregunta": "¿Cómo están atendiendo hoy las solicitudes o problemas de sus clientes?"
    },
    "expense": {
        "slot_detalle": "detalle_expense",
        "pregunta": "¿Cómo manejan hoy los gastos y comprobaciones del personal?"
    },
    "forms": {
        "slot_detalle": "detalle_forms",
        "pregunta": "¿Qué tipo de formulario necesitas capturar hoy?"
    },
    "fsm": {
        "slot_detalle": "detalle_fsm",
        "pregunta": "¿Tienen técnicos o personal que sale a visitas?"
    },
    "invoice": {
        "slot_detalle": "detalle_invoice",
        "pregunta": "¿Buscas algo completo o solo un sistema sencillo para facturar?"
    },
    "landingpage": {
        "slot_detalle": "detalle_landingpage",
        "pregunta": "¿Están manejando campañas que necesiten una landing dedicada?"
    },
    "lens": {
        "slot_detalle": "detalle_lens",
        "pregunta": "¿Atienden casos donde el cliente necesita mostrar lo que está viendo?"
    },
    "mail": {
        "slot_detalle": "detalle_mail",
        "pregunta": "¿Cómo administran hoy su correo corporativo?"
    },
    "marketing_automation": {
        "slot_detalle": "detalle_marketing_automation",
        "pregunta": "¿Tienen hoy algún flujo de nutrición de prospectos o todo lo hacen manual?"
    },
    "pagesense": {
        "slot_detalle": "detalle_pagesense",
        "pregunta": "¿Tienen una página crítica que quisieran mejorar, como cotización o contacto?"
    },
    "people": {
        "slot_detalle": "detalle_people",
        "pregunta": "¿Cómo están administrando hoy a su personal y asistencias?"
    },
    "projects": {
        "slot_detalle": "detalle_projects",
        "pregunta": "¿En qué tipo de proyectos trabajan hoy?"
    },
    "qntrl": {
        "slot_detalle": "detalle_qntrl",
        "pregunta": "¿Tienen procesos internos con muchas aprobaciones o pasos manuales?"
    },
    "recruit": {
        "slot_detalle": "detalle_recruit",
        "pregunta": "¿Cómo están llevando sus procesos de reclutamiento actualmente?"
    },
    "salesiq": {
        "slot_detalle": "detalle_salesiq",
        "pregunta": "¿Cómo están atendiendo hoy los mensajes que les llegan por la página o redes?"
    },
    "shifts": {
        "slot_detalle": "detalle_shifts",
        "pregunta": "¿Tienen personal en diferentes horarios o rotaciones?"
    },
    "sign": {
        "slot_detalle": "detalle_sign",
        "pregunta": "¿Manejan contratos o documentos que se firman frecuentemente?"
    },
    "social": {
        "slot_detalle": "detalle_social",
        "pregunta": "¿Qué redes manejan hoy y cómo planifican su contenido?"
    },
    "sprints": {
        "slot_detalle": "detalle_sprints",
        "pregunta": "¿Su operación requiere trabajo ágil o desarrollo continuo?"
    },
    "survey": {
        "slot_detalle": "detalle_survey",
        "pregunta": "¿A quién te gustaría encuestar: clientes, usuarios o tu equipo interno?"
    },
    "teaminbox": {
        "slot_detalle": "detalle_teaminbox",
        "pregunta": "¿Tienen correos como info@, ventas@ o soporte@ que recibe todo el equipo?"
    },
    "voice": {
        "slot_detalle": "detalle_voice",
        "pregunta": "¿Cómo manejan hoy las llamadas de su equipo comercial?"
    },
    "workdrive": {
        "slot_detalle": "detalle_workdrive",
        "pregunta": "¿Cómo organizan hoy sus archivos y documentos internos?"
    },
    "workerly": {
        "slot_detalle": "detalle_workerly",
        "pregunta": "¿Trabajan con personal externo o eventual?"
    },
    "writer_sheet_show": {
        "slot_detalle": "detalle_writer_sheet_show",
        "pregunta": "¿Utilizan hoy herramientas colaborativas para documentos?"
    },
    "zia_insights": {
        "slot_detalle": "detalle_zia_insights",
        "pregunta": "¿Qué tipo de métricas te gustaría entender mejor?"
    },
    "zia_predictions": {
        "slot_detalle": "detalle_zia_predictions",
        "pregunta": "¿Qué decisiones te gustaría anticipar en tu operación?"
    },
    "zia_search": {
        "slot_detalle": "detalle_zia_search",
        "pregunta": "¿Tienen muchos datos repartidos entre diferentes aplicaciones?"
    },
    "zia_vision": {
        "slot_detalle": "detalle_zia_vision",
        "pregunta": "¿Manejan documentos que les gustaría digitalizar sin hacerlo manual?"
    },
    "zia_voice": {
        "slot_detalle": "detalle_zia_voice",
        "pregunta": "¿Te gustaría consultar información de tu negocio solo hablando o escribiendo?"
    },
    "zoho_crm": {
        "slot_detalle": "detalle_zoho_crm",
        "pregunta": "¿Cómo están manejando hoy sus prospectos y oportunidades?"
    },
    "zoho_crm_plus": {
        "slot_detalle": "detalle_zoho_crm_plus",
        "pregunta": "¿Buscas una visión completa del cliente en ventas y servicio?"
    },
    "zoho_finance_plus": {
        "slot_detalle": "detalle_zoho_finance_plus",
        "pregunta": "¿Dónde sienten más fricción hoy: facturación, inventarios, cobranza o conciliación?"
    },
    "zoho_one": {
        "slot_detalle": "detalle_zoho_one",
        "pregunta": "Si tuvieras que elegir por dónde empezar con Zoho One, ¿qué área de tu empresa es hoy la más crítica?"
    },
    "zoho_one_essentials": {
        "slot_detalle": "detalle_zoho_one_essentials",
        "pregunta": "¿Prefieren empezar con lo esencial o buscan algo muy completo desde el inicio?"
    },
    "zoho_workplace": {
        "slot_detalle": "detalle_zoho_workplace",
        "pregunta": "¿El reto principal está en la colaboración interna o en la operación del negocio?"
    },
}


def _cargar_knowledge_pack() -> None:
    """
    Carga todos los .txt de core/knowledge_pack/ en PRODUCT_KNOWLEDGE.
    Cada archivo .txt se indexa por su nombre base (sin extensión), en minúsculas.
    Ejemplo: core/knowledge_pack/zoho_crm.txt -> key "zoho_crm".
    """
    if not KNOWLEDGE_DIR.exists():
        logger.warning(
            "dialog_engine: carpeta core/knowledge_pack no encontrada en %s. "
            "Copia ahí los .txt del ZIP para activar el contexto Zoho.",
            KNOWLEDGE_DIR,
        )
        return

    num_docs = 0
    for path in KNOWLEDGE_DIR.glob("*.txt"):
        try:
            key = path.stem.lower()
            text = path.read_text(encoding="utf-8").strip()
            if text:
                PRODUCT_KNOWLEDGE[key] = text
                num_docs += 1
        except Exception as e:
            logger.warning("dialog_engine: no se pudo leer %s: %r", path, e)

    logger.info(
        "dialog_engine: knowledge_pack cargado con %d documentos desde %s.",
        num_docs,
        KNOWLEDGE_DIR,
    )


_cargar_knowledge_pack()


def _detectar_productos_en_texto(texto: str) -> List[str]:
    """
    Detecta cuáles productos Zoho parecen mencionarse en el texto,
    usando el nombre del archivo y algunos sinónimos.
    """
    t = (texto or "").lower()
    encontrados: List[str] = []

    # 1) Si el nombre base del knowledge aparece tal cual
    for key in PRODUCT_KNOWLEDGE.keys():
        base = key.replace("_", " ")
        if base and base in t:
            encontrados.append(key)

    # 2) Si algún sinónimo coincide
    for key, synonyms in PRODUCT_SYNONYMS.items():
        for trig in synonyms:
            if trig and trig.lower() in t and key not in encontrados:
                encontrados.append(key)
                break

    return encontrados


def _construir_contexto_zoho(session_state: Dict[str, Any], mensaje_usuario: str) -> str:
    """
    Arma un bloque de texto con:
    - Productos Zoho relevantes según el mensaje actual y el historial.
    - Descripciones desde PRODUCT_KNOWLEDGE.
    - Preguntas de diagnóstico sugeridas + nombre de slot detalle_*.
    """
    texto_ref = mensaje_usuario or ""
    for msg in session_state.get("historial", [])[-4:]:
        contenido = msg.get("content", "")
        if isinstance(contenido, str):
            texto_ref += " " + contenido

    keys = _detectar_productos_en_texto(texto_ref)
    if not keys:
        return ""

    # Limitamos a máximo 5 productos por turno
    keys = keys[:5]

    partes: List[str] = []

    for key in keys:
        doc = PRODUCT_KNOWLEDGE.get(key, "").strip()
        if not doc:
            continue
        partes.append(f"[{key}]\n{doc}\n")

    # Preguntas de diagnóstico solo para los productos detectados
    diag_lines: List[str] = []
    for key in keys:
        cfg = PRODUCT_DIAG_CONFIG.get(key)
        if not cfg:
            continue
        diag_lines.append(
            f"- Para {key} (guarda la respuesta en el slot \"{cfg['slot_detalle']}\"): {cfg['pregunta']}"
        )

    if diag_lines:
        partes.append("\nPREGUNTAS_DE_DIAGNOSTICO_SUGERIDAS_PARA_ESTOS_PRODUCTOS:\n")
        partes.extend(diag_lines)

    return "\n\n".join(partes)


# ---------------------------------------------------------------------
# Estado en memoria por sesión
# ---------------------------------------------------------------------

ESTADO_SESIONES: Dict[str, Dict[str, Any]] = {}


def _get_session_state(session_id: str) -> Dict[str, Any]:
    """
    Crea o recupera el estado de una sesión:
    - fase: bienvenida / captura_datos / confirmacion_final / despedida
    - slots: dict de campos capturados
    - historial: mensajes previos (user/assistant)
    """
    if session_id not in ESTADO_SESIONES:
        ESTADO_SESIONES[session_id] = {
            "fase": "bienvenida",
            "slots": {},
            "historial": [],
        }
    return ESTADO_SESIONES[session_id]


# ---------------------------------------------------------------------
# Prompt del sistema
# ---------------------------------------------------------------------

# Pequeña descripción de los slots detalle_* para el prompt
DETALLE_SLOTS_DESC_LINEAS: List[str] = []
for key, cfg in PRODUCT_DIAG_CONFIG.items():
    DETALLE_SLOTS_DESC_LINEAS.append(
        f"- {key}: guarda la respuesta de diagnóstico en el slot \"{cfg['slot_detalle']}\"."
    )
DETALLE_SLOTS_DESC = "\n".join(DETALLE_SLOTS_DESC_LINEAS)

SYSTEM_PROMPT = f"""
Eres Nacho, el asistente inteligente de Evolución i3 que atiende a las personas en un tótem de recepción.
Hablas en ESPAÑOL LATINO, tono cálido, relajado y MUY natural, como si platicaras con alguien en persona
(o en una llamada), nada robotizado.

🎭 TU ROL
- Eres un asistente inteligente, no solo un generador de propuestas.
- Puedes platicar prácticamente de CUALQUIER tema que la persona saque:
  trabajo, tecnología, vida diaria, dudas random, etc.
- Siempre respondes primero de forma natural y empática a lo que te digan.
- Al mismo tiempo, tu objetivo de fondo es entender el contexto de la persona y su empresa
  para poder preparar una PROPUESTA y una INFOGRAFÍA personalizada cuando tenga sentido.

📌 DATOS QUE NECESITAS (SLOTS PRINCIPALES)
Tienes una lista interna de datos obligatorios que hay que recopilar para poder armar bien la propuesta:

{CAMPOS_REQUERIDOS_DESC}

Y también es MUY útil, aunque no salga en esa lista, saber de forma clara la solución que quieren implementar:
- solucion_a_implementar: una frase corta que describa qué quieren lograr o qué solución buscan contigo.

Tú NO mencionas la palabra "slots" ni "campos requeridos". Solo usas esa lista para guiar tus preguntas mentalmente.

🧪 SLOTS OPCIONALES DE DIAGNÓSTICO POR PRODUCTO ZOHO
Existen SLOTS OPCIONALES DE DIAGNÓSTICO por producto Zoho, por ejemplo:

{DETALLE_SLOTS_DESC}

Reglas para estos:
- Solo los usas cuando la persona hable de ese producto, o cuando de forma natural le preguntes algo relacionado.
- La respuesta debe quedarse en el slot correspondiente como un TEXTO CORTO en español (no una biblia).
- No inventes información; solo guarda lo que realmente haya dicho.
- No hagas más de 1 pregunta de diagnóstico extra por turno y máximo 3 en toda la conversación.

📚 CONTEXTO ZOHO (KNOWLEDGE PACK)
En cada turno recibirás un bloque "CONTEXTO_ZOHO_RELEVANTE" con:
- Descripciones de productos Zoho desde el knowledge pack.
- Una sección "PREGUNTAS_DE_DIAGNOSTICO_SUGERIDAS_PARA_ESTOS_PRODUCTOS" con:
  - una pregunta sugerida por producto, y
  - el nombre del slot detalle_* donde debes guardar la respuesta.

Úsalo así:
- Te sirve para explicar mejor los beneficios con ejemplos sencillos.
- Te ayuda a elegir una posible pregunta extra de diagnóstico SI tiene sentido.
- Nunca menciones que estás leyendo documentos ni nombres internos técnicos.

🧩 CÓMO HACER PREGUNTAS (SUAVE, NO INVASIVO)
- Nunca digas cosas como:
  - "te voy a hacer unas preguntas"
  - "necesito llenar unos campos"
  - "vamos a completar un formulario"
- En lugar de eso, integra las preguntas en la plática, por ejemplo:
  - "Por cierto, ¿cómo se llama tu empresa?"
  - "¿A qué correo te gustaría que te mandemos la información?"
  - "¿Cómo están manejando hoy a sus clientes o prospectos? Cuéntame tantito."
- Haz UNA sola pregunta principal por turno. Máximo dos si van muy relacionadas.
- Si ya tienes un dato (por ejemplo el nombre), no lo vuelvas a pedir a menos que haya confusión.

🔄 CORRECCIÓN Y ACTUALIZACIÓN DE DATOS (MUY IMPORTANTE)
- Si la persona dice que se equivocó en algún dato (nombre, correo, teléfono, empresa, etc.) y proporciona uno nuevo,
  SIEMPRE toma el ÚLTIMO dato como el correcto.
- En esos casos, incluye en "slots_detectados" el dato ACTUALIZADO, aunque ya lo hubieras llenado antes
  en un turno anterior.
- Si la persona pide eliminar o cancelar un dato (por ejemplo, "mejor no te doy mi teléfono"),
  puedes mandar ese slot con valor vacío (""), null o similar, para que el sistema lo considere nuevamente como pendiente.
- Resumen:
  - Última información = la que manda.
  - Si un slot viene vacío o null, se borra / limpia en el estado interno.

🗣 ESTILO DE CONVERSACIÓN (MODO LLAMADA / STREAMING / MUCHA FLUIDEZ)
- Piensa siempre que estás en una llamada MUY fluida entre dos personas.
- Respondes en ESPAÑOL LATINO neutro, con expresiones naturales:
  - "súper", "perfecto", "me hace sentido", "claro", "dale", "¿cómo ves?", etc.
- Tus respuestas deben ser CORTAS para que el TTS en streaming se sienta natural:
  - Máximo 2–3 frases por turno.
  - Evita párrafos largos o discursos muy técnicos.
- Puedes usar pequeñas muletillas naturales (sin exagerar), por ejemplo:
  - "Mira...", "Va, entiendo...", "Ok, entonces...", antes de explicar algo.

🎯 ESTRATEGIA PARA OBTENER INFORMACIÓN
1) INICIO
   - Saluda de forma breve y cálida.
   - Preséntate como asistente inteligente:
     - "Soy Nacho, el asistente inteligente de Evolución i3."
   - Explica en una sola frase lo que puedes hacer:
     - "Puedo platicar contigo, resolver dudas y, si quieres, te ayudo a preparar una propuesta con una infografía para tu empresa."
   - En el PRIMER turno, SIEMPRE pide directamente el NOMBRE de la persona, por ejemplo:
     - "Para empezar, ¿cómo te llamas?"
     - "¿Cuál es tu nombre?"
   - Ya después podrás preguntar por la empresa, el rol, etc. en los siguientes turnos.

2) DESCUBRIMIENTO
   - Mezcla preguntas de contexto con lo que la persona va diciendo.
   - Siempre que detectes algo útil para la propuesta (problemas, herramientas que usan, número de personas, etc.), guárdalo como dato interno.
   - No dispares preguntas una tras otra; deja que la persona hable y tú guías con preguntas pequeñas.

3) CUANDO YA TIENES CASI TODO
   - Si ya tienes lo mínimo para armar la propuesta, puedes decir algo como:
     - "Con lo que me contaste ya puedo ir preparando algo para tu empresa."
   - Después puedes ofrecer afinar detalles, pero sin presionar:
     - "Si quieres, afinamos un poquito más qué parte te preocupa más ahora mismo."

4) CUANDO YA TENGAS TODO PARA LA PROPUESTA
   - Cuando consideres que "campos_completos_model" debe ser true (ya tenemos todos los datos obligatorios),
     en "assistant_text" menciona de forma breve y natural algo como:
     - "Con todo lo que me compartiste ya puedo mandar tu información a nuestro equipo para preparar la propuesta."
     - "En segundo plano se estará generando tu propuesta y una infografía personalizada."
     - "La infografía la vamos a mandar a tu correo y también la imprimimos aquí para ti."
     - "Además, uno de nuestros asesores de Evolución i3 se pondrá en contacto contigo."
   - No repitas este mensaje muchas veces: basta con mencionarlo una vez cuando se completa la información.

5) SI LA PERSONA NO QUIERE DAR ALGÚN DATO
   - Acepta la respuesta sin insistir.
   - Cambia a una pregunta más general o sigue la conversación por otro lado.
   - Ejemplo:
     - "No te preocupes, con lo que ya me contaste también puedo ir armando algo."

6) AÚN SI YA TENGO TODOS LOS DATOS
   - La conversación NO se cierra automáticamente.
   - La persona puede seguir preguntando sobre Zoho, sobre su empresa o de otros temas.
   - También puede aclarar o corregir datos (nombre, correo, teléfono, empresa, etc.).
   - Solo cierras la conversación si la persona claramente se despide.

🌀 SI LA PERSONA SOLO PREGUNTA ALGO SUELTO / RANDOM
- Si en el turno ACTUAL la persona solamente:
  - hace una pregunta puntual (por ejemplo: “¿qué es un CRM?”, “qué es Zoho One?”, “ustedes qué hacen?”), o
  - comenta algo que NO tiene que ver con su empresa, procesos, Zoho o una propuesta,
  entonces debes comportarte como en una charla normal:
  - Responde primero la duda de forma clara y sencilla, en 1–3 frases máximo.
  - NO intentes pedir nombre, empresa, correo ni teléfono en ese mismo turno.
  - Puedes cerrar con una invitación suave, por ejemplo:
    - "Si después te interesa, con gusto vemos tu caso y te ayudo a armar algo a la medida."
- Solo empiezas a hacer preguntas de datos (nombre, empresa, correo…) cuando:
  - la persona muestre interés real (por ejemplo: “quiero algo así para mi empresa”, “me interesa una propuesta”, “me sirve para mi negocio”), o
  - pregunte explícitamente cómo le podríamos ayudar a ELLA o a SU EMPRESA.

En estos casos de pregunta suelta:
- "slots_detectados" normalmente puede ir vacío o solo con lo que de verdad haya dicho (por ejemplo, si menciona su empresa de forma espontánea).
- "campos_pendientes_model" debe seguir marcando los datos que faltan.
- "campos_completos_model" casi siempre será false.

🚫 COSAS QUE NO DEBES HACER NUNCA
- No sonar robótico, no usar frases ultra formales que nadie diría hablando normal.
- No mostrar listas largas ni texto técnico si la persona no lo pidió.
- No escribir más de 3–4 frases en una sola respuesta.
- No decir "slot", "campo requerido", "plantilla", "knowledge pack", "API" ni nada interno del sistema.
- No responder con markdown, emojis raros o formato de código; solo texto plano conversacional.

📦 FORMATO DE SALIDA (MUY IMPORTANTE)
Siempre debes responder en UN SOLO JSON válido (sin texto extra, sin comentarios, sin backticks),
con esta estructura:

{{
  "assistant_text": "lo que Nacho diría en voz alta al visitante, en tono MUY natural y fluido",
  "slots_detectados": {{
      "nombre": "si lo detectas aquí",
      "empresa": "si lo detectas aquí",
      "correo": "si lo detectas aquí",
      "telefono": "si describe su teléfono o WhatsApp",
      "solucion_a_implementar": "si describe su solución a implementar",
      "detalle_zoho_crm": "si describe su situación de CRM",
      "detalle_salesiq": "si describe su situación de chats/mensajes web",
      "detalle_desk": "si describe su situación de soporte"
  }},
  "campos_pendientes_model": ["lista de campos obligatorios que sigan faltando"],
  "campos_completos_model": false,
  "es_despedida_model": false
}}

- "assistant_text" debe ser una frase lista para decirse en voz alta, muy fluida y humana.
- "slots_detectados" SOLO debe contener los campos que hayas detectado en ese turno.
- "campos_pendientes_model" es tu mejor estimación de qué campos obligatorios siguen faltando.
- "campos_completos_model" debe ser true SOLO si crees que ya están todos los obligatorios.
- "es_despedida_model" debe ser true SOLO cuando estés claramente cerrando la conversación
  (por ejemplo: el usuario se despide, tú respondes con un agradecimiento y un cierre amable),
  y ya no planeas seguir haciendo más preguntas en turnos siguientes.
"""


def _build_user_content(
    session_state: Dict[str, Any],
    texto_usuario: str,
) -> str:
    """
    Construye el contenido del mensaje de usuario que verá el modelo:
    - Mensaje real del usuario.
    - Estado de slots y fase.
    - Contexto Zoho relevante (si aplica), con preguntas de diagnóstico sugeridas.
    """
    slots = session_state.get("slots", {}) or {}
    fase = session_state.get("fase", "bienvenida")
    campos_pendientes_calc = [c for c in CAMPOS_REQUERIDOS if not slots.get(c)]

    contexto_zoho = _construir_contexto_zoho(session_state, texto_usuario)

    partes: List[str] = []
    partes.append(f'MENSAJE_USUARIO:\n"{texto_usuario}"\n')
    partes.append("ESTADO_ACTUAL:\n")
    partes.append(f"- fase_actual: {fase}\n")
    partes.append(
        f"- slots_actuales: {json.dumps(slots, ensure_ascii=False)}\n"
    )
    partes.append(
        f"- campos_requeridos: {json.dumps(CAMPOS_REQUERIDOS, ensure_ascii=False)}\n"
    )
    partes.append(
        f"- campos_pendientes_calculados: {json.dumps(campos_pendientes_calc, ensure_ascii=False)}\n"
    )

    if contexto_zoho:
        partes.append("\nCONTEXTO_ZOHO_RELEVANTE:\n")
        partes.append(contexto_zoho)

    return "\n".join(partes)


def _build_messages(
    session_state: Dict[str, Any],
    texto_usuario: str,
) -> List[Dict[str, str]]:
    """
    Arma la lista de mensajes para OpenAI:
    - system: prompt general
    - historial: últimas interacciones (solo texto natural, no JSON interno)
    - user: mensaje actual + estado + contexto Zoho
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Historial: solo assistant_text y texto del usuario, no JSON interno
    for msg in session_state.get("historial", [])[-8:]:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append(
                {
                    "role": msg["role"],
                    "content": str(msg["content"]),
                }
            )

    user_content = _build_user_content(session_state, texto_usuario)
    messages.append({"role": "user", "content": user_content})
    return messages


def _llamar_modelo(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Llama a OpenAI (chat.completions o ChatCompletion clásico) y devuelve
    un dict con al menos:
        - assistant_text: str
        - slots_detectados: dict
        - campos_pendientes_model: list
        - campos_completos_model: bool
        - es_despedida_model: bool (opcional, por defecto False)
    """
    if _CLIENT is None:
        logger.error(
            "dialog_engine: OpenAI no está inicializado. Revisa tu OPENAI_API_KEY."
        )
        return {
            "assistant_text": (
                "Estoy teniendo un problema técnico, pero sigamos. "
                "Por favor, dime tu nombre completo y el nombre de tu empresa."
            ),
            "slots_detectados": {},
            "campos_pendientes_model": CAMPOS_REQUERIDOS,
            "campos_completos_model": False,
            "es_despedida_model": False,
        }

    try:
        if _USE_NEW_CHAT:
            # Cliente moderno OpenAI (1.x): client.chat.completions.create(...)
            resp = _CLIENT.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=OPENAI_TEMPERATURE,
            )
            text = resp.choices[0].message.content  # type: ignore[assignment]
        else:
            # Cliente clásico (openai.ChatCompletion.create)
            resp = _CLIENT.ChatCompletion.create(  # type: ignore[attr-defined]
                model=OPENAI_MODEL,
                messages=messages,
                temperature=OPENAI_TEMPERATURE,
            )
            text = resp.choices[0].message["content"]  # type: ignore[index]

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("La respuesta JSON no es un objeto dict.")
        return data

    except Exception as e:
        logger.exception(
            "dialog_engine: error llamando a OpenAI o parseando JSON: %r",
            e,
        )
        # Fallback conservador
        return {
            "assistant_text": (
                "Tuve un pequeño problema técnico, pero no te preocupes. "
                "Cuéntame por favor tu nombre y el nombre de tu empresa."
            ),
            "slots_detectados": {},
            "campos_pendientes_model": CAMPOS_REQUERIDOS,
            "campos_completos_model": False,
            "es_despedida_model": False,
        }


def _normalizar_correo(valor: Any) -> Any:
    """
    Normaliza correos que vienen del ASR, por ejemplo:
    - 'usuario arroba gmail.com'
    - 'usuarioarrobagmail.com'
    - con espacios antes/después

    Reglas:
    - Si no es string, se regresa tal cual.
    - Quita TODOS los espacios.
    - Pasa todo a minúsculas.
    - Reemplaza 'arroba' por '@' si no existía.
    """
    if not isinstance(valor, str):
        return valor

    # Quitar espacios al inicio/fin
    correo = valor.strip()

    # Quitar TODOS los espacios intermedios (y otros espacios raros)
    # 'usuario arroba gmail.com' -> 'usuarioarrobagmail.com'
    correo = "".join(correo.split())

    # Normalizar a minúsculas
    correo = correo.lower()
    correo = correo.replace("punto",".")
    # Si ya trae @, lo regresamos así
    if "@" in correo:
        return correo

    # Reemplazamos 'arroba' por '@'
    # Ej: 'usuarioarrobagmail.com' -> 'usuario@gmail.com'
    if "arroba" in correo:
        correo = correo.replace("arroba", "@")

    return correo


def procesar_turno_dialogo(
    session_id: str,
    texto_usuario: str,
) -> Tuple[str, Dict[str, Any], List[str], bool, bool]:
    """
    Función principal llamada desde amain.py (/chat/turn).

    Recibe:
        - session_id: ID de la sesión de la cámara / visitante.
        - texto_usuario: lo que dijo la persona (ya transcrito).

    Devuelve:
        - assistant_text: lo que Nacho va a decir.
        - slots: dict con los campos acumulados de la sesión.
        - campos_pendientes: lista de campos obligatorios que faltan (calculado).
        - campos_completos: bool -> True si ya están todos los obligatorios.
        - es_despedida: bool -> True si este turno se considera una despedida.
    """
    texto_usuario = (texto_usuario or "").strip()
    session_state = _get_session_state(session_id)

    # 🔴 FORCED EXIT: si el usuario dice la palabra clave, cortamos aquí
    texto_lower = texto_usuario.lower()
    for kw in FORCED_EXIT_KEYWORDS:
        if kw in texto_lower:
            slots = session_state.get("slots", {})
            if not isinstance(slots, dict):
                slots = {}
            campos_pendientes = [c for c in CAMPOS_REQUERIDOS if not slots.get(c)]
            campos_completos = len(campos_pendientes) == 0
            assistant_text = (
                "Perfecto, lo dejamos hasta aquí. "
                "Muchas gracias por tu tiempo y que tengas un excelente día."
            )
            es_despedida = True

            # Guardamos en historial para coherencia
            session_state.setdefault("historial", []).append(
                {"role": "user", "content": texto_usuario}
            )
            session_state["historial"].append(
                {"role": "assistant", "content": assistant_text}
            )

            logger.info(
                "[dialog_engine] session_id=%s | FORCED_EXIT por keyword=%r | slots=%r | pendientes=%r | completos=%s",
                session_id,
                kw,
                slots,
                campos_pendientes,
                campos_completos,
            )
            return assistant_text, slots, campos_pendientes, campos_completos, es_despedida

    if not texto_usuario:
        assistant_text = (
            "No escuché nada claro. ¿Me podrías repetir, por favor?"
        )
        session_state.setdefault("historial", []).append(
            {"role": "user", "content": texto_usuario}
        )
        session_state["historial"].append(
            {"role": "assistant", "content": assistant_text}
        )
        slots = session_state.get("slots", {})
        if not isinstance(slots, dict):
            slots = {}
        campos_pendientes = [c for c in CAMPOS_REQUERIDOS if not slots.get(c)]
        campos_completos = len(campos_pendientes) == 0
        es_despedida = False
        return assistant_text, slots, campos_pendientes, campos_completos, es_despedida

    # ✨ Detección de correcciones explícitas para datos ya dados
    correcciones = [
        "me llamo",
        "mi nombre es",
        "no es ese",
        "corrige",
        "mejor usa",
        "te di mal",
        "no ese",
        "cámbialo",
        "cambialo",
        "no es correcto",
        "no era ese correo",
        "no era ese teléfono",
        "no era ese telefono",
    ]
    if any(p in texto_usuario.lower() for p in correcciones):
        session_state["fase"] = "correccion_datos"

    # Construimos mensajes para el modelo
    messages = _build_messages(session_state, texto_usuario)
    data = _llamar_modelo(messages)

    assistant_text = str(data.get("assistant_text") or "").strip()
    if not assistant_text:
        assistant_text = (
            "Perfecto, te escucho. Cuéntame por favor cuál es tu nombre y el nombre de tu empresa."
        )

    # Slots detectados en este turno (solo los nuevos/actualizados)
    slots_detectados = data.get("slots_detectados") or {}
    if not isinstance(slots_detectados, dict):
        slots_detectados = {}

    # Actualizamos slots acumulados de la sesión
    slots = session_state.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}

    # 🔄 Lógica de actualización/corrección de datos
    # - Si llega un valor nuevo, sobreescribe.
    # - Si llega vacío / None / [] / "__BORRAR__", elimina ese slot (lo vuelve pendiente).
    # - Correo se normaliza si viene con 'arroba'.
    for k, v in slots_detectados.items():
        # Normalizar correo si aplica
        if k in ("correo", "email", "Email"):
            v = _normalizar_correo(v)

        if v in (None, "", [], "__BORRAR__"):
            if k in slots:
                del slots[k]
        else:
            slots[k] = v

    # 👇 Construir OBJETIVO dinámico a partir de solucion_a_implementar + todos los detalle_*
    partes_objetivo: List[str] = []

    # 1) Si hay solucion_a_implementar, va primero
    sol_impl = slots.get("solucion_a_implementar")
    if isinstance(sol_impl, str) and sol_impl.strip():
        partes_objetivo.append(sol_impl.strip())

    # 2) Agregamos todos los slots que empiecen con detalle_*
    for key, value in slots.items():
        if not key.startswith("detalle_"):
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        # ejemplo: "detalle_zoho_crm manejan los prospectos uno a uno"
        partes_objetivo.append(f"{key} {value.strip()}")

    # 3) Si juntamos algo, generamos OBJETIVO
    if partes_objetivo:
        slots["OBJETIVO"] = " | ".join(partes_objetivo)

    # Actualizamos en el estado (por si agregamos/actualizamos OBJETIVO)
    session_state["slots"] = slots

    # Recalculamos campos pendientes en base a CAMPOS_REQUERIDOS
    campos_pendientes = [c for c in CAMPOS_REQUERIDOS if not slots.get(c)]
    campos_completos = len(campos_pendientes) == 0

    # Fase simple según avance
    fase_anterior = session_state.get("fase", "bienvenida")
    if fase_anterior == "bienvenida" and any(
        slots.get(c) for c in CAMPOS_REQUERIDOS
    ):
        session_state["fase"] = "captura_datos"
    if campos_completos:
        # Ya tengo todo, pero mantengo la conversación abierta
        session_state["fase"] = "confirmacion_final_pero_conversacion_abierta"

    # Bandera de despedida que viene del modelo (controlada por fase)
    es_despedida_raw = data.get("es_despedida_model", False)

    if session_state.get("fase") == "confirmacion_final_pero_conversacion_abierta":
        # En esta fase solo nos despedimos si el usuario claramente se despide
        texto_l = texto_usuario.lower()
        if any(x in texto_l for x in ["adios", "adiós", "gracias", "nos vemos", "hasta luego", "bye"]):
            es_despedida = True
        else:
            es_despedida = False
    else:
        # Caso normal: respetamos la señal del modelo
        if isinstance(es_despedida_raw, str):
            es_despedida = es_despedida_raw.lower() in ("true", "1", "yes", "si", "sí")
        else:
            es_despedida = bool(es_despedida_raw)

    # 🚀 REGLA DURA: en la presentación SIEMPRE pide el nombre
    if fase_anterior == "bienvenida" and not slots.get("nombre"):
        assistant_text = (
            "Hola, soy Nacho, el asistente inteligente de Evolución i3. "
            "Para empezar, ¿cómo te llamas?"
        )

    # Guardamos historial solo con texto normal (sin JSON interno)
    session_state.setdefault("historial", []).append(
        {"role": "user", "content": texto_usuario}
    )
    session_state["historial"].append(
        {"role": "assistant", "content": assistant_text}
    )

    logger.info(
        "[dialog_engine] session_id=%s | fase=%s | slots=%r | pendientes=%r | completos=%s | es_despedida=%s",
        session_id,
        session_state.get("fase"),
        slots,
        campos_pendientes,
        campos_completos,
        es_despedida,
    )

    return assistant_text, slots, campos_pendientes, campos_completos, es_despedida
