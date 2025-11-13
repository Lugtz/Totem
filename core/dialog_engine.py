# -*- coding: utf-8 -*-
"""
core/dialog_engine.py

Motor conversacional para el Totem Evolución IA3.

- Usa OpenAI (GPT-4o / GPT-4o-mini) para conversar como "Nacho".
- Va llenando campos para una propuesta (CAMPOS_REQUERIDOS + opcionales).
- Devuelve el texto que se hablará + slots actuales + campos pendientes.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from core.logger import get_logger
from core.untils import estado_global, reiniciar_estado

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# 1) Cargar .env desde la raíz del proyecto
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent.resolve()
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    logger.info("core.dialog_engine: .env cargado desde %s", ENV_PATH)
else:
    load_dotenv()
    logger.warning(
        "core.dialog_engine: No se encontró .env en %s; "
        "se intentó load_dotenv() sin ruta.",
        ENV_PATH,
    )

# ----------------------------------------------------------------------
# 2) Validar y crear cliente OpenAI con API key explícita
# ----------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY no está configurada.\n\n"
        "Verifica que exista el archivo .env en la carpeta raíz del proyecto "
        f"({ROOT_DIR}) y que contenga una línea como:\n\n"
        "OPENAI_API_KEY=tu_clave_aqui\n"
    )

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

logger.info(
    "core.dialog_engine: Cliente OpenAI inicializado con modelo %s",
    OPENAI_MODEL,
)

# ----------------------------------------------------------------------
# 3) Definición de campos requeridos y opcionales
# ----------------------------------------------------------------------
# 👉 Estos son los que SÍ se deben preguntar (obligatorios)
CAMPOS_REQUERIDOS: Dict[str, str] = {
    "nombre": "el nombre completo de la persona de contacto",
    "email": "el correo electrónico de contacto (si la persona quiere compartirlo)",
    "empresa": "el nombre de la empresa u organización",
    "solucion": "el objetivo o solución/proyecto que le interesa (por ejemplo, Totem de recepción, implementación de Zoho, mejora de CRM, etc.)",
    "duracion": "la duración aproximada del proyecto (en semanas o meses)",
    "moneda": "la moneda principal del proyecto (MXN, USD, etc.)",
    "semana_piloto": "cuándo le gustaría arrancar el piloto o inicio del proyecto (semana o mes aproximado)",
    "licencia": "la licencia o paquete de Zoho que piensa usar (Zoho One, CRM Plus, etc.)",
    "version": "la versión de la licencia (Standard, Professional, Enterprise, etc.)",
}

# 👉 Estos son OPCIONALES: NO se preguntan, solo se llenan si la persona lo menciona
CAMPOS_OPCIONALES: Dict[str, str] = {
    "precio": "el presupuesto aproximado o rango de inversión del proyecto (solo si la persona lo menciona)",
    "iva": "el detalle del IVA (incluido/no incluido/porcentaje, solo si lo menciona)",
    "precio_mensual_usuario": "el precio mensual por usuario de las licencias (si lo menciona)",
    "precio_anual_usuario": "el precio anual por usuario de las licencias (si lo menciona)",
    "moneda_licencias": "la moneda en la que se cobran las licencias (MXN, USD, etc., solo si lo menciona)",
}

TODOS_LOS_CAMPOS: Dict[str, str] = {**CAMPOS_REQUERIDOS, **CAMPOS_OPCIONALES}

CAMPOS_REQUERIDOS_DESC = "\n".join(
    [f"- {k}: {v}" for k, v in CAMPOS_REQUERIDOS.items()]
)
CAMPOS_OPCIONALES_DESC = "\n".join(
    [f"- {k}: {v}" for k, v in CAMPOS_OPCIONALES.items()]
)

SYSTEM_PROMPT = f"""
Eres Nacho, el avatar de Evolución i3 que atiende a las personas en un tótem de recepción.
Hablas en ESPAÑOL LATINO, tono cálido, profesional y muy natural (como humano, no como robot).

Tu trabajo:
1. Dar la bienvenida, explicar que puedes generar una propuesta y una infografía personalizada para su empresa.
2. Hacer preguntas suaves, de a poco, para llenar SOLO estos datos principales (obligatorios):

{CAMPOS_REQUERIDOS_DESC}

3. Además hay datos OPCIONALES que solo debes registrar si la persona los menciona espontáneamente.
   NO los preguntes tú directamente, no insistas en ellos:

{CAMPOS_OPCIONALES_DESC}

4. Mientras todavía falte AL MENOS UNO de los datos OBLIGATORIOS, SIEMPRE debes hacer
   al menos UNA pregunta clara para obtener uno de esos datos.
   - Prioriza en este orden aproximado: nombre, empresa, email, solucion (objetivo),
     duración, semana_piloto, licencia, versión, moneda.
5. Una vez que tengas al menos nombre, empresa y solucion (objetivo),
   ya puedes decir que vas a preparar la propuesta y la infografía personalizadas
   con base en lo que la persona te compartió.

Respondes SIEMPRE en formato JSON ESTRICTO, sin texto extra, con esta forma:

{{
  "assistant_text": "Lo que le dirás en voz alta al visitante, en español latino, máximo 3-4 frases.",
  "slots_detectados": {{
    "nombre": "... o null si no lo detectas",
    "email": "...",
    "empresa": "...",
    "solucion": "...",
    "duracion": "...",
    "moneda": "...",
    "semana_piloto": "...",
    "licencia": "...",
    "version": "...",

    "precio": "...",
    "iva": "...",
    "precio_mensual_usuario": "...",
    "precio_anual_usuario": "...",
    "moneda_licencias": "..."
  }}
}}

Reglas IMPORTANTES:
- NO escribas nada fuera del JSON.
- NO uses bloques de código ni ``` ni etiquetas tipo markdown.
- Si no detectas un dato, deja ese valor en null o en cadena vacía "".
- NO preguntes de forma directa por IVA, presupuesto/precio, ni precios por usuario.
  Si la persona menciona esos datos, los capturas en los slots, pero no los pides tú.
- Habla siempre en primera persona como “Nacho”, y menciona Evolución i3
  de forma natural cuando tenga sentido.
- Cuando ya tengas al menos nombre, empresa y solucion, puedes decir algo como:
  "Con esta información ya puedo preparar tu propuesta y tu infografía personalizada."
"""

# ----------------------------------------------------------------------
# 4) Helpers de sesión y parsing de JSON
# ----------------------------------------------------------------------
def _get_session(session_id: str) -> Dict:
    if session_id not in estado_global:
        reiniciar_estado(session_id)
    return estado_global[session_id]


def _extraer_json_desde_respuesta(raw: str) -> Dict:
    """
    Intenta extraer el JSON de la respuesta del modelo.
    Soporta respuestas con ```json ... ``` o JSON directo.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}

    if "```" in raw:
        inicio = raw.find("```")
        fin = raw.rfind("```")
        if inicio != -1 and fin != -1 and fin > inicio:
            contenido = raw[inicio + 3: fin].strip()
            if contenido.lower().startswith("json"):
                contenido = contenido[4:].strip()
            raw = contenido

    try:
        return json.loads(raw)
    except Exception as e:
        logger.exception(
            "No se pudo parsear JSON de la respuesta del modelo: %s", e
        )
        return {}


# ----------------------------------------------------------------------
# 5) Función principal: procesar turno de diálogo
# ----------------------------------------------------------------------
def procesar_turno_dialogo(
    session_id: str, user_text: str
) -> Tuple[str, Dict[str, str], List[str], bool]:
    """
    Procesa un turno de diálogo:
    - Envía todo el contexto a OpenAI.
    - Devuelve:
        texto_respuesta: lo que dirá Nacho.
        slots: todos los campos acumulados hasta ahora.
        campos_pendientes: lista de claves OBLIGATORIAS aún vacías.
        campos_completos: True si ya están todos los OBLIGATORIOS.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        user_text = (
            "Solo estoy saludando, aún no he dicho nada relevante, pero "
            "quiero que sigas la conversación de forma natural."
        )

    sesion = _get_session(session_id)
    slots_actuales: Dict[str, str] = sesion.get("slots", {})
    history: List[Dict[str, str]] = sesion.get("history", [])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turno in history:
        messages.append({"role": "user", "content": turno["user"]})
        messages.append({"role": "assistant", "content": turno["assistant"]})

    messages.append({"role": "user", "content": user_text})

    logger.info(
        "Llamando a OpenAI (%s) para session_id=%s", OPENAI_MODEL, session_id
    )

    respuesta = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw = respuesta.choices[0].message.content or ""
    parsed = _extraer_json_desde_respuesta(raw)

    assistant_text = None
    nuevos_slots: Dict[str, str] = {}

    if isinstance(parsed, dict):
        assistant_text = parsed.get("assistant_text")
        posibles_slots = parsed.get("slots_detectados", {})
        if isinstance(posibles_slots, dict):
            nuevos_slots = posibles_slots

    if not isinstance(nuevos_slots, dict):
        nuevos_slots = {}

    # ------------------------------------------------------------------
    # Normalizar y combinar con lo ya existente (OBLIGATORIOS + OPCIONALES)
    # ------------------------------------------------------------------
    slots_normalizados: Dict[str, str] = {}
    for clave in TODOS_LOS_CAMPOS.keys():
        valor_actual = slots_actuales.get(clave)
        valor_nuevo = nuevos_slots.get(clave)

        if isinstance(valor_nuevo, str):
            valor_nuevo = valor_nuevo.strip()
        if isinstance(valor_actual, str):
            valor_actual = valor_actual.strip()

        if valor_nuevo:
            slots_normalizados[clave] = valor_nuevo
        elif valor_actual:
            slots_normalizados[clave] = valor_actual
        else:
            slots_normalizados[clave] = ""

    # Solo consideramos pendientes los OBLIGATORIOS
    campos_pendientes: List[str] = [
        k for k in CAMPOS_REQUERIDOS.keys() if not slots_normalizados.get(k)
    ]
    campos_completos: bool = len(campos_pendientes) == 0

    # ------------------------------------------------------------------
    # Fallback inteligente cuando no tenemos assistant_text
    # ------------------------------------------------------------------
    if not assistant_text:
        if campos_pendientes:
            siguiente = campos_pendientes[0]
            desc = CAMPOS_REQUERIDOS.get(siguiente, siguiente)
            assistant_text = (
                f"Para poder completar bien tu propuesta necesito un dato más: "
                f"{desc}. ¿Me lo podrías compartir, por favor?"
            )
        else:
            assistant_text = (
                "Perfecto, ya tengo la información principal para preparar tu propuesta "
                "y tu infografía personalizada."
            )

    sesion["slots"] = slots_normalizados
    history.append({"user": user_text, "assistant": assistant_text})
    sesion["history"] = history
    estado_global[session_id] = sesion

    logger.info(
        "Slots actuales para session_id=%s: %s", session_id, slots_normalizados
    )
    logger.info("Campos pendientes (obligatorios): %s", campos_pendientes)

    return assistant_text, slots_normalizados, campos_pendientes, campos_completos
