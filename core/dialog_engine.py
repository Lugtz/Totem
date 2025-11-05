# ==========================================================
# core/dialog_engine.py
# Motor conversacional dual (libre/propuesta) para Evolución i3
# Compatible con live_test.py y FastAPI
# ==========================================================

import os
import re
import random
from pathlib import Path
from typing import Dict, Tuple, Optional
import requests

from dotenv import load_dotenv
from openai import OpenAI
from core.logger import get_logger

# ----------------------------------------------------------
# Carga del entorno (.env)
# ----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

LOG = get_logger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ Falta la variable de entorno OPENAI_API_KEY en .env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------
# Configuración de slots
# ----------------------------------------------------------
REQUIRED_SLOTS = [
    "Name",
    "Nombre de la empresa",
    "Email",
    "Solucion a implementar",
    "Duracion",
    "Precio",
    "Moneda",
]

OPTIONAL_SLOTS = [
    "Semanas piloto",
    "IVA",
    "Porcentaje de inicio",
    "Porcentaje al cierre",
    "Licencia a utilizar",
    "Versión",
    "Precio mensual por Usuario",
    "Precio Anual por usuario",
    "Moneda de las Licencias",
]

PROPOSAL_KEYWORDS = [
    "propuesta", "cotización", "cotizacion", "precio", "presupuesto",
    "demo", "plan", "servicio", "me interesa", "cuánto cuesta",
    "cuanto cuesta", "cuánto vale", "cuanto vale", "cotiza", "cotizar",
]

# ----------------------------------------------------------
# Regex de extracción
# ----------------------------------------------------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PRICE_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})+|\d+)(?:\s*(?:mxn|usd|\$))?", re.IGNORECASE)
DUR_RE = re.compile(r"(?:(\d{1,3})\s*(semanas?|mes(?:es)?))|(\b(?:1|una)\s*semana\b)|(\b(?:1|un)\s*mes\b)", re.IGNORECASE)

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None

def _extract_price(text: str) -> Optional[str]:
    m = PRICE_RE.search(text.replace("\u00a0", " "))
    if not m:
        return None
    value = m.group(1)
    return value.replace(".", "").replace(",", "")

def _extract_currency(text: str) -> Optional[str]:
    t = text.lower()
    if "usd" in t or "dólar" in t or "dolar" in t or "us$" in t:
        return "USD"
    if "mxn" in t or "peso" in t or "$" in t:
        return "MXN"
    return None

def _extract_duration(text: str) -> Optional[str]:
    t = text.lower()
    m = DUR_RE.search(t)
    if m:
        span = m.span()
        return _norm_space(text[span[0]:span[1]])
    return None

def _extract_company(text: str) -> Optional[str]:
    t = text.strip()
    patterns = [
        r"\bempresa\s*[:=]\s*([A-Za-z0-9\.\-&\s]+)",
        r"\bmi empresa es\s+([A-Za-z0-9\.\-&\s]+)",
        r"\btrabajo en\s+([A-Za-z0-9\.\-&\s]+)",
        r"\bde la empresa\s+([A-Za-z0-9\.\-&\s]+)",
    ]
    for p in patterns:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            return _norm_space(m.group(1))
    return None

def _extract_name(text: str) -> Optional[str]:
    m = re.search(r"\b(?:me llamo|mi nombre es|soy)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s\.-]+)", text, re.IGNORECASE)
    if m:
        name_full = re.split(r"\bde\b", m.group(1), flags=re.IGNORECASE)[0]
        return _norm_space(name_full)
    return None

def _extract_solution(text: str) -> Optional[str]:
    if any(k in text.lower() for k in ["implementar", "automatizar", "crm", "forms", "desk", "voz", "asistente"]):
        return _norm_space(text)
    return None

# ----------------------------------------------------------
# Conversación libre
# ----------------------------------------------------------
SYSTEM_PROMPT = (
    "Eres Nacho, el asistente oficial de Evolución i3. "
    "Respondes en español, con tono cálido, claro y profesional. "
    "Conversas de forma natural. "
    "No pidas datos personales salvo que el usuario mencione que quiere una propuesta."
)

def free_talk(user_text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_completion_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        LOG.error(f"⚠️ OpenAI Error: {e}")
        return random.choice([
            "Soy Nacho, de Evolución i3. ¿En qué te puedo apoyar?",
            "Con gusto te ayudo. ¿Qué te gustaría lograr con automatización o CRM?",
            "Cuéntame tu idea y vemos juntos la mejor solución."
        ])

# ----------------------------------------------------------
# Intención y slots
# ----------------------------------------------------------
def detect_intent(user_text: str) -> str:
    t = user_text.lower()
    for kw in PROPOSAL_KEYWORDS:
        if kw in t:
            return "proposal"
    return "free"

def detect_slot_from_text(session_slots: Dict[str, str], text: str) -> Tuple[Optional[str], Optional[str]]:
    for extractor, key in [
        (_extract_email, "Email"),
        (_extract_price, "Precio"),
        (_extract_currency, "Moneda"),
        (_extract_duration, "Duracion"),
        (_extract_name, "Name"),
        (_extract_company, "Nombre de la empresa"),
        (_extract_solution, "Solucion a implementar"),
    ]:
        v = extractor(text)
        if v and not session_slots.get(key):
            return key, v
    return None, None

def _next_question_for(missing_slot: str, slots: Dict[str, str]) -> str:
    name = slots.get("Name", "")
    prompts = {
        "Name": "¿Podrías decirme tu nombre?",
        "Nombre de la empresa": f"Gracias {name}. ¿Con qué empresa estás hoy?",
        "Email": "¿Cuál es tu correo de contacto?",
        "Solucion a implementar": "¿Qué solución te gustaría implementar o automatizar (CRM, Forms, Desk, voz, etc.)?",
        "Duracion": "¿Cuánto tiempo estimas de duración (por ejemplo, 6 semanas o 2 meses)?",
        "Precio": "¿Con qué presupuesto estimado contamos? (por ejemplo, 120,000)",
        "Moneda": "¿La moneda será MXN o USD?",
    }
    return prompts.get(missing_slot, "¿Podrías especificar un poco más?")

def _missing_list(slots: Dict[str, str]):
    return [s for s in REQUIRED_SLOTS if not slots.get(s)]

def _sanitize_slots(s: Dict[str, str]) -> Dict[str, str]:
    return {k: str(v) for k, v in s.items() if v}

# ----------------------------------------------------------
# Procesador principal
# ----------------------------------------------------------
def process_turn(session: Dict, user_text: str) -> Dict:
    session.setdefault("slots", {})
    session.setdefault("_mode", "free")
    session.setdefault("_miss_count", 0)
    slots = session["slots"]
    mode = session["_mode"]

    if mode == "free":
        intent = detect_intent(user_text)
        if intent == "proposal":
            session["_mode"] = "proposal"
            return {"response": "Perfecto. Puedo prepararte una propuesta personalizada. ¿Podrías decirme tu nombre?",
                    "slots": slots, "missing": _missing_list(slots)}
        reply = free_talk(user_text)
        return {"response": reply, "slots": slots, "missing": _missing_list(slots)}

    if mode == "proposal":
        det_slot, det_val = detect_slot_from_text(slots, user_text)
        if det_slot and det_val:
            slots[det_slot] = det_val
            session["_miss_count"] = 0
        else:
            session["_miss_count"] += 1
            if session["_miss_count"] == 1:
                return {"response": "Disculpa, no alcancé a capturar ese dato. ¿Podrías repetirlo?",
                        "slots": slots, "missing": _missing_list(slots)}
            if session["_miss_count"] >= 2:
                session["_mode"] = "free"
                session["_miss_count"] = 0
                return {"response": "Parece que no logramos avanzar. Si no te interesa ahora, no pasa nada.",
                        "slots": slots, "missing": _missing_list(slots)}

        missing = _missing_list(slots)
        if missing:
            question = _next_question_for(missing[0], slots)
            return {"response": question, "slots": slots, "missing": missing}
        else:
            session["_mode"] = "free"
            return {"response": "Perfecto, ya tengo todo. ¿Quieres que genere tu propuesta ahora?",
                    "slots": slots, "missing": []}

    return {"response": free_talk(user_text), "slots": slots, "missing": _missing_list(slots)}

# ----------------------------------------------------------
# 🔹 Compatibilidad con live_test.py
# ----------------------------------------------------------
def start_session(session_id: str):
    """Inicia sesión remota (para pruebas locales con live_test.py)."""
    try:
        res = requests.post(f"{BASE_URL}/session/start", json={"session_id": session_id}, timeout=5)
        res.raise_for_status()
        LOG.info(f"✅ Sesión iniciada correctamente: {session_id}")
    except Exception as e:
        LOG.error(f"❌ Error al iniciar sesión en {BASE_URL}: {e}")
