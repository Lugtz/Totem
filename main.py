# main.py
import os, uuid, json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List, Optional

# 🔴 Cargar .env ANTES de importar nada que lo use
from dotenv import load_dotenv
load_dotenv()

from core.logger import get_logger
from core import dialog_engine, proposal_trigger, infographic_engine, mailer, tts_engine
from core.blender_bridge import BlenderBridge

LOG = get_logger(__name__)
app = FastAPI(title=os.getenv("APP_NAME", "TOTEM"))

# ========= Helper UTF-8 (sin mojibake) =========
def json_utf8(content, status_code: int = 200) -> Response:
    return Response(
        content=json.dumps(content, ensure_ascii=False),
        status_code=status_code,
        media_type="application/json; charset=utf-8",
    )

@app.middleware("http")
async def force_utf8(request: Request, call_next):
    resp = await call_next(request)
    ct = resp.headers.get("content-type", "")
    if "application/json" in ct and "charset=" not in ct.lower():
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp

# ========= Estado =========
SESSIONS: Dict[str, Dict] = {}

# ========= Modelos =========
class SessionStartResponse(BaseModel):
    session_id: str

class ChatTurnRequest(BaseModel):
    session_id: str
    text: str

class ChatTurnResponse(BaseModel):
    response: str
    slots: Dict[str, str] = {}
    missing: List[str] = []

class ProposalTriggerRequest(BaseModel):
    session_id: str

class ProposalTriggerResponse(BaseModel):
    proposal_pdf_url: Optional[str] = None
    message: str

class InfographicGenerateRequest(BaseModel):
    session_id: str

class InfographicGenerateResponse(BaseModel):
    png_path: str
    pdf_path: Optional[str] = None

class InfographicEmailRequest(BaseModel):
    session_id: str
    email: EmailStr

class BlenderJawRequest(BaseModel):
    val: float = Field(ge=0.0, le=1.0)

class BlenderHeadRequest(BaseModel):
    yaw: float = Field(ge=0.0, le=1.0)
    pitch: float = Field(ge=0.0, le=1.0)

# ========= Blender Bridge =========
BLENDER = BlenderBridge(
    mode=os.getenv("BLENDER_BRIDGE_MODE", "file"),
    file_path=os.getenv("BLENDER_FILE_PATH", "data/blender/jaw.json"),
    udp_host=os.getenv("BLENDER_UDP_HOST", "127.0.0.1"),
    udp_port=int(os.getenv("BLENDER_UDP_PORT", "9001")),
)

# ========= Endpoints =========
@app.get("/health")
def health():
    return json_utf8({"status": "ok"})

@app.post("/session/start", response_model=SessionStartResponse)
def start_session():
    sid = str(uuid.uuid4())
    # state de diálogo
    SESSIONS[sid] = {
        "slots": {},
        "artifacts": {},
        "_mode": "free",
        "_confused": 0,
        "_last_missing": None,
    }
    LOG.info({"evt":"session_start","sid":sid})
    return json_utf8({"session_id": sid})

@app.post("/chat/turn", response_model=ChatTurnResponse)
def chat_turn(req: ChatTurnRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    out = dialog_engine.process_turn(sess, req.text)
    # Guardar slots y estado devuelto por el motor
    sess["slots"].update(out.get("slots", {}))
    for k in ("_mode","_confused","_last_missing"):
        if k in out:
            sess[k] = out[k]
    payload = {
        "response": out.get("response", ""),
        "slots": sess["slots"],
        "missing": out.get("missing", []),
    }
    return json_utf8(payload)

@app.post("/proposal/trigger", response_model=ProposalTriggerResponse)
def trigger(req: ProposalTriggerRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    url = proposal_trigger.trigger(sess["slots"])
    if not url:
        return json_utf8({"message":"No se pudo generar la propuesta","proposal_pdf_url":None})
    sess["artifacts"]["proposal_pdf_url"] = url
    return json_utf8({"message":"Propuesta generada (no enviada por email)","proposal_pdf_url":url})

@app.post("/infographic/generate", response_model=InfographicGenerateResponse)
def infgen(req: InfographicGenerateRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    png, pdf = infographic_engine.generate(sess["slots"])
    sess["artifacts"]["infographic_png"] = png
    sess["artifacts"]["infographic_pdf"] = pdf
    return json_utf8({"png_path": png, "pdf_path": pdf})

@app.post("/infographic/email")
def infemail(req: InfographicEmailRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    art = sess["artifacts"]
    if not art.get("infographic_png"):
        raise HTTPException(422, "Primero genera la infografía")
    ok = mailer.send_infographic(req.email, art["infographic_png"], art.get("infographic_pdf"))
    if not ok:
        raise HTTPException(500, "Error al enviar email")
    return json_utf8({"status":"ok","message":f"Infografía enviada a {req.email}"})

# ====== TTS (se mantiene) ======
class SpeakRequest(BaseModel):
    text: str
    rate_wpm: Optional[int] = 160

@app.post("/tts/speak")
def tts_speak(req: SpeakRequest):
    utt_id = tts_engine.speak(req.text, rate_wpm=req.rate_wpm, on_viseme=lambda v: BLENDER.set_jaw(v))
    return json_utf8({"status":"ok","utterance_id": utt_id})
