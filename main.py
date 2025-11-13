# -*- coding: utf-8 -*-
import os, uuid, json, asyncio, httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from core.logger import get_logger
from core import dialog_engine, proposal_trigger, infographic_engine, mailer, tts_engine
from core.blender_bridge import BlenderBridge

LOG = get_logger(__name__)
app = FastAPI(title=os.getenv("APP_NAME", "TOTEM"))

# ========= Helper UTF-8 =========
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

class SpeakRequest(BaseModel):
    text: str
    rate_wpm: Optional[int] = 160

# ========= Blender Bridge =========
BLENDER = BlenderBridge(
    mode=os.getenv("BLENDER_BRIDGE_MODE", "file"),
    file_path=os.getenv("BLENDER_FILE_PATH", "data/blender/jaw.json"),
    udp_host=os.getenv("BLENDER_UDP_HOST", "127.0.0.1"),
    udp_port=int(os.getenv("BLENDER_UDP_PORT", "9001")),
)

# ========= Endpoints =========
@app.get("/health")
async def health():
    return json_utf8({"status": "ok"})

@app.post("/session/start", response_model=SessionStartResponse)
async def start_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "slots": {},
        "artifacts": {},
        "_mode": "free",
        "_confused": 0,
        "_last_missing": None,
    }
    LOG.info({"evt": "session_start", "sid": sid})
    return json_utf8({"session_id": sid})

@app.post("/chat/turn", response_model=ChatTurnResponse)
async def chat_turn(req: ChatTurnRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")

    try:
        out = await asyncio.to_thread(dialog_engine.process_turn, sess, req.text)
        sess["slots"].update(out.get("slots", {}))
        for k in ("_mode", "_confused", "_last_missing"):
            if k in out:
                sess[k] = out[k]
        payload = {
            "response": out.get("response", ""),
            "slots": sess["slots"],
            "missing": out.get("missing", []),
        }
        LOG.info({"evt": "chat_turn", "sid": req.session_id, "payload": payload})
        return json_utf8(payload)
    except Exception as e:
        LOG.error({"evt": "chat_turn_error", "error": str(e)})
        raise HTTPException(500, f"Error interno: {e}")

@app.post("/proposal/trigger", response_model=ProposalTriggerResponse)
async def trigger(req: ProposalTriggerRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")

    try:
        # 1️⃣ Generar propuesta PDF (Catalyst)
        url = await asyncio.to_thread(proposal_trigger.trigger, sess["slots"])
        if not url:
            return json_utf8({"message": "No se pudo generar la propuesta", "proposal_pdf_url": None})
        sess["artifacts"]["proposal_pdf_url"] = url

        # 2️⃣ Crear Lead directo en Zoho CRM
        zoho_url = os.getenv("ZOHO_CRM_URL")
        token = os.getenv("ZOHO_AUTHTOKEN")
        if not token:
            LOG.warning("⚠️ ZOHO_AUTHTOKEN no definido, no se creará el Lead.")
        else:
            data = {
                "data": [{
                    "Company": sess["slots"].get("Empresa", "Sin empresa"),
                    "Last_Name": sess["slots"].get("Name", "Visitante"),
                    "Email": sess["slots"].get("Email"),
                    "Lead_Source": "Totem IA3",
                    "Description": "Propuesta generada automáticamente por Totem IA3",
                    "Proposal_Link": url
                }]
            }
            async with httpx.AsyncClient(timeout=15) as client:
                await client.post(
                    zoho_url,
                    headers={"Authorization": f"Zoho-authtoken {token}", "Content-Type": "application/json"},
                    json=data,
                )
            LOG.info({"evt": "lead_created", "sid": req.session_id, "url": url})

        return json_utf8({"message": "Propuesta generada y Lead creado en CRM", "proposal_pdf_url": url})

    except Exception as e:
        LOG.error({"evt": "proposal_trigger_error", "error": str(e)})
        raise HTTPException(500, f"Error al generar la propuesta: {e}")

@app.post("/infographic/generate", response_model=InfographicGenerateResponse)
async def infgen(req: InfographicGenerateRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    try:
        png, pdf = await asyncio.to_thread(infographic_engine.generate, sess["slots"])
        sess["artifacts"]["infographic_png"] = png
        sess["artifacts"]["infographic_pdf"] = pdf
        LOG.info({"evt": "infographic_generated", "sid": req.session_id})
        return json_utf8({"png_path": png, "pdf_path": pdf})
    except Exception as e:
        LOG.error({"evt": "infographic_error", "error": str(e)})
        raise HTTPException(500, f"Error al generar la infografía: {e}")

@app.post("/tts/speak")
async def tts_speak(req: SpeakRequest):
    try:
        asyncio.create_task(asyncio.to_thread(
            tts_engine.speak, req.text, req.rate_wpm, lambda v: BLENDER.set_jaw(v)
        ))
        LOG.info({"evt": "tts_speak", "text": req.text})
        return json_utf8({"status": "ok"})
    except Exception as e:
        LOG.error({"evt": "tts_error", "error": str(e)})
        raise HTTPException(500, f"Error en TTS: {e}")
