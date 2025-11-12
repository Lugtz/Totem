# -*- coding: utf-8 -*-
# ============================================================
# TOTEM EVOLUCIÓN IA3 – ORQUESTADOR PRINCIPAL (PRIORIDAD 1)
# ============================================================

import asyncio
import threading
from fastapi import FastAPI, Request
from typing import Dict

# ------------------------------------------------------------
# IMPORTS REALES (según tu estructura actual)
# ------------------------------------------------------------
from core.dialog_engine import procesar_turno_dialogo, CAMPOS_REQUERIDOS
from core.tts_engine import synth_tts, play_audio
from core.animation_controller import mover_avatar_from_audio
from core.proposal_trigger import generar_propuesta_pdf
from core.infographic_engine import generar_infografia_png
from core.health import health_check
from core.logger import log_info
from core.utils import estado_global, reiniciar_estado      # <-- FIX
from core.camera_agent import iniciar_detector

# ------------------------------------------------------------
# CONFIGURACIÓN FASTAPI
# ------------------------------------------------------------
app = FastAPI(title="Totem Evolución IA3", version="1.0.0")

# ------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------

def _hablar_y_animar(texto: str):
    """
    Genera audio con TTS y ejecuta la animación sincronizada en paralelo.
    """
    try:
        wav_path = synth_tts(texto)
        threading.Thread(target=play_audio, args=(wav_path,), daemon=True).start()
        threading.Thread(target=mover_avatar_from_audio, args=(wav_path,), daemon=True).start()
    except Exception as e:
        print(f"⚠ Error en _hablar_y_animar: {e}")

async def _finalizar_y_reiniciar():
    """
    Espera unos segundos y reinicia el estado global.
    """
    await asyncio.sleep(10)
    reiniciar_estado()
    print("🔄 Sistema reiniciado. Listo para el siguiente visitante.")

def on_person_detected():
    """
    Callback que se ejecuta cuando la cámara detecta una persona.
    """
    if estado_global.get("en_sesion"):
        return

    estado_global["en_sesion"] = True
    msg = "Hola, bienvenido a Evolución i3. ¿Cuál es tu nombre?"
    log_info(msg)
    _hablar_y_animar(msg)

# ------------------------------------------------------------
# EVENTO DE INICIO
# ------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    print("🚀 Totem Evolución IA3 iniciado correctamente.")
    print("🎥 Activando detector de personas...")
    threading.Thread(target=iniciar_detector, args=(on_person_detected,), daemon=True).start()

# ------------------------------------------------------------
# ENDPOINT PRINCIPAL DE CHAT
# ------------------------------------------------------------

@app.post("/chat/turn")
async def chat_turn(req: Request):
    """
    Endpoint principal de conversación con el visitante.
    """
    try:
        data = await req.json()
        texto = data.get("text", "").strip()
        if not texto:
            return {"ok": False, "error": "Texto vacío"}

        print(f"🧠 Visitante: {texto}")

        respuesta, campos_llenos = procesar_turno_dialogo(texto, estado_global)
        _hablar_y_animar(respuesta)

        if campos_llenos:
            print("🧾 Campos completos → generando propuesta e infografía...")
            campos: Dict = estado_global["campos"].copy()

            asyncio.create_task(generar_propuesta_pdf(campos))
            asyncio.create_task(generar_infografia_png(campos))
            _hablar_y_animar("Perfecto. Estoy generando tu propuesta e infografía. Dame un momento, por favor.")
            asyncio.create_task(_finalizar_y_reiniciar())

        faltantes = [c for c in CAMPOS_REQUERIDOS if c not in estado_global["campos"]]

        return {
            "ok": True,
            "respuesta": respuesta,
            "capturado": estado_global["campos"],
            "faltan": faltantes,
        }

    except Exception as e:
        print(f"❌ Error en chat_turn: {e}")
        return {"ok": False, "error": str(e)}

# ------------------------------------------------------------
# ENDPOINT DE SALUD
# ------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "ok": True,
        "en_sesion": estado_global.get("en_sesion", False),
        "status": health_check(),
    }

# ------------------------------------------------------------
# EJECUCIÓN LOCAL
# ------------------------------------------------------------

if __name__ == "__main__":    # <-- FIX
    import uvicorn
    uvicorn.run("amain:app", host="127.0.0.1", port=8000, reload=True)
