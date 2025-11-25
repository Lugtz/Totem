# -*- coding: utf-8 -*-
"""
core.tts_engine
Síntesis de voz para Totem Evolución IA3.

Adaptado para ejecutarse correctamente en:
- Linux (usa ffplay)
- Windows (usa playsound como fallback)

Detección automática del sistema operativo.
"""

import os
import time
import uuid
import platform
from pathlib import Path
from typing import Optional, Union

import requests
from dotenv import load_dotenv

from core.logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# 1. DETECTAR SISTEMA OPERATIVO
# ----------------------------------------------------------------------
SO = platform.system()  # "Linux", "Windows", "Darwin"

logger.info(f"tts_engine: Sistema detectado → {SO}")

# ----------------------------------------------------------------------
# 2. RUTAS Y ENTORNO
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent.resolve()
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

TTS_OUTPUT_DIR = ROOT_DIR / "data" / "tts_outputs"
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 3. CONFIGURACIÓN DE ELEVENLABS
# ----------------------------------------------------------------------
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "sVNMtqOxjmuk9xGHdR88")  # Adam

def _check_key():
    if not ELEVEN_API_KEY:
        raise ValueError("❌ ELEVEN_API_KEY no está definido en .env")

# ----------------------------------------------------------------------
# 4. SÍNTESIS DE VOZ
# ----------------------------------------------------------------------
def synth_tts(texto: str, nombre_archivo: str = "dialogo", **kwargs) -> Optional[Path]:
    """
    Genera un MP3 con ElevenLabs.
    """
    _check_key()

    texto = (texto or "").strip()
    if not texto:
        logger.warning("core.tts_engine: synth_tts llamado con texto vacío.")
        return None

    base = Path(nombre_archivo).stem or "dialogo"
    output_path = TTS_OUTPUT_DIR / f"{base}_{uuid.uuid4()}.mp3"

    logger.info(f"tts_engine: generando audio en {output_path}")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": texto,
        "voice_settings": {
            "stability": 0.50,
            "similarity_boost": 0.80
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers)

        if response.status_code != 200:
            logger.error("elevenlabs error: %s", response.text)
            return None

        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"tts_engine: audio generado → {output_path}")
        return output_path

    except Exception:
        logger.exception("tts_engine: error al sintetizar TTS.")
        return None

# ----------------------------------------------------------------------
# 5. REPRODUCCIÓN MULTIPLATAFORMA
# ----------------------------------------------------------------------
def _play_linux(path: Path):
    """
    Reproduce audio usando ffplay en Linux.
    """
    try:
        os.system(f"ffplay -nodisp -autoexit '{path}' 2>/dev/null")
    except Exception:
        logger.exception("tts_engine: error usando ffplay en Linux.")


def _play_windows(path: Path):
    """
    Reproduce audio en Windows usando playsound.
    """
    try:
        from playsound import playsound
        playsound(str(path))
    except Exception:
        logger.exception("tts_engine: error usando playsound en Windows.")


def play_audio(ruta: Optional[Union[str, Path]]) -> None:
    """
    Selecciona el método de reproducción dependiendo de Linux o Windows.
    """
    if ruta is None:
        logger.warning("tts_engine: play_audio llamado con ruta=None.")
        return

    path = Path(ruta)
    if not path.is_file():
        logger.warning(f"tts_engine: archivo no encontrado → {path}")
        return

    try:
        if SO == "Linux":
            _play_linux(path)

        elif SO == "Windows":
            _play_windows(path)

        else:
            logger.warning("tts_engine: SO desconocido, intentando abrir archivo...")
            os.system(f"'{path}'")

    except Exception:
        logger.exception("tts_engine: error general al reproducir audio.")

# ----------------------------------------------------------------------
# 6. HABLAR DIRECTAMENTE
# ----------------------------------------------------------------------
def speak(texto: str):
    """
    Genera TTS y lo reproduce automáticamente.
    """
    ruta = synth_tts(texto)
    if ruta:
        play_audio(ruta)
