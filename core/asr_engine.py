# -*- coding: utf-8 -*-
"""
core.asr_engine
Captura audio del micrófono y lo transcribe con OpenAI Whisper.
"""

import os
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
import winsound

from core.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).parent.parent.resolve()
AUDIO_INPUT_DIR = ROOT_DIR / "data" / "asr_inputs"
AUDIO_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar .env para asegurar que OPENAI_API_KEY está en el entorno
env_path = ROOT_DIR / ".env"
if env_path.is_file():
    load_dotenv(env_path)

# Cliente OpenAI (usa OPENAI_API_KEY del entorno)
client = OpenAI()

SAMPLE_RATE = 16000


def grabar_una_frase(segundos: int = 6) -> Optional[Path]:
    """
    Graba audio del micrófono durante 'segundos' y guarda un WAV temporal.
    Devuelve la ruta al archivo o None si falla.
    """
    try:
        # Bip para avisar que ya está grabando
        winsound.Beep(1000, 300)
    except Exception:
        # Si falla el beep no pasa nada
        pass

    logger.info("asr_engine: grabando audio (%s s)...", segundos)

    try:
        frames = int(segundos * SAMPLE_RATE)
        audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()

        file_path = AUDIO_INPUT_DIR / "asr_input.wav"
        sf.write(str(file_path), audio, SAMPLE_RATE)
        logger.info("asr_engine: audio guardado en %s", file_path)
        return file_path
    except Exception:
        logger.exception("asr_engine: error grabando audio del micrófono.")
        return None


def transcribir_audio(file_path: Path) -> Optional[str]:
    """
    Envía el WAV a OpenAI Whisper y devuelve el texto transcrito.
    """
    try:
        with file_path.open("rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="es",
            )
        text = (result.text or "").strip()
        if not text:
            logger.warning("asr_engine: transcripción vacía.")
            return None
        logger.info("asr_engine: texto transcrito: %s", text)
        return text
    except Exception:
        logger.exception("asr_engine: error al transcribir audio con Whisper.")
        return None


def escuchar_y_transcribir(segundos: int = 6) -> Optional[str]:
    """
    Flujo completo: graba una frase y la transcribe.
    Devuelve el texto en español o None si falla.
    """
    wav_path = grabar_una_frase(segundos=segundos)
    if not wav_path:
        return None
    return transcribir_audio(wav_path)
