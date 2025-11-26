# -*- coding: utf-8 -*-
"""
core.tts_engine

TTS para Totem Evolución IA3 usando ElevenLabs (modo SEGURO, sin streaming directo).

- Usa ElevenLabs text_to_speech.convert para generar audio en formato PCM crudo
  y lo convertimos a WAV en disco.
- Funciones públicas:
    - synth_tts(texto: str, nombre_archivo: Optional[str] = None) -> str
    - play_audio(path: str, device_index: Optional[int] = None) -> None
    - speak(texto: str, device_index: Optional[int] = None) -> None

Reglas:
- Si ElevenLabs falla (voz no encontrada, formato, sin API key, etc.), solo se loguea; el servidor
  NUNCA se cae.
- Si no hay API key, simplemente no se reproduce nada.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

from core.logger import get_logger

# 🔹 IMPORTS NUEVOS PARA HABLAR CON NACHO (PUERTO 7000)
import urllib.parse
import urllib.request

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# Rutas y entorno
# ----------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

TTS_OUTPUT_DIR = ROOT_DIR / "data" / "tts_outputs"
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Configuración ElevenLabs
# ----------------------------------------------------------------------

ELEVEN_API_KEY = (
    os.getenv("ELEVEN_API_KEY")
    or os.getenv("ELEVENLABS_API_KEY")
    or ""
)

# Voz de Nacho (o la que definas en .env).
# Si ELEVEN_VOICE_ID está vacío, usamos una voz pública por defecto (George).
ELEVEN_VOICE_ID = (os.getenv("ELEVEN_VOICE_ID") or "").strip() or "JBFqnCBsd6RMkjVDRZzb"
ELEVEN_MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_multilingual_v2")

# ✅ Formato de salida: PCM crudo a 16 kHz por defecto (válido para todos los tiers).
# Luego lo convertimos nosotros a WAV.
ELEVEN_OUTPUT_FORMAT = os.getenv("ELEVEN_OUTPUT_FORMAT", "pcm_16000")

_client: Optional[ElevenLabs] = None
if ELEVEN_API_KEY:
    try:
        _client = ElevenLabs(api_key=ELEVEN_API_KEY)
        logger.info(
            "[TTS] Cliente ElevenLabs inicializado (model_id=%s, voice_id=%s, format=%s)",
            ELEVEN_MODEL_ID,
            ELEVEN_VOICE_ID,
            ELEVEN_OUTPUT_FORMAT,
        )
    except Exception as e:
        logger.exception("[TTS] No se pudo inicializar ElevenLabs: %s", e)
        _client = None
else:
    logger.warning(
        "[TTS] ELEVEN_API_KEY no configurada; la síntesis de voz quedará deshabilitada."
    )

# ----------------------------------------------------------------------
# Configuración Nacho (puerto 7000)
# ----------------------------------------------------------------------

# Puedes sobreescribir esto en .env con NACHO_BASE_URL, por ejemplo:
# NACHO_BASE_URL=http://127.0.0.1:7000
NACHO_BASE_URL = os.getenv("NACHO_BASE_URL", "http://localhost:7000").rstrip("/")


# ----------------------------------------------------------------------
# Utilidades internas
# ----------------------------------------------------------------------

def _build_filename(nombre_archivo: Optional[str] = None) -> Path:
    """
    Construye un nombre de archivo WAV en data/tts_outputs.
    """
    if not nombre_archivo:
        nombre_archivo = f"tts_{int(time.time())}_{uuid.uuid4().hex}.wav"
    if not nombre_archivo.lower().endswith(".wav"):
        nombre_archivo += ".wav"
    return TTS_OUTPUT_DIR / nombre_archivo


def _parse_pcm_samplerate(fmt: str) -> int:
    """
    Extrae el sample rate de cadenas como 'pcm_16000', 'pcm_44100', etc.
    Si algo falla, devuelve 44100.
    """
    try:
        parts = fmt.split("_")
        if len(parts) >= 2 and parts[0] == "pcm":
            return int(parts[1])
    except Exception:
        pass
    return 44100


# 🔹 NUEVO: mandar el texto a Nacho (puerto 7000)
def _send_text_to_nacho(texto: str) -> None:
    """
    Envía el texto al visor de Nacho vía HTTP GET:
        http://localhost:7000/Texto%20codificado

    - No lanza excepciones hacia afuera.
    - Si falla, solo loguea un warning.
    """
    texto = (texto or "").strip()
    if not texto:
        return

    try:
        encoded = urllib.parse.quote(texto, safe="")
        url = f"{NACHO_BASE_URL}/{encoded}"
        logger.info("[Nacho] Enviando texto al visor: %s", url)
        # No nos interesa la respuesta, solo disparar la animación
        with urllib.request.urlopen(url, timeout=1) as _:
            pass
    except Exception as e:
        logger.warning("[Nacho] No se pudo enviar texto al visor: %s", e)


# ----------------------------------------------------------------------
# API pública: synth_tts / play_audio / speak
# ----------------------------------------------------------------------

def synth_tts(texto: str, nombre_archivo: Optional[str] = None) -> str:
    """
    Genera un archivo WAV con ElevenLabs (usando PCM crudo) y devuelve la ruta absoluta como str.
    Si algo falla, devuelve "" (cadena vacía) y NUNCA lanza excepción.
    """
    texto = (texto or "").strip()
    if not texto:
        logger.debug("[TTS] synth_tts llamado con texto vacío.")
        return ""

    if _client is None:
        logger.warning("[TTS] synth_tts llamado pero ElevenLabs no está configurado.")
        return ""

    out_path = _build_filename(nombre_archivo)

    try:
        logger.info(
            "[TTS] Generando audio con ElevenLabs convert (voice_id=%s, model_id=%s, format=%s)",
            ELEVEN_VOICE_ID,
            ELEVEN_MODEL_ID,
            ELEVEN_OUTPUT_FORMAT,
        )

        audio_bytes = _client.text_to_speech.convert(
            voice_id=ELEVEN_VOICE_ID,
            model_id=ELEVEN_MODEL_ID,
            text=texto,
            output_format=ELEVEN_OUTPUT_FORMAT,
        )

        # Unificamos en un solo buffer de bytes
        if isinstance(audio_bytes, (bytes, bytearray)):
            raw = bytes(audio_bytes)
        else:
            chunks = []
            for chunk in audio_bytes:
                chunks.append(chunk)
            raw = b"".join(chunks)

        if not raw:
            logger.error("[TTS] ElevenLabs devolvió audio vacío.")
            return ""

        # Si estamos en PCM crudo, lo convertimos a WAV con soundfile
        if ELEVEN_OUTPUT_FORMAT.startswith("pcm_"):
            sr = _parse_pcm_samplerate(ELEVEN_OUTPUT_FORMAT)
            # PCM 16-bit little-endian -> int16
            pcm16 = np.frombuffer(raw, dtype=np.int16)
            if pcm16.size == 0:
                logger.error("[TTS] Buffer PCM vacío después de convertir a int16.")
                return ""
            # Normalizamos a float32 para soundfile
            audio = pcm16.astype("float32") / 32768.0
            sf.write(str(out_path), audio, sr)
            logger.info(
                "[TTS] WAV generado a partir de PCM: %s (sr=%d, frames=%d)",
                out_path,
                sr,
                len(pcm16),
            )
        else:
            # Otros formatos (mp3/opus/etc.) → guardamos tal cual.
            with open(out_path, "wb") as f:
                f.write(raw)
            logger.info("[TTS] Archivo de audio generado: %s", out_path)

        return str(out_path)

    except ApiError as e:
        # Aquí caen errores tipo invalid_output_format, voice_not_found, límite de uso, etc.
        logger.error("[TTS] ApiError de ElevenLabs en synth_tts: %s", e)
        # NO volvemos a lanzar la excepción
        return ""

    except Exception as e:
        logger.exception("[TTS] Error inesperado en synth_tts: %s", e)
        # NO volvemos a lanzar la excepción
        return ""


def play_audio(path: str, device_index: Optional[int] = None) -> None:
    """
    Reproduce un archivo de audio (WAV recomendado).
    Si algo falla, solo se loguea el error y continúa.
    """
    if not path:
        logger.warning("[TTS] play_audio llamado con path vacío; nada que reproducir.")
        return

    p = Path(path)
    if not p.exists():
        logger.warning("[TTS] Archivo de audio no existe: %s", p)
        return

    try:
        data, sr = sf.read(str(p), dtype="float32")
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]

        logger.info(
            "[TTS] Reproduciendo audio %s (sr=%d, channels=%d, device=%s)",
            p.name,
            sr,
            channels,
            str(device_index),
        )

        sd.play(data, sr, device=device_index)
        sd.wait()

    except Exception as e:
        logger.exception("[TTS] Error al reproducir audio con sounddevice: %s", e)
        # NO lanzamos la excepción; solo la registramos.


def speak(texto: str, device_index: Optional[int] = None) -> None:
    """
    Azúcar sintáctico: genera el audio y lo reproduce.
    Internamente llama a synth_tts() y play_audio().
    Además, ENVÍA el texto a Nacho (puerto 7000) para que mueva la boca.
    Nunca lanza excepción hacia afuera.
    """
    try:
        # 1) Generamos el audio (si ElevenLabs está habilitado)
        path = synth_tts(texto)

        # 2) Siempre que haya texto, lo mandamos al visor Nacho
        #    para que se anime con visemas por texto.
        _send_text_to_nacho(texto)

        # 3) Si el TTS generó audio, lo reproducimos
        if path:
            play_audio(path, device_index=device_index)
        else:
            logger.warning("[TTS] speak() no pudo generar audio; revisa logs anteriores.")
    except Exception as e:
        logger.exception("[TTS] Error inesperado en speak(): %s", e)
        # Tampoco lanzamos nada hacia afuera.
