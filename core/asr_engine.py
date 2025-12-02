# -*- coding: utf-8 -*-
"""
core.asr_engine
Captura audio del micrófono y lo transcribe con OpenAI Whisper.

Esta versión:
- NO usa una duración fija.
- Graba hasta que detecta ~2.5 segundos seguidos de silencio.
- Tiene un límite máximo de seguridad (por defecto 300s) por si algo sale mal.
- Es MÁS SENSIBLE a la voz (umbral de energía más bajo).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
import winsound

from core.logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# Rutas y entorno
# ----------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.parent.resolve()
AUDIO_INPUT_DIR = ROOT_DIR / "data" / "asr_inputs"
AUDIO_INPUT_DIR.mkdir(parents=True, exist_ok=True)

env_path = ROOT_DIR / ".env"
if env_path.is_file():
    load_dotenv(env_path)

# Cliente OpenAI (usa OPENAI_API_KEY del entorno)
client = OpenAI()

# ----------------------------------------------------------------------
# Parámetros de audio / VAD
# ----------------------------------------------------------------------

SAMPLE_RATE = 16000          # 16 kHz, recomendado para Whisper
CHANNELS = 1
DTYPE = "float32"

# ✅ VAD: si hay más de SILENCE_SECONDS sin voz, cortamos
#    Lo subimos un poco para que no corte tan agresivo
SILENCE_SECONDS = 2.5  # antes 2.0

# ✅ Umbral de energía para considerar que "hay voz"
#    MÁS SENSIBLE: valor más bajo → detecta voz aunque hables suave
#    Si aún corta raro, puedes bajar a 0.004 o 0.003.
ENERGY_THRESHOLD = 0.006  # antes 0.010

# Límite máximo de seguridad (en segundos) para que no sea infinito si algo falla
MAX_RECORD_SECONDS = 300.0   # 5 minutos


def _rms_energy(block: np.ndarray) -> float:
    """
    Calcula la energía RMS de un bloque de audio.
    """
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block ** 2)))


def grabar_una_frase_vad(
    silence_seconds: float = SILENCE_SECONDS,
    energy_threshold: float = ENERGY_THRESHOLD,
    max_seconds: float = MAX_RECORD_SECONDS,
) -> Optional[Path]:
    """
    Graba audio del micrófono usando VAD:
    - Arranca y sigue leyendo bloques.
    - Si detecta más de `silence_seconds` sin voz, corta.
    - También corta si supera `max_seconds` (seguridad).
    Devuelve la ruta al WAV o None si falla.
    """
    try:
        # Bip para avisar que ya está grabando
        try:
            winsound.Beep(1000, 300)
        except Exception:
            pass

        logger.info(
            "asr_engine: iniciando grabación con VAD (silence=%.1fs, thresh=%.4f, max=%.1fs)...",
            silence_seconds,
            energy_threshold,
            max_seconds,
        )

        # ✅ Tamaño del bloque en segundos (más corto para reaccionar más rápido)
        block_duration = 0.05  # 50 ms (antes 0.1)
        block_frames = int(block_duration * SAMPLE_RATE)

        # Buffer donde acumularemos todos los bloques
        audio_buffer = []

        start_time = time.time()
        last_voice_time = start_time

        def callback(indata, frames, time_info, status):
            nonlocal audio_buffer, last_voice_time

            if status:
                logger.warning("asr_engine: status de sounddevice: %s", status)

            # Copiamos para no depender del buffer interno
            block = indata.copy().reshape(-1)

            # Guardamos el bloque
            audio_buffer.append(block)

            # Calculamos energía RMS para VAD
            energy = _rms_energy(block)
            now = time.time()

            # 🔍 Log muy ligero de energía (comenta si llena mucho el log)
            # logger.debug("asr_engine: energy=%.5f", energy)

            # Si hay voz (por encima del umbral), actualizamos last_voice_time
            if energy >= energy_threshold:
                last_voice_time = now

        # Abrimos el stream de entrada
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=block_frames,
            callback=callback,
        ):
            while True:
                now = time.time()
                elapsed = now - start_time
                silence_elapsed = now - last_voice_time

                # Seguridad: no pasarnos del tiempo máximo absoluto
                if elapsed >= max_seconds:
                    logger.info(
                        "asr_engine: se alcanzó el tiempo máximo (%.1fs), deteniendo grabación.",
                        max_seconds,
                    )
                    break

                # Si llevamos más de `silence_seconds` sin voz -> cortamos
                if silence_elapsed >= silence_seconds and elapsed > 0:
                    logger.info(
                        "asr_engine: se detectaron %.1f s de silencio, deteniendo grabación.",
                        silence_elapsed,
                    )
                    break

                # Dormimos poquito para no reventar la CPU
                time.sleep(0.03)

        if not audio_buffer:
            logger.warning("asr_engine: no se recibió audio del micrófono.")
            return None

        # Unimos todos los bloques en un solo array
        audio = np.concatenate(audio_buffer, axis=0)

        # Guardamos a WAV
        file_path = AUDIO_INPUT_DIR / "asr_input.wav"
        sf.write(str(file_path), audio, SAMPLE_RATE)
        dur_estimada = len(audio) / float(SAMPLE_RATE)
        logger.info(
            "asr_engine: audio guardado en %s (duración aprox: %.1f s)",
            file_path,
            dur_estimada,
        )

        return file_path

    except Exception:
        logger.exception("asr_engine: error grabando audio del micrófono con VAD.")
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


def escuchar_y_transcribir() -> Optional[str]:
    """
    Flujo completo:
    - Graba usando VAD (sin duración fija).
    - Corta cuando haya ~2.5s sin voz.
    - Transcribe con Whisper.
    Devuelve el texto en español o None si falla.
    """
    wav_path = grabar_una_frase_vad()
    if not wav_path:
        return None
    return transcribir_audio(wav_path)
