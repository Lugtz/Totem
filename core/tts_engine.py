# -*- coding: utf-8 -*-
"""
core.tts_engine
Síntesis de voz para Totem Evolución IA3.

- Modelo principal: XTTS v2 (requiere speaker_wav para clonar voz).
- Si no hay WAV de Nacho, usa voz genérica (pero nunca pasa speaker_wav=None).
"""

import os
import time
from pathlib import Path
from typing import Optional, Union

import winsound
from dotenv import load_dotenv
from TTS.api import TTS  # type: ignore

from core.logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# 1. RUTAS Y ENTORNO
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
# 2. CONFIGURACIÓN DE MODELO
# ----------------------------------------------------------------------
TTS_MODEL = os.getenv(
    "TTS_MODEL",
    "tts_models/multilingual/multi-dataset/xtts_v2",
)
TTS_LANG = os.getenv("TTS_LANG", "es")
TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")

# Ruta FIJA al WAV de Nacho (lo que tú me diste)
# C:\Users\kayit\Downloads\Totem\Totem\data\audio\nacho.wav
SPEAKER_WAV_PATH = ROOT_DIR / "data" / "audio" / "nacho.wav"

_tts_instance: Optional[TTS] = None


def _resolve_speaker_wav() -> Optional[Path]:
    """
    Devuelve la ruta al WAV de Nacho si existe, si no, None.
    """
    p = SPEAKER_WAV_PATH
    if p.is_file():
        logger.info("core.tts_engine: usando speaker_wav=%s", p)
        return p

    logger.warning(
        "core.tts_engine: NO se encontró la voz de Nacho en %s, "
        "usaré voz genérica del modelo %s.",
        p,
        TTS_MODEL,
    )
    return None


def _move_to_device(tts: TTS) -> TTS:
    """
    Intenta mover el modelo al dispositivo indicado (cuda/cpu).
    """
    if not TTS_DEVICE:
        return tts

    try:
        tts = tts.to(TTS_DEVICE)
    except Exception:
        logger.exception(
            "core.tts_engine: no se pudo mover el modelo a '%s', se usará CPU.",
            TTS_DEVICE,
        )
    return tts


def _get_tts() -> TTS:
    """
    Devuelve una instancia única del modelo de TTS.
    """
    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance

    logger.info(
        "core.tts_engine: Inicializando TTS → model=%s, lang=%s, device=%s",
        TTS_MODEL,
        TTS_LANG,
        TTS_DEVICE,
    )
    tts = TTS(TTS_MODEL)
    tts = _move_to_device(tts)
    _tts_instance = tts
    return _tts_instance


def synth_tts(texto: str, nombre_archivo: str = "dialogo", **kwargs) -> Optional[Path]:
    """
    Genera un WAV con la voz de Nacho (si hay WAV) o voz genérica.
    - texto: lo que va a decir Nacho.
    - nombre_archivo: base del nombre del archivo de salida (sin extensión).

    Devuelve la ruta al WAV o None si algo falla.

    Acepta **kwargs para ser compatible con llamadas antiguas
    (por ejemplo si alguien le pasa nombre_base, etc.).
    """
    texto = (texto or "").strip()
    if not texto:
        logger.warning("core.tts_engine: synth_tts llamado con texto vacío.")
        return None

    base = Path(nombre_archivo).stem or "dialogo"
    out_path = TTS_OUTPUT_DIR / f"{base}_{int(time.time())}.wav"
    logger.info("core.tts_engine: generando audio en %s...", out_path)

    speaker_path = _resolve_speaker_wav()

    try:
        tts = _get_tts()
        # Si es XTTS y tenemos WAV de Nacho, usamos clonación
        if "xtts" in TTS_MODEL.lower() and speaker_path is not None:
            tts.tts_to_file(
                text=texto,
                speaker_wav=str(speaker_path),
                language=TTS_LANG,
                file_path=str(out_path),
            )
        else:
            # Modelo genérico (o XTTS sin WAV) sin pasar speaker_wav=None
            if "xtts" in TTS_MODEL.lower():
                logger.info(
                    "core.tts_engine: usando XTTS sin WAV de Nacho (voz genérica)."
                )
                tts.tts_to_file(
                    text=texto,
                    language=TTS_LANG,
                    file_path=str(out_path),
                )
            else:
                # Otros modelos (es/css10/vits, etc.)
                tts.tts_to_file(
                    text=texto,
                    file_path=str(out_path),
                )

        logger.info(
            "core.tts_engine: audio TTS generado correctamente en %s", out_path
        )
        return out_path
    except Exception:
        logger.exception(
            "core.tts_engine: error al sintetizar TTS (modelo=%s).",
            TTS_MODEL,
        )
        return None


def play_audio(ruta: Optional[Union[str, Path]]) -> None:
    """
    Reproduce un WAV (bloqueante). Si la ruta es None o no existe, no hace nada.
    """
    if ruta is None:
        logger.warning(
            "core.tts_engine: play_audio llamado con ruta=None; se omite reproducción."
        )
        return

    path = Path(ruta)
    if not path.is_file():
        logger.warning(
            "core.tts_engine: play_audio no encontró el archivo de audio: %s",
            path,
        )
        return

    try:
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    except Exception:
        logger.exception(
            "core.tts_engine: error reproduciendo audio con winsound (%s).",
            path,
        )
