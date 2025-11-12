# -*- coding: utf-8 -*-
# core/tts_engine.py
# TTS unificado para Totem (XTTS v2 en GPU, con reproducción por sounddevice)

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import tempfile

import sounddevice as sd
import soundfile as sf

# Coqui TTS (XTTS v2)
try:
    import torch
    from TTS.api import TTS
except Exception as e:  # no TTS instalado
    TTS = None
    torch = None

# =========================
# Config y modelo global
# =========================
_TTS_MODEL = None
_DEFAULT_LANG = os.getenv("TTS_LANG", "es")  # fuerza español neutro
# Puedes poner aquí tu voz de referencia por defecto (WAV mono/16k ~ 48k):
_DEFAULT_SPEAKER = os.getenv("TTS_SPEAKER_WAV", "")  # e.g., "mi_referencia.wav"

def _load_tts():
    """Carga perezosa el modelo XTTS v2 en GPU si está disponible."""
    global _TTS_MODEL
    if _TTS_MODEL is not None:
        return _TTS_MODEL
    if TTS is None:
        raise RuntimeError("❌ Falta el paquete 'TTS'. Instala:  python -m pip install TTS==0.21.3")

    use_gpu = False
    if torch is not None:
        try:
            use_gpu = bool(torch.cuda.is_available())
        except Exception:
            use_gpu = False

    _TTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    return _TTS_MODEL

# =========================
# API pública (amain.py)
# =========================
def synth_tts(
    text: str,
    speaker_wav: Optional[str] = None,
    language: Optional[str] = None,
    out_wav: Optional[str] = None,
) -> str:
    """
    Sintetiza 'text' a WAV y devuelve la ruta del archivo.
    - speaker_wav: ruta a WAV de referencia (clonado). Si no existe, usa la voz base.
    - language: código de idioma (por defecto 'es').
    - out_wav: ruta destino; si no se da, se crea un temporal.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("synth_tts: texto vacío")

    model = _load_tts()

    if out_wav is None:
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(fd)
        out_wav = temp_path

    lang = language or _DEFAULT_LANG

    spk = None
    # 1) explícito
    if speaker_wav and Path(speaker_wav).exists():
        spk = speaker_wav
    # 2) variable de entorno por defecto
    elif _DEFAULT_SPEAKER and Path(_DEFAULT_SPEAKER).exists():
        spk = _DEFAULT_SPEAKER

    model.tts_to_file(
        text=text,
        speaker_wav=spk,
        language=lang,
        file_path=out_wav,
    )
    return out_wav


def play_audio(wav_path: str, block: bool = False):
    """
    Reproduce un WAV usando sounddevice. Si block=True, espera a terminar.
    """
    wav_path = str(wav_path)
    if not Path(wav_path).exists():
        raise FileNotFoundError(f"play_audio: no existe {wav_path}")

    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    sd.stop()  # detén cualquier reproducción previa
    sd.play(data, sr)
    if block:
        sd.wait()


def stop_audio():
    """Detiene cualquier reproducción en curso."""
    sd.stop()


__all__ = ["synth_tts", "play_audio", "stop_audio"]
