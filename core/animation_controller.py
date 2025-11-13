# -*- coding: utf-8 -*-
# core/animation_controller.py
from __future__ import annotations
import os, time, math
from typing import Optional

import soundfile as sf
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

BLENDER_UDP_HOST = os.getenv("BLENDER_UDP_HOST", "127.0.0.1")
BLENDER_UDP_PORT = int(os.getenv("BLENDER_UDP_PORT", "9001"))

def _jaw_from_frame(frame: np.ndarray) -> float:
    # RMS -> normaliza 0..1 con compresión suave
    rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
    jaw = min(1.0, max(0.0, (rms * 6.0)))
    return jaw

def mover_avatar_from_audio(wav_path: str):
    """
    Lee el WAV y manda /jaw 20-40 veces por segundo. Si no hay puente, no truena.
    """
    try:
        client = SimpleUDPClient(BLENDER_UDP_HOST, BLENDER_UDP_PORT)
    except Exception as e:
        print("[anim] ⚠ No pude crear cliente OSC:", e)
        client = None

    try:
        data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        hop = max(1, int(sr / 30))  # ~30 FPS
        total = data.shape[0]
        last_t = time.time()
        i = 0
        while i < total:
            frame = data[i:i+hop, :].mean(axis=1)  # mono
            jaw = _jaw_from_frame(frame)
            if client:
                try:
                    client.send_message("/jaw", jaw)
                except Exception:
                    pass
            # pacing ~ 33ms
            now = time.time()
            delay = 0.033 - (now - last_t)
            if delay > 0:
                time.sleep(delay)
            last_t = time.time()
            i += hop
        # cierra al final
        if client:
            try:
                client.send_message("/jaw", 0.0)
            except Exception:
                pass
    except Exception as e:
        print("[anim] ⚠ Error al animar:", e)
