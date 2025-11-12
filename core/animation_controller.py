# core/animacion.py
# -*- coding: utf-8 -*-
import time, numpy as np, soundfile as sf

def _set_jaw(value: float):
    value = max(0.0, min(0.45, value))
    # En producción manda este valor a tu rig (OSC / Panda3D / Blender)
    print(f"👄 jaw={value:.2f}")

def mover_avatar_from_audio(wav_path: str, fps: int = 30):
    try:
        data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        hop = max(1, int(sr / fps))
        jaw_prev = 0.0
        t0 = time.time()

        for i in range(0, len(data), hop):
            block = data[i:i+hop, :]
            if block.size == 0:
                break
            rms = float(np.sqrt(np.mean(block**2)))   # 0..1 aprox
            target = min(0.45, rms * 2.5)             # ganancia
            jaw = jaw_prev + (target - jaw_prev) * 0.35  # suavizado
            _set_jaw(jaw)
            jaw_prev = jaw

            # Mantén el timing aproximado
            dt = (t0 + (i / sr)) - time.time()
            if dt > 0:
                time.sleep(min(dt, 0.05))
        _set_jaw(0.0)
    except Exception as e:
        print(f"⚠️ Animación error: {e}")
