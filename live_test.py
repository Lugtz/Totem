# live_test.py  (robusto: list/diag/meter/record)  SIN webrtcvad
# .env: lee OPENAI_API_KEY automáticamente.
# Soporta:
#   --mode list|diag|meter|record   y también  --list  (alias de list)
#
import argparse
import contextlib
import os
import queue
import sys
import time
import wave
from datetime import datetime

import numpy as np
import requests
import sounddevice as sd

OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-1"
DEFAULT_LANGUAGE = "es"

# ---------- .env loader ----------
def _load_dotenv_into_environ(filename: str = ".env"):
    try:
        path = filename if os.path.isabs(filename) else os.path.join(os.getcwd(), filename)
        if not os.path.isfile(path):
            here = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(here, filename)
            if not os.path.isfile(path):
                return
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                os.environ.setdefault(k, v)
    except Exception:
        pass

def _get_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    _load_dotenv_into_environ(".env")
    return os.environ.get("OPENAI_API_KEY", "").strip()

# ---------- util WAV ----------
def _write_wav(path: str, pcm_bytes: bytes, sample_rate: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

def _rms_int16(frame: np.ndarray) -> float:
    f32 = frame.astype(np.int32)
    if f32.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(f32 * f32))
    return float(rms) / 32768.0

# ---------- modos ----------
def mode_list():
    devices = sd.query_devices()
    default_in, default_out = sd.default.device
    print("\n=== DISPOSITIVOS DE AUDIO ===")
    for idx, d in enumerate(devices):
        mark = "  <-- DEFAULT INPUT" if idx == default_in else ""
        dsr = d.get("default_samplerate", "n/a")
        print(f"[{idx}] {d['name']} (in:{d['max_input_channels']} out:{d['max_output_channels']} default_sr:{dsr}){mark}")
    print(f"\nDefault input index: {default_in}")
    print(f"Default output index: {default_out}\n")

def mode_diag(device: int, rate: int, secs: float):
    print(f"Preflight: device={device}, rate={rate}, secs={secs}")
    sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
    print("✓ check_input_settings OK")
    frames = int(rate * secs)
    buf = sd.rec(frames, samplerate=rate, channels=1, dtype="int16", device=device)
    sd.wait()
    x = buf.reshape(-1).astype(np.int32)
    rms = float(np.sqrt(np.mean(x * x))) / 32768.0 if x.size else 0.0
    print(f"✓ stream OK, RMS_promedio={rms:.4f} (habla durante la prueba para ver >0.02 aprox)")

def mode_meter(device: int, rate: int, frame_ms: int):
    q = queue.Queue()
    frame = int(rate * frame_ms / 1000)
    def cb(indata, frames, time_info, status):
        if status: pass
        q.put(indata.copy())
    with sd.InputStream(samplerate=rate, channels=1, dtype="int16", callback=cb, device=device):
        print("Medidor RMS (Ctrl+C para salir)")
        while True:
            data = q.get().reshape(-1)
            for i in range(0, len(data), frame):
                chunk = data[i:i + frame]
                if len(chunk) < frame: break
                r = _rms_int16(chunk)
                bar = "▮" * int(min(50, r * 100))
                sys.stdout.write(f"\rRMS:{r:.3f} {bar:<50}")
                sys.stdout.flush()

# ---------- captura robusta ----------
def _record_fixed_seconds(device, sample_rate, secs=5, channels=1, dtype="int16"):
    frames = int(sample_rate * secs)
    data = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype=dtype, device=device)
    sd.wait()
    if dtype == "float32":
        data16 = np.clip((data * 32767.0), -32768, 32767).astype(np.int16)
    else:
        data16 = data
    pcm = data16.reshape(-1).tobytes(order="C")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("recordings", f"{ts}_fallback.wav")
    _write_wav(path, pcm, sample_rate)
    return path

def _try_stream_once(device, sample_rate, frame_ms, attempt, q_in, first_chunk_timeout):
    channels, dtype = attempt
    frame_size = int(sample_rate * frame_ms / 1000)
    def callback(indata, frames, time_info, status):
        if status: pass
        q_in.put(indata.copy())
    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            callback=callback,
            device=device,
            blocksize=frame_size,
            latency="low"
        )
        stream.__enter__()
    except Exception:
        return None, None
    deadline = time.time() + first_chunk_timeout
    got = False
    while time.time() < deadline and not got:
        try:
            _ = q_in.get(timeout=0.5)
            got = True
        except queue.Empty:
            pass
    if not got:
        stream.__exit__(None, None, None)
        return None, None
    return (stream, {"channels": channels, "dtype": dtype})

def _record_until_silence(
    device=None,
    sample_rate=44100,
    frame_ms=30,
    calib_ms=400,
    start_factor=2.0,
    stop_factor=1.4,
    preroll_ms=300,
    start_min_frames=3,
    stop_min_frames=12,
    max_record_ms=15000,
    force_start=False,
    print_rms=False,
    debug=False
):
    assert frame_ms in (10, 20, 30), "frame_ms debe ser 10, 20 o 30"
    q_in = queue.Queue()
    audio_frames = []
    frame_size = int(sample_rate * frame_ms / 1000)

    # 1) Intentar varias combinaciones hasta que llegue el primer chunk
    attempts = [ (1, "int16"), (1, "float32"), (2, "int16"), (2, "float32") ]
    stream = None
    used = None
    for attempt in attempts:
        stream, used = _try_stream_once(device, sample_rate, frame_ms, attempt, q_in, first_chunk_timeout=3.0)
        if stream is not None:
            break
    if stream is None:
        # Sin callback: fallback a captura directa 5 s
        return _record_fixed_seconds(device, sample_rate, secs=5, channels=1, dtype="int16")

    channels = used["channels"]
    dtype = used["dtype"]

    voiced = False
    hot_run = 0
    cold_run = 0
    max_frames = int(max_record_ms / frame_ms)
    preroll_frames = int(preroll_ms / frame_ms)
    ring = []

    # 2) Calibración con el stream ya abierto
    need = max(1, int(calib_ms / frame_ms))
    noise_vals = []
    t0 = time.time()
    while len(noise_vals) < need and (time.time() - t0) < 5.0:
        try:
            chunk = q_in.get(timeout=1.0)
        except queue.Empty:
            continue
        if dtype == "float32":
            chunk = np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16)
        chunk = chunk.reshape(-1)
        i = 0
        while i + frame_size <= len(chunk) and len(noise_vals) < need:
            fr = chunk[i:i + frame_size]
            i += frame_size
            noise_vals.append(_rms_int16(fr))
    if not noise_vals:
        stream.__exit__(None, None, None)
        return ""

    noise_floor = max(1e-5, float(np.median(noise_vals)))
    start_thr = noise_floor * start_factor
    stop_thr = noise_floor * stop_factor
    if debug:
        print(f"[DEBUG] ({channels}ch,{dtype}) noise_floor={noise_floor:.5f}  start_thr={start_thr:.5f}  stop_thr={stop_thr:.5f}")

    print("🎤 Grabando (arranque forzado)..." if force_start else "🎤 Escuchando (habla para iniciar)...")
    if force_start:
        voiced = True
        ring.clear()

    t_start = time.time()
    noaudio_deadline = time.time() + 8.0   # si no llega audio en 8s, aborta limpio
    try:
        while True:
            try:
                chunk = q_in.get(timeout=0.8)
            except queue.Empty:
                if not voiced and time.time() > noaudio_deadline:
                    stream.__exit__(None, None, None)
                    return ""  # hará fallback afuera
                if not voiced:
                    continue
                else:
                    break

            if dtype == "float32":
                chunk = np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16)

            chunk = chunk.reshape(-1)
            i = 0
            while i + frame_size <= len(chunk):
                fr = chunk[i:i + frame_size]
                i += frame_size
                r = _rms_int16(fr)

                if print_rms:
                    bar = "▮" * int(min(50, r * 100))
                    sys.stdout.write(f"\rRMS:{r:.3f} hot:{hot_run:02d} cold:{cold_run:02d} {bar:<50}")
                    sys.stdout.flush()

                if not voiced:
                    ring.append(fr)
                    if len(ring) > preroll_frames: ring.pop(0)
                    if r > start_thr:
                        hot_run += 1
                        if hot_run >= start_min_frames:
                            audio_frames.extend(ring)
                            ring.clear()
                            voiced = True
                            hot_run = 0
                            cold_run = 0
                    else:
                        hot_run = 0
                else:
                    audio_frames.append(fr)
                    if r < stop_thr:
                        cold_run += 1
                        if cold_run >= stop_min_frames: break
                    else:
                        cold_run = 0

                if voiced and len(audio_frames) >= max_frames:
                    break

            if voiced and (cold_run >= stop_min_frames or len(audio_frames) >= max_frames):
                break
            if (time.time() - t_start) > 60 and not voiced:
                stream.__exit__(None, None, None)
                return ""
    finally:
        try:
            stream.__exit__(None, None, None)
        except Exception:
            pass

    if print_rms: print()
    if not audio_frames:
        return ""

    pcm = b"".join([f.tobytes() for f in audio_frames])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("recordings", f"{ts}.wav")
    _write_wav(path, pcm, sample_rate)
    return path

# ---------- transcripción ----------
def _transcribe(file_path: str, model=DEFAULT_MODEL, language=DEFAULT_LANGUAGE, temperature=0.0):
    key = _get_openai_key()
    if not key:
        raise RuntimeError("Falta OPENAI_API_KEY (agrega en .env como OPENAI_API_KEY=sk-XXXX).")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    headers = {"Authorization": f"Bearer {key}"}
    data = {
        "model": model,
        "language": language,
        "temperature": str(temperature),
        "response_format": "verbose_json"
    }
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "audio/wav")}
        r = requests.post(OPENAI_TRANSCRIBE_URL, headers=headers, data=data, files=files, timeout=120)
        r.raise_for_status()
        payload = r.json()
    return payload.get("text", "").strip()

def mode_record(args):
    try:
        sd.check_input_settings(device=args.device, samplerate=args.rate, channels=1, dtype="int16")
    except Exception as e:
        print(f"❌ Configuración no soportada (device={args.device}, rate={args.rate}): {e}")
        return
    print("✅ Sesión lista. ENTER para grabar | Ctrl+C para salir\n")
    while True:
        try:
            _ = input("Listo. ENTER para grabar... ")
        except KeyboardInterrupt:
            print("\n👋 Saliendo.")
            return
        try:
            wav_path = _record_until_silence(
                device=args.device,
                sample_rate=args.rate,
                calib_ms=args.calib_ms,
                start_factor=args.start_factor,
                stop_factor=args.stop_factor,
                start_min_frames=args.start_min_frames,
                stop_min_frames=args.stop_min_frames,
                max_record_ms=args.max_ms,
                force_start=args.force_start,
                print_rms=args.print_rms,
                debug=args.debug
            )
        except KeyboardInterrupt:
            print("\n(Interrumpido durante la escucha)\n")
            continue

        if not wav_path:
            print("⚠️  No llegó audio del stream; haré una captura directa de 5 s y la transcribo…")
            wav_path = _record_fixed_seconds(args.device, args.rate, secs=5, channels=1, dtype="int16")

        print(f"💾 Guardado: {wav_path}")
        try:
            text = _transcribe(wav_path)
            print(f"📝 Transcripción: {text}\n")
        except Exception as e:
            print(f"❌ Error en transcripción: {e}\n")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["list", "diag", "meter", "record"], default="record")
    ap.add_argument("--list", action="store_true", help="Alias: lista dispositivos y sale")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--rate", type=int, default=44100)
    # diag
    ap.add_argument("--secs", type=float, default=1.0)
    # meter
    ap.add_argument("--frame_ms", type=int, default=50)
    # record (VAD energía)
    ap.add_argument("--calib_ms", type=int, default=400)
    ap.add_argument("--start_factor", type=float, default=2.0)
    ap.add_argument("--stop_factor", type=float, default=1.4)
    ap.add_argument("--start_min_frames", type=int, default=3)
    ap.add_argument("--stop_min_frames", type=int, default=12)
    ap.add_argument("--max_ms", type=int, default=15000)
    ap.add_argument("--force_start", action="store_true")
    ap.add_argument("--print_rms", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.list:
        mode_list()
        return

    if args.mode == "list":
        mode_list()
    elif args.mode == "diag":
        mode_diag(args.device, args.rate, args.secs)
    elif args.mode == "meter":
        mode_meter(args.device, args.rate, args.frame_ms)
    else:
        mode_record(args)

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
