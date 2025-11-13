# live_test.py  — Grabación + Transcripción + Envío a IA (FastAPI)
# Requisitos: pip install sounddevice numpy requests
# .env (en la raíz del proyecto):
#   OPENAI_API_KEY=sk-...
#   TOTEM_API_BASE=http://127.0.0.1:8000
#   # TOTEM_SESSION_ID=<opcional, si ya tienes una>
#   # MIC_DEVICE=54           # opcional por defecto
#   # MIC_RATE=16000          # 16000 para BT Hands-Free; 44100/48000 para Realtek/USB
#
# Modos:
#   python live_test.py --list
#   python live_test.py --mode list|diag|meter|record [--device N] [--rate 44100]
#   python live_test.py --mode meter --device N --rate 16000
#   python live_test.py --device N --rate 16000 --print_rms --debug

import argparse
import contextlib
import json
import os
import queue
import sys
import time
import wave
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import requests
import sounddevice as sd



#PROGRAMA DE LA VOZ 
from core.voz import hablar


OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-1"
DEFAULT_LANGUAGE = "es"
_SESSION_FILE = ".totem_session"

# -------------------- .env utils --------------------
def _load_dotenv_into_environ(filename: str = ".env"):
    """Carga pares KEY=VALUE al entorno si no existen."""
    try:
        search = [
            os.path.join(os.getcwd(), filename),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        ]
        for path in search:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        os.environ.setdefault(k, v)
                break
    except Exception:
        pass

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    if v is None:
        _load_dotenv_into_environ(".env")
        v = os.environ.get(name, default)
    return v

def _get_openai_key() -> str:
    key = (_get_env("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Falta OPENAI_API_KEY (ponla en .env).")
    return key

# -------------------- WAV utils --------------------
def _write_wav(path: str, pcm_bytes: bytes, sample_rate: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

def _rms_int16(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    x = frame.astype(np.int32)
    return float(np.sqrt(np.mean(x * x))) / 32768.0

# -------------------- List / Diag / Meter --------------------
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

# -------------------- Captura robusta --------------------
def _record_fixed_seconds(device: Optional[int], sample_rate: int, secs=5, channels=1, dtype="int16") -> str:
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
    """Intenta abrir stream con (channels,dtype); devuelve (stream, params) si llega un chunk, si no None."""
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
    device: Optional[int] = None,
    sample_rate: int = 44100,
    frame_ms: int = 30,
    calib_ms: int = 400,
    start_factor: float = 2.0,
    stop_factor: float = 1.4,
    preroll_ms: int = 300,
    start_min_frames: int = 3,
    stop_min_frames: int = 12,
    max_record_ms: int = 15000,
    force_start: bool = False,
    print_rms: bool = False,
    debug: bool = False
) -> str:
    assert frame_ms in (10, 20, 30), "frame_ms debe ser 10, 20 o 30"
    q_in: "queue.Queue[np.ndarray]" = queue.Queue()
    audio_frames: List[np.ndarray] = []
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
        # Sin callback: captura directa 5 s
        return _record_fixed_seconds(device, sample_rate, secs=5, channels=1, dtype="int16")

    channels = used["channels"]
    dtype = used["dtype"]

    voiced = False
    hot_run = 0
    cold_run = 0
    max_frames = int(max_record_ms / frame_ms)
    preroll_frames = int(preroll_ms / frame_ms)
    ring: List[np.ndarray] = []

    # 2) Calibración
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
                    return ""
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

# -------------------- Transcripción (OpenAI) --------------------
def _transcribe(file_path: str, model=DEFAULT_MODEL, language=DEFAULT_LANGUAGE, temperature=0.0) -> str:
    key = _get_openai_key()
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
    return (payload.get("text") or "").strip()

# -------------------- Integración con IA (FastAPI) --------------------
def _api_base() -> str:
    base = _get_env("TOTEM_API_BASE", "http://127.0.0.1:8000") or "http://127.0.0.1:8000"
    return base.rstrip("/")

def _read_session_from_file() -> Optional[str]:
    try:
        if os.path.isfile(_SESSION_FILE):
            with open(_SESSION_FILE, "r", encoding="utf-8") as f:
                sid = f.read().strip()
                if sid:
                    return sid
    except Exception:
        pass
    return None

def _write_session_to_file(sid: str):
    try:
        with open(_SESSION_FILE, "w", encoding="utf-8") as f:
            f.write(sid.strip())
    except Exception:
        pass

def _ensure_session_id(force: bool = False) -> str:
    """
    Devuelve un session_id válido. Si force=True, siempre crea una nueva sesión
    y la guarda en el archivo .totem_session.
    """
    if not force:
        sid_env = (_get_env("TOTEM_SESSION_ID") or "").strip()
        if sid_env:
            return sid_env
        sid_file = _read_session_from_file()
        if sid_file:
            return sid_file

    # crear nueva
    url = f"{_api_base()}/session/start"
    payload = {"client": "live_test"}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()
    sid = data.get("session_id") or data.get("id") or data.get("session") or ""
    if not sid:
        raise RuntimeError(f"Respuesta inesperada en /session/start: {data}")
    _write_session_to_file(sid)
    return sid

def _send_to_ai(user_text: str) -> str:
    """
    Envía el texto a /chat/turn.
    - Prioriza el esquema {'session_id', 'text'} (según tu 422).
    - Si responde 404 'session not found', crea nueva sesión y reintenta una vez.
    - Si responde 422, imprime el detalle para ver qué campo falta.
    """
    def _post_with_sid(sid: str, body: dict) -> requests.Response:
        url = f"{_api_base()}/chat/turn"
        return requests.post(url, json=body, timeout=40)

    # Orden de prueba: primero el que pide tu backend
    payload_shapes = [
        lambda sid: {"session_id": sid, "text": user_text},      # <-- principal
        lambda sid: {"session_id": sid, "user_input": user_text},
        lambda sid: {"session": sid, "user_input": user_text},
        lambda sid: {"sessionId": sid, "userInput": user_text},
        lambda sid: {"session_id": sid, "message": user_text},
        lambda sid: {"session_id": sid, "input": user_text},
    ]

    # 1) asegurar/obtener sesión
    sid = _ensure_session_id(force=False)

    last_err = None
    for shape in payload_shapes:
        body = shape(sid)
        try:
            r = _post_with_sid(sid, body)
            if r.status_code == 200:
                data = r.json()
                return (
                    data.get("assistant")
                    or data.get("reply")
                    or data.get("answer")
                    or data.get("text")
                    or json.dumps(data, ensure_ascii=False)
                )

            # 404 session not found -> crear nueva sesión y reintentar UNA vez con el mismo shape
            try:
                err_json = r.json()
            except Exception:
                err_json = {"detail": r.text}

            if r.status_code == 404 and isinstance(err_json.get("detail"), str) and "session" in err_json["detail"].lower():
                sid = _ensure_session_id(force=True)
                body_retry = shape(sid)
                r2 = _post_with_sid(sid, body_retry)
                if r2.status_code == 200:
                    data = r2.json()
                    return (
                        data.get("assistant")
                        or data.get("reply")
                        or data.get("answer")
                        or data.get("text")
                        or json.dumps(data, ensure_ascii=False)
                    )
                else:
                    try:
                        err2 = r2.json()
                    except Exception:
                        err2 = {"detail": r2.text}
                    print(f"⚠️ /chat/turn rechazó (retry) payload {body_retry} -> status={r2.status_code}, detail={err2}\n")
                    last_err = f"status={r2.status_code}, tried_body={body_retry}, detail={err2}"
                    continue

            # otros códigos (incluye 422 con detalle de Pydantic)
            print(f"⚠️ /chat/turn rechazó payload {body} -> status={r.status_code}, detail={err_json}\n")
            last_err = f"status={r.status_code}, tried_body={body}, detail={err_json}"

        except Exception as e:
            last_err = f"exception={e}"

    raise RuntimeError(f"No pude entregar a /chat/turn con ninguno de los esquemas: {last_err}")


def detectar_voz(threshold=0.02, duracion=0.5, device=None, rate=16000):
    """
    Escucha el micrófono por 'duracion' segundos y devuelve True si detecta sonido.
    threshold: nivel mínimo RMS para considerar que hay voz.
    """
    audio = sd.rec(int(duracion * rate), samplerate=rate, channels=1, dtype="float32", device=device)
    sd.wait()
    rms = np.sqrt(np.mean(np.square(audio)))
    return rms > threshold

# -------------------- Modo record (E2E) --------------------
def mode_record(args):
    # Defaults desde .env si no los pasan por CLI
    device = args.device
    if device is None:
        dev_env = _get_env("MIC_DEVICE")
        if dev_env not in (None, "", "None"):
            try:
                device = int(dev_env)
            except Exception:
                device = None
    rate = args.rate if args.rate is not None else int(_get_env("MIC_RATE", "44100"))

    # Preflight suave
    try:
        sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
    except Exception as e:
        print(f"⚠️ Config no validada (device={device}, rate={rate}): {e}")

    print("✅ Sesión lista. ENTER para grabar | Ctrl+C para salir\n")#----------------------------------------------------------------------------------------------------------------

    while True:
        try:
            print("🎤 Esperando que alguien hable...")
            # Espera hasta que detecte voz
            while not detectar_voz(threshold=0.03, duracion=0.3, device=device, rate=rate):
                time.sleep(0.1)  # evita uso excesivo de CPU

            print("🎙️ ¡Voz detectada! Iniciando grabación...")

            wav_path = _record_until_silence(
                device=device,
                sample_rate=rate,
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

            if not wav_path:
                print("⚠️  No llegó audio del stream; haré una captura directa de 5 s y la transcribo…")
                wav_path = _record_fixed_seconds(device, rate, secs=5, channels=1, dtype="int16")

            print(f"💾 Guardado: {wav_path}")

            # Transcripción
            text = _transcribe(wav_path)
            print(f"📝 Transcripción: {text}\n")

            # Enviar a la IA
            try:
                ai_reply = _send_to_ai(text)
                data = json.loads(ai_reply)
                solo_response = data["response"]
                hablar(solo_response)  # tu función de voz clonada
                print(f"🤖 IA: {solo_response}\n")

            except Exception as e:
                print(f"⚠️ No se pudo enviar a la IA ({_api_base()}): {e}\n")

        except KeyboardInterrupt:
            print("\n👋 Saliendo.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["list", "diag", "meter", "record"], default="record")
    ap.add_argument("--list", action="store_true", help="Alias de --mode list")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--rate", type=int, default=None)
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

    if args.list or args.mode == "list":
        mode_list()
    elif args.mode == "diag":
        mode_diag(args.device, args.rate or 44100, args.secs)
    elif args.mode == "meter":
        mode_meter(args.device, args.rate or 44100, args.frame_ms)
    else:
        mode_record(args)

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()