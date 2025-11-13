# -*- coding: utf-8 -*-
"""
core/camera_agent.py

Módulo de detección de personas con YOLO + saludo y conversación por voz
contra el backend FastAPI (amain.py).

Flujo:
- Se carga el modelo YOLO (yolov8n.pt) y se abre la cámara 0.
- Cuando detecta una o más personas, se:
    1) Genera y reproduce un saludo con TTS.
    2) Llama a /session/start para obtener session_id.
    3) Inicia un bucle de conversación por voz:
        - Espera el bip.
        - Usa ASR con VAD (webrtcvad) hasta 2 s de silencio (máx ~20 s).
        - Envía el texto a /chat/turn.
        - Reproduce la respuesta con TTS.
        - Termina cuando el backend devuelve terminar=True o no hay texto.
"""

import time
import threading
from typing import Optional

import cv2
import requests
from ultralytics import YOLO

from core.logger import get_logger
from core.tts_engine import synth_tts, play_audio
from core.asr_engine import escuchar_y_transcribir

# ----------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------
logger = get_logger(__name__)

API_BASE_URL = "http://127.0.0.1:8000"

YOLO_MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
PERSON_CLASS_ID = 0           # ID de "person" en COCO
CONFIDENCE_THRESHOLD = 0.5    # Umbral de confianza mínimo

SALUDO_COOLDOWN_SECONDS = 10.0  # Tiempo mínimo entre saludos para no spamear

# ----------------------------------------------------------------------
# Estado interno del detector
# ----------------------------------------------------------------------
_model: Optional[YOLO] = None
_detector_thread: Optional[threading.Thread] = None
_running: bool = False


# ----------------------------------------------------------------------
# Utilidad opcional: bip antes de escuchar
# ----------------------------------------------------------------------
try:
    import winsound
except ImportError:  # Linux / Mac
    winsound = None


def _beep():
    """Emite un bip corto en Windows; en otros SO no hace nada."""
    if winsound is not None:
        try:
            winsound.Beep(1200, 400)
        except Exception:
            # Si algo falla con el beep, no queremos tirar el flujo
            logger.warning("[camera] No se pudo reproducir el bip.")


# ----------------------------------------------------------------------
# Conversación por voz
# ----------------------------------------------------------------------
def _iniciar_conversacion_local(session_id: str) -> None:
    """
    Bucle de conversación por voz:
    - Pide audio al usuario usando ASR con VAD.
    - Envía cada turno a /chat/turn.
    - Reproduce la respuesta con TTS.
    - Termina cuando el backend indica terminar=True o no hay texto.
    """
    logger.info(
        "\n================ DIÁLOGO POR VOZ CON NACHO ================\n"
    )
    print(
        "\n================ DIÁLOGO POR VOZ CON NACHO ================\n\n"
        "Cuando escuches el bip, habla normal.\n"
        "Di algo como 'gracias Nacho' o 'adiós' para terminar la conversación.\n"
    )

    while True:
        # Bip para indicar que ya puede hablar
        print("\n🎙️ Habla después del bip (me detengo tras ~2 s de silencio)...")
        _beep()

        # ✅ Aquí usamos tu nuevo ASR con VAD (20 s máx, 2 s de silencio)
        user_text = escuchar_y_transcribir()  # <-- SIN 'segundos='

        if not user_text:
            logger.info("[camera] No se reconoció texto; terminando conversación.")
            print("⚠️ No se entendió nada, finalizando esta conversación.\n")
            break

        logger.info("👤 Tú (transcrito): %s", user_text)
        print(f"👤 Tú (transcrito): {user_text}\n")

        # Llamamos al backend /chat/turn
        payload = {
            "session_id": session_id,
            "texto": user_text,
            "via": "voz",
        }

        try:
            r = requests.post(f"{API_BASE_URL}/chat/turn", json=payload, timeout=60)
            r.raise_for_status()
        except Exception as e:
            logger.exception("[camera] Error llamando a /chat/turn.")
            print(f"❌ Error llamando a /chat/turn: {e}")
            break

        data = {}
        try:
            data = r.json()
        except Exception:
            logger.exception("[camera] No se pudo parsear JSON de /chat/turn.")
            print("❌ Respuesta no válida de /chat/turn.")
            break

        respuesta = data.get("respuesta", "") or ""
        terminar = bool(data.get("terminar", False))

        # Mostramos la respuesta del asistente
        logger.info("🤖 Nacho: %s", respuesta)
        print(f"🤖 Nacho: {respuesta}\n")

        # Reproducir TTS de la respuesta
        try:
            audio_path = synth_tts(
                respuesta,
                nombre_archivo=f"dialogo_{int(time.time())}.wav",
            )
            play_audio(audio_path)
        except Exception:
            logger.exception("[camera] Error al reproducir respuesta TTS desde /chat/turn.")

        if terminar:
            logger.info("[camera] Conversación por voz finalizada (terminar=True).")
            print("🔚 Nacho dio por terminada la conversación.\n")
            break

    logger.info("[camera] Conversación por voz finalizada.")
    print("✅ Conversación por voz finalizada.\n")


def _saludar_visitante() -> None:
    """
    Saludo inicial cuando YOLO detecta una persona:
    - Genera y reproduce saludo TTS.
    - Llama a /session/start para crear una sesión de diálogo.
    - Inicia el bucle de conversación por voz.
    """
    logger.info("Persona detectada → iniciando conversación.")
    saludo = (
        "Hola, ¿cómo estás? "
        "Soy Nacho, el asistente virtual de Evolución i3. "
        "Podemos conversar un momento y, si quieres, te ayudo a crear una propuesta para tu empresa."
    )

    try:
        audio_path = synth_tts(
            saludo,
            nombre_archivo=f"saludo_yolo_{int(time.time())}.wav",
        )
        play_audio(audio_path)
    except Exception:
        logger.exception("[camera] Error al reproducir saludo TTS.")

    # Crear sesión en el backend
    try:
        resp = requests.post(f"{API_BASE_URL}/session/start", json={"via": "voz"}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        session_id = data.get("session_id")
        logger.info("[camera] Sesión de diálogo iniciada: %s", session_id)
    except Exception:
        logger.exception("[camera] Error creando sesión en /session/start.")
        return

    if not session_id:
        logger.error("[camera] /session/start no devolvió session_id.")
        return

    # Iniciar bucle de conversación con esa sesión
    try:
        _iniciar_conversacion_local(session_id)
    except Exception:
        logger.exception("[camera] Error en la conversación por voz después del saludo.")


# ----------------------------------------------------------------------
# Bucle de detección con YOLO
# ----------------------------------------------------------------------
def _detectar_personas_loop() -> None:
    """
    Hilo que:
    - Lee frames de la cámara.
    - Ejecuta YOLOv8 para detectar personas.
    - Si detecta al menos una persona y ha pasado el cooldown,
      llama a _saludar_visitante().
    """
    global _running, _model

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("[camera] No se pudo abrir la cámara %s.", CAMERA_INDEX)
        return

    logger.info("[camera] Detector activo en cámara %s usando YOLO/COCO.", CAMERA_INDEX)

    last_saludo_time = 0.0

    try:
        while _running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            # Ejecutar YOLO sobre el frame
            if _model is None:
                logger.error("[camera] Modelo YOLO no inicializado.")
                time.sleep(0.5)
                continue

            results = _model(frame, verbose=False)
            num_personas = 0

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for cls_id, conf in zip(boxes.cls, boxes.conf):
                    if int(cls_id) == PERSON_CLASS_ID and float(conf) >= CONFIDENCE_THRESHOLD:
                        num_personas += 1

            if num_personas > 0:
                logger.info("[camera] Persona detectada por YOLO (n=%d).", num_personas)
                now = time.time()
                # Cooldown para no disparar saludo sin parar
                if now - last_saludo_time >= SALUDO_COOLDOWN_SECONDS:
                    last_saludo_time = now
                    _saludar_visitante()

            # Pequeña pausa para no saturar CPU
            time.sleep(0.05)
    finally:
        cap.release()
        logger.info("[camera] Detector detenido.")


# ----------------------------------------------------------------------
# API pública
# ----------------------------------------------------------------------
def iniciar_detector() -> None:
    """
    Función pública llamada desde amain.py
    - Carga el modelo YOLO si no está cargado.
    - Lanza el hilo de detección si no está ya corriendo.
    """
    global _model, _detector_thread, _running

    if _detector_thread and _detector_thread.is_alive():
        logger.warning("[camera] El detector ya está en ejecución.")
        return

    logger.info("[camera] Cargando modelo YOLO (%s)...", YOLO_MODEL_PATH)
    try:
        _model = YOLO(YOLO_MODEL_PATH)
    except Exception:
        logger.exception("[camera] Error al cargar modelo YOLO.")
        _model = None
        return

    _running = True
    _detector_thread = threading.Thread(
        target=_detectar_personas_loop,
        name="camera-detector",
        daemon=True,
    )
    _detector_thread.start()
    logger.info("[camera] Hilo de detección iniciado.")


def detener_detector() -> None:
    """
    Detiene el detector de personas (si se está usando en otros contextos).
    """
    global _running, _detector_thread
    _running = False
    if _detector_thread and _detector_thread.is_alive():
        _detector_thread.join(timeout=2.0)
    logger.info("[camera] detener_detector() llamado.")
