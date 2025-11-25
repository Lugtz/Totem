# -*- coding: utf-8 -*-
"""
core/camera_agent.py

Módulo de detección de personas con YOLO + saludo y conversación por voz
contra el backend FastAPI (amain.py).

Flujo:
- Se carga el modelo YOLO (yolov8n.pt) y se abre la cámara 0.
- Cuando detecta una o más personas (aplicando filtros de zona y tamaño),
    1) Genera y reproduce un saludo con TTS.
    2) Llama a /session/start para obtener session_id.
    3) Inicia un bucle de conversación por voz:
        - Espera el bip.
        - Usa ASR con VAD (webrtcvad) hasta 2 s de silencio (máx ~20 s).
        - Envía el texto a /chat/turn.
        - Reproduce la respuesta con TTS.
        - Termina cuando el backend devuelve terminar=True o no hay texto.

Mejoras para expo:
- ROI (zona de interés) en el centro de la imagen.
- Filtro por tamaño de la caja (solo personas "cerca").
- Requiere varios frames seguidos con persona antes de saludar.
- Evita iniciar nueva conversación si ya hay una activa.
- Reproduce un audio de invitación en loop mientras no hay personas.
"""

import os
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

# Tiempo mínimo entre saludos para no spamear
SALUDO_COOLDOWN_SECONDS = 15.0  # puedes subirlo a 20–30 s en expo

# Audio de invitación en modo idle (loop)
IDLE_AUDIO_PATH = os.path.join("data", "audio", "idle_loop.mp3")

# ----------------------------------------------------------------------
# Filtros para expo / tótem
# ----------------------------------------------------------------------
# ROI (Relative coordinates 0.0–1.0). Solo contamos personas en esta zona.
# Ejemplo: banda central de la imagen.
ROI_X_MIN = 0.20
ROI_X_MAX = 0.80
ROI_Y_MIN = 0.15
ROI_Y_MAX = 0.95

# Tamaño mínimo de la persona (en proporción a la altura del frame)
# Para considerar que la persona está "cerca" del tótem.
MIN_PERSON_HEIGHT_RATIO = 0.35  # 35% de la altura del frame (ajusta según tu cámara)

# Número de frames consecutivos con persona válida antes de saludar
FRAMES_ESTABLES_DETECCION = 5

# ----------------------------------------------------------------------
# Estado interno del detector
# ----------------------------------------------------------------------
_model: Optional[YOLO] = None
_detector_thread: Optional[threading.Thread] = None
_running: bool = False

# Frames con persona válida en ROI
_frames_con_persona: int = 0

# Flag para no iniciar varias conversaciones a la vez
_conversacion_activa: bool = False

# Estado para audio idle
_idle_thread: Optional[threading.Thread] = None
_idle_running: bool = False
_persona_presente: bool = False

# Lock de audio para no solapar reproducciones
_audio_lock = threading.Lock()

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
            logger.warning("[camera] No se pudo reproducir el bip.")


# ----------------------------------------------------------------------
# Utilidades de audio
# ----------------------------------------------------------------------
def _play_audio_seguro(path: str) -> None:
    """
    Wrapper para reproducir audio sin solapar con otros sonidos.
    Usa un lock global para evitar que suenen dos cosas a la vez.
    """
    if not path:
        return
    with _audio_lock:
        try:
            play_audio(path)
        except Exception:
            logger.exception("[camera] Error reproduciendo audio: %s", path)


# ----------------------------------------------------------------------
# Utilidades de filtro
# ----------------------------------------------------------------------
def _persona_valida_en_roi(box, frame_width: int, frame_height: int) -> bool:
    """
    Aplica filtros de:
    - Zona de interés (ROI) en coordenadas relativas.
    - Tamaño mínimo de la caja (altura relativa).

    Devuelve True solo si la persona está en la zona frontal y lo
    suficientemente cerca.
    """
    # box.xyxy[0] = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    w = x2 - x1
    h = y2 - y1

    # Centro de la caja en píxeles
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    # Normalizar a 0–1
    cx_norm = cx / float(frame_width)
    cy_norm = cy / float(frame_height)
    h_ratio = h / float(frame_height)

    dentro_roi = (
        ROI_X_MIN <= cx_norm <= ROI_X_MAX and
        ROI_Y_MIN <= cy_norm <= ROI_Y_MAX
    )
    suficiente_altura = h_ratio >= MIN_PERSON_HEIGHT_RATIO

    # Debug opcional:
    logger.debug(
        "[camera] box: cx=%.2f, cy=%.2f, h_ratio=%.2f, ROI=%s, cerca=%s",
        cx_norm, cy_norm, h_ratio, dentro_roi, suficiente_altura
    )

    return dentro_roi and suficiente_altura


# ----------------------------------------------------------------------
# Loop de audio idle
# ----------------------------------------------------------------------
def _idle_audio_loop() -> None:
    """
    Reproduce en loop un audio de invitación mientras:
    - No haya una conversación activa.
    - No haya personas válidas en el ROI.
    """
    global _idle_running

    logger.info("[camera] Loop de audio idle iniciado.")
    while _idle_running:
        # Si hay conversación o persona presente, no reproducimos
        if _conversacion_activa or _persona_presente:
            time.sleep(0.5)
            continue

        # Si no existe el archivo, solo avisamos de vez en cuando
        if not os.path.exists(IDLE_AUDIO_PATH):
            logger.warning(
                "[camera] Audio idle no encontrado en %s. "
                "Crea este archivo para usar el modo invitación.",
                IDLE_AUDIO_PATH,
            )
            time.sleep(5.0)
            continue

        logger.info("[camera] Reproduciendo audio idle: %s", IDLE_AUDIO_PATH)
        _play_audio_seguro(IDLE_AUDIO_PATH)
        # Al terminar, el while decidirá si lo repite o no según el estado

    logger.info("[camera] Loop de audio idle detenido.")


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
        print("\n🎙️ Habla después del bip (me detengo tras ~2 s de silencio)...")
        _beep()

        # ASR con VAD (20 s máx, 2 s de silencio)
        user_text = escuchar_y_transcribir()

        if not user_text:
            logger.info("[camera] No se reconoció texto; terminando conversación.")
            print("⚠️ No se entendió nada, finalizando esta conversación.\n")
            break

        logger.info("👤 Tú (transcrito): %s", user_text)
        print(f"👤 Tú (transcrito): {user_text}\n")

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

        try:
            data = r.json()
        except Exception:
            logger.exception("[camera] No se pudo parsear JSON de /chat/turn.")
            print("❌ Respuesta no válida de /chat/turn.")
            break

        respuesta = data.get("respuesta", "") or ""
        terminar = bool(data.get("terminar", False))

        logger.info("🤖 Nacho: %s", respuesta)
        print(f"🤖 Nacho: {respuesta}\n")

        try:
            audio_path = synth_tts(
                respuesta,
                nombre_archivo=f"dialogo_{int(time.time())}.wav",
            )
            _play_audio_seguro(audio_path)
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
    - Genera y reproduce saludo TTS (o audio pregrabado si así se desea).
    - Llama a /session/start para crear una sesión de diálogo.
    - Inicia el bucle de conversación por voz.
    """
    global _conversacion_activa

    # Si ya hay conversación, no iniciar otra
    if _conversacion_activa:
        logger.info("[camera] Ya hay una conversación activa; se ignora nuevo saludo.")
        return

    _conversacion_activa = True
    try:
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
            _play_audio_seguro(audio_path)
        except Exception:
            logger.exception("[camera] Error al reproducir saludo TTS.")

        # Crear sesión en el backend
        try:
            resp = requests.post(
                f"{API_BASE_URL}/session/start",
                json={"via": "voz"},
                timeout=20
            )
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
    finally:
        # Libera el lock de conversación
        _conversacion_activa = False


# ----------------------------------------------------------------------
# Bucle de detección con YOLO
# ----------------------------------------------------------------------
def _detectar_personas_loop() -> None:
    """
    Hilo que:
    - Lee frames de la cámara.
    - Ejecuta YOLOv8 para detectar personas.
    - Aplica filtros de ROI, tamaño y estabilidad.
    - Si detecta al menos una persona válida y ha pasado el cooldown,
      llama a _saludar_visitante().
    """
    global _running, _model, _frames_con_persona, _persona_presente

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

            if _model is None:
                logger.error("[camera] Modelo YOLO no inicializado.")
                time.sleep(0.5)
                continue

            H, W = frame.shape[:2]

            # Ejecutar YOLO sobre el frame
            results = _model(frame, verbose=False)

            personas_validas = 0

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id != PERSON_CLASS_ID or conf < CONFIDENCE_THRESHOLD:
                        continue

                    # Filtro de ROI + tamaño
                    if _persona_valida_en_roi(box, W, H):
                        personas_validas += 1

            if personas_validas > 0:
                _frames_con_persona += 1
            else:
                _frames_con_persona = 0

            # Actualizamos flag de persona presente (o conversación activa)
            if personas_validas > 0 or _conversacion_activa:
                _persona_presente = True
            else:
                _persona_presente = False

            if personas_validas > 0:
                logger.info(
                    "[camera] Personas válidas en ROI (cerca): n=%d, frames_con_persona=%d",
                    personas_validas, _frames_con_persona
                )

            # Solo disparamos saludo si:
            # - Hay personas válidas.
            # - Llevamos varios frames seguidos viéndolas.
            # - Ya pasó el cooldown.
            # - No hay conversación activa.
            now = time.time()
            if (
                personas_validas > 0 and
                _frames_con_persona >= FRAMES_ESTABLES_DETECCION and
                (now - last_saludo_time) >= SALUDO_COOLDOWN_SECONDS and
                not _conversacion_activa
            ):
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
    - Inicia el loop de audio idle.
    """
    global _model, _detector_thread, _running, _idle_thread, _idle_running

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

    # Iniciar loop de audio idle
    if not _idle_running:
        _idle_running = True
        _idle_thread = threading.Thread(
            target=_idle_audio_loop,
            name="camera-idle-audio",
            daemon=True,
        )
        _idle_thread.start()
        logger.info("[camera] Hilo de audio idle iniciado.")


def detener_detector() -> None:
    """
    Detiene el detector de personas (si se está usando en otros contextos)
    y el loop de audio idle.
    """
    global _running, _detector_thread, _idle_running, _idle_thread

    _running = False
    if _detector_thread and _detector_thread.is_alive():
        _detector_thread.join(timeout=2.0)

    _idle_running = False
    if _idle_thread and _idle_thread.is_alive():
        _idle_thread.join(timeout=2.0)

    logger.info("[camera] detener_detector() llamado.")
# ----------------------------------------------------------------------
# Ejecución directa como script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Permite ejecutar el detector directamente con:
        python -m core.camera_agent

    - Inicia el detector de personas.
    - Inicia el loop de audio idle.
    - Mantiene el proceso vivo hasta Ctrl + C.
    """
    logger.info("[camera] Ejecutando camera_agent en modo standalone...")

    try:
        iniciar_detector()

        # Mantener el proceso vivo
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("[camera] Deteniendo detector por teclado...")
        detener_detector()
        print("[camera] Salida limpia.")
