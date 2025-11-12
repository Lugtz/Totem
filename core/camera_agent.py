"""
Totem – camera_agent.py

Cámara UVC + (opcional) control OBSBOT Tiny 2 Lite.
- Captura estable con OpenCV (Windows: DirectShow/MSMF)
- Cola de frames en hilo aparte
- "Patrullaje" por *software* (pan digital por recorte) si no hay PTZ
- Hook para detector (YOLO u HOG) con recentrado automático
- (Opcional) Control PTZ por atajos configurables de OBSBOT Center vía pyautogui

Uso rápido (stand‑alone):
    python camera_agent.py --list
    python camera_agent.py --device 0 --res 1280x720 --fps 30 --show
    # Teclas en ventana:
    #  q: salir | p: patrullaje SW | g: alternar detector (YOLO->HOG->OFF)
    #  o: alternar "patrulla" PTZ (hotkey OBSBOT) | ← → ↑ ↓ : PTZ con hotkeys

NOTAS OBSBOT:
- Configura en OBSBOT Center atajos globales para PTZ (izq/der/arr/abajo) y Auto‑Scan/Patrol.
- Decláralos como variables de entorno (ejemplo más abajo) o usa los defaults.

Env vars (opcionales):
    OBSBOT_LEFT="ctrl+alt+left"     OBSBOT_RIGHT="ctrl+alt+right"
    OBSBOT_UP="ctrl+alt+up"         OBSBOT_DOWN="ctrl+alt+down"
    OBSBOT_PATROL_TOGGLE="ctrl+alt+p"

"""
from __future__ import annotations

import os
import sys
import cv2
import time
import math
import queue
import ctypes
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

# ===== Utilidades de plataforma =====
_IS_WIN = os.name == "nt"

# En Windows, forzamos backend preferido
_CAP_BACKEND = None
if _IS_WIN:
    # CAP_MSMF suele dar baja latencia; CAP_DSHOW expone más controles.
    try:
        _CAP_BACKEND = cv2.CAP_DSHOW  # fallback razonable
    except Exception:
        _CAP_BACKEND = 0


def _set_thread_name(name: str) -> None:
    try:
        threading.current_thread().name = name
    except Exception:
        pass


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: Optional[int] = _CAP_BACKEND
    queue_size: int = 2  # mantener baja latencia
    convert_rgb: bool = False  # OpenCV entrega BGR; poner True si requieres RGB
    auto_exposure: bool = True
    exposure: Optional[float] = None  # milisegundos o valor driver‑específico
    autofocus: bool = True
    focus: Optional[float] = None


class OBSBotPTZ:
    """Control PTZ vía hotkeys de OBSBOT Center usando pyautogui.
    Requiere que el usuario haya configurado atajos globales en la app.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.hk_left = os.getenv("OBSBOT_LEFT", "ctrl+alt+left")
        self.hk_right = os.getenv("OBSBOT_RIGHT", "ctrl+alt+right")
        self.hk_up = os.getenv("OBSBOT_UP", "ctrl+alt+up")
        self.hk_down = os.getenv("OBSBOT_DOWN", "ctrl+alt+down")
        self.hk_patrol = os.getenv("OBSBOT_PATROL_TOGGLE", "ctrl+alt+p")
        try:
            import pyautogui  # type: ignore
            self._pg = pyautogui
            self.available = True
        except Exception:
            self._pg = None
            self.available = False

    @staticmethod
    def _press_combo(pg, combo: str) -> None:
        # combo como "ctrl+alt+left"
        keys = [k.strip() for k in combo.split("+") if k.strip()]
        if not keys:
            return
        if len(keys) == 1:
            pg.press(keys[0])
        else:
            pg.hotkey(*keys)

    def set_enabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def patrol_toggle(self) -> None:
        if not (self.enabled and self.available):
            return
        self._press_combo(self._pg, self.hk_patrol)

    def left(self) -> None:
        if self.enabled and self.available:
            self._press_combo(self._pg, self.hk_left)

    def right(self) -> None:
        if self.enabled and self.available:
            self._press_combo(self._pg, self.hk_right)

    def up(self) -> None:
        if self.enabled and self.available:
            self._press_combo(self._pg, self.hk_up)

    def down(self) -> None:
        if self.enabled and self.available:
            self._press_combo(self._pg, self.hk_down)


class CameraAgent:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=cfg.queue_size)
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._roi: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h en píxeles
        self._patrol_sw = False
        self._patrol_phase = 0.0
        self._detector_mode = "off"  # 'yolo' | 'hog' | 'off'
        self._yolo = None
        self._ptz = OBSBotPTZ()

    # ---------- Gestión cámara ----------
    def start(self) -> None:
        if self._cap is not None:
            return
        idx = self.cfg.device_index
        cap = cv2.VideoCapture(idx, self.cfg.backend if self.cfg.backend else 0)
        if not cap or not cap.isOpened():
            # intento sin backend explícito
            cap = cv2.VideoCapture(idx)
        if not cap or not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara index {idx}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # Exposición/enfoque (si el driver lo permite)
        if self.cfg.auto_exposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # valor típico en Windows
        if self.cfg.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(self.cfg.exposure))
        if self.cfg.autofocus:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        if self.cfg.focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, float(self.cfg.focus))

        self._cap = cap
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.5)
            self._t = None
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        # drenar cola
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except Exception:
                break

    def get_latest(self, timeout: float = 0.0) -> Optional[np.ndarray]:
        try:
            if timeout <= 0:
                return self._q.get_nowait()
            return self._q.get(timeout=timeout)
        except Exception:
            return None

    # ---------- Detectors ----------
    def set_detector(self, mode: str) -> None:
        mode = mode.lower()
        if mode not in ("yolo", "hog", "off"):
            mode = "off"
        if mode == "yolo" and self._yolo is None:
            try:
                from ultralytics import YOLO  # type: ignore
                self._yolo = YOLO("yolov8n.pt")
            except Exception:
                print("[camera_agent] YOLO no disponible, usando HOG.")
                mode = "hog"
        if mode == "hog":
            # inicializar HOG sólo una vez
            if not hasattr(self, "_hog"):
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self._hog = hog
        self._detector_mode = mode

    # ---------- Patrullaje ----------
    def toggle_patrol_sw(self) -> bool:
        self._patrol_sw = not self._patrol_sw
        if not self._patrol_sw:
            self._roi = None
        return self._patrol_sw

    def toggle_ptz(self) -> None:
        self._ptz.set_enabled(True)
        self._ptz.patrol_toggle()

    # ---------- Bucle de captura ----------
    def _loop(self) -> None:
        _set_thread_name("CameraAgentLoop")
        last_ts = 0.0
        while not self._stop.is_set():
            ret, frame = self._cap.read() if self._cap else (False, None)
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Patrulla por software (pan digital)
            if self._patrol_sw:
                frame = self._apply_digital_patrol(frame)

            # Detección opcional
            if self._detector_mode != "off":
                frame = self._annotate_with_detector(frame)

            if self.cfg.convert_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mantener cola pequeña (descartar frames viejos)
            if self._q.full():
                try:
                    self._q.get_nowait()
                except Exception:
                    pass
            self._q.put(frame, timeout=0.01)

        # salir

    # ---------- Lógica de patrullaje digital ----------
    def _apply_digital_patrol(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        # ROI = 70% del ancho/alto, barrido seno‑coseno
        roi_w = int(w * 0.70)
        roi_h = int(h * 0.70)
        # velocidad (ciclos por segundo)
        speed = 0.15
        self._patrol_phase = (self._patrol_phase + speed / max(self.cfg.fps, 1)) % 1.0
        # trayecto Lissajous simple
        x_center = (w - roi_w) // 2 + int(((w - roi_w) // 2) * math.sin(2 * math.pi * self._patrol_phase))
        y_center = (h - roi_h) // 2 + int(((h - roi_h) // 2) * math.cos(2 * math.pi * self._patrol_phase))
        x = max(0, min(w - roi_w, x_center))
        y = max(0, min(h - roi_h, y_center))
        self._roi = (x, y, roi_w, roi_h)
        return frame[y : y + roi_h, x : x + roi_w]

    # ---------- Detección + anotación ----------
    def _annotate_with_detector(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        largest_box = None

        if self._detector_mode == "yolo" and self._yolo is not None:
            try:
                res = self._yolo.predict(source=frame, verbose=False, imgsz=640)
                for r in res:
                    for b, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        cls = int(c)
                        if hasattr(self._yolo, "names"):
                            name = self._yolo.names.get(cls, str(cls))  # type: ignore
                        else:
                            name = str(cls)
                        if name in ("person", "0") and float(conf) >= 0.35:
                            x1, y1, x2, y2 = map(int, b.tolist())
                            area = (x2 - x1) * (y2 - y1)
                            if largest_box is None or area > largest_box[4]:
                                largest_box = (x1, y1, x2, y2, area)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"person {conf:.2f}", (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                                        lineType=cv2.LINE_AA)
            except Exception:
                pass
        elif self._detector_mode == "hog" and hasattr(self, "_hog"):
            try:
                rects, weights = self._hog.detectMultiScale(frame, winStride=(8, 8))
                for (x, y, rw, rh), wgt in zip(rects, weights):
                    if wgt < 0.3:
                        continue
                    area = rw * rh
                    if largest_box is None or area > largest_box[4]:
                        largest_box = (x, y, x + rw, y + rh, area)
                    cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 255), 2)
                    cv2.putText(frame, f"person {wgt:.2f}", (x, max(15, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                                lineType=cv2.LINE_AA)
            except Exception:
                pass

        # Si hay ROI por patrulla SW, recentrar suavemente hacia la detección
        if self._patrol_sw and largest_box is not None and self._roi is not None:
            x1, y1, x2, y2, _ = largest_box
            bx = (x1 + x2) // 2
            by = (y1 + y2) // 2
            rx, ry, rw, rh = self._roi
            # mover ROI un paso hacia el centro del bounding box
            target_x = max(0, min(w - rw, bx - rw // 2))
            target_y = max(0, min(h - rh, by - rh // 2))
            alpha = 0.15  # suavizado
            new_x = int(rx * (1 - alpha) + target_x * alpha)
            new_y = int(ry * (1 - alpha) + target_y * alpha)
            self._roi = (new_x, new_y, rw, rh)
            frame = frame[new_y : new_y + rh, new_x : new_x + rw]

        return frame


# ========= CLI para prueba manual =========

def _list_devices(max_idx: int = 10) -> None:
    print("Buscando cámaras... (0..{0})".format(max_idx))
    for i in range(max_idx + 1):
        cap = cv2.VideoCapture(i, _CAP_BACKEND if _CAP_BACKEND else 0)
        ok = cap.isOpened()
        if ok:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[{i}] OK  {w}x{h} @ {fps:.0f}fps")
        else:
            print(f"[{i}] —")
        try:
            cap.release()
        except Exception:
            pass


def _run_standalone(args: List[str]) -> int:
    import argparse

    p = argparse.ArgumentParser("camera_agent")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--res", type=str, default="1280x720")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--show", action="store_true")
    p.add_argument("--list", action="store_true")
    ns = p.parse_args(args)

    if ns.list:
        _list_devices(12)
        return 0

    try:
        w, h = map(int, ns.res.lower().split("x"))
    except Exception:
        w, h = 1280, 720

    cfg = CameraConfig(device_index=ns.device, width=w, height=h, fps=ns.fps)
    agent = CameraAgent(cfg)
    agent.start()
    agent.set_detector("off")

    win = None
    if ns.show:
        win = "Totem – CameraAgent"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, w, h)

    print("Controles: q salir | p patrulla SW | g detector (YOLO→HOG→OFF) | o patrulla PTZ | ← → ↑ ↓ PTZ")

    # Intentar habilitar PTZ si hay pyautogui
    if agent._ptz.available:
        agent._ptz.set_enabled(True)

    detector_cycle = ["yolo", "hog", "off"]
    det_idx = 2  # 'off'

    while True:
        frame = agent.get_latest(timeout=0.5)
        if frame is None:
            continue
        if win:
            cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            on = agent.toggle_patrol_sw()
            print(f"Patrulla SW: {'ON' if on else 'OFF'}")
        elif key == ord("g"):
            det_idx = (det_idx + 1) % len(detector_cycle)
            agent.set_detector(detector_cycle[det_idx])
            print(f"Detector: {agent._detector_mode}")
        elif key == ord("o"):
            agent.toggle_ptz()
            print("PTZ: toggle patrol (hotkey)")
        elif key in (81, 2424832):  # left arrow
            agent._ptz.left()
        elif key in (83, 2555904):  # right arrow
            agent._ptz.right()
        elif key in (82, 2490368):  # up arrow
            agent._ptz.up()
        elif key in (84, 2621440):  # down arrow
            agent._ptz.down()

    agent.stop()
    if win:
        cv2.destroyWindow(win)
    return 0


if __name__ == "__main__":
    sys.exit(_run_standalone(sys.argv[1:]))
