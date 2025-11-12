# obsbot_patrol.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class PatrolConfig:
    out_w: int = 1920           # tamaño de salida (ancho del recorte)
    out_h: int = 1080           # tamaño de salida (alto del recorte)
    idle_seconds_to_patrol: float = 2.0  # esperar sin personas para patrullar
    pan_speed_px_per_sec: int = 600      # velocidad de paneo digital (px/seg)
    soften_centering: float = 0.15       # 0..1 suavizado al centrar a la persona
    margin_keep_person_px: int = 120     # margen para no recentrar si ya está en cuadro
    ping_pong: bool = True               # patrullaje ida y vuelta

class DigitalPanner:
    """
    Paneo digital: mueve una ventana (out_w x out_h) dentro del frame completo
    sin mover físicamente la cámara. Si luego conectamos PTZ real, solo
    reemplazamos por otra clase con la misma interfaz.
    """
    def __init__(self, frame_w: int, frame_h: int, cfg: PatrolConfig):
        self.fw = frame_w
        self.fh = frame_h
        self.cfg = cfg
        self.x = max(0, (self.fw - self.cfg.out_w) // 2)
        self.y = max(0, (self.fh - self.cfg.out_h) // 2)
        self._dir = 1   # 1 -> derecha, -1 -> izquierda
        self._last = time.time()

    def crop_coords(self):
        x0 = max(0, min(self.x, self.fw - self.cfg.out_w))
        y0 = max(0, min(self.y, self.fh - self.cfg.out_h))
        return x0, y0, x0 + self.cfg.out_w, y0 + self.cfg.out_h

    def step_patrol(self):
        now = time.time()
        dt = now - self._last
        self._last = now
        step = int(self.cfg.pan_speed_px_per_sec * dt) * self._dir
        self.x += step

        # Rebotes a los bordes
        if self.x <= 0:
            self.x = 0
            if self.cfg.ping_pong:
                self._dir = 1
        if self.x >= self.fw - self.cfg.out_w:
            self.x = self.fw - self.cfg.out_w
            if self.cfg.ping_pong:
                self._dir = -1

    def center_on(self, target_x: int):
        """Centra suavemente el recorte sobre un x absoluto de la persona."""
        desired_x = max(0, min(target_x - self.cfg.out_w // 2, self.fw - self.cfg.out_w))
        delta = desired_x - self.x
        self.x += int(delta * self.cfg.soften_centering)

    def ensure_person_inside(self, bbox):
        """Si ya está dentro con margen, no muevas; si sale, haz 'center_on'."""
        (x1, y1, x2, y2) = bbox
        x0, _, x3, _ = self.crop_coords()

        # Si caja visible con margen, no re-centrar agresivo
        if x1 >= x0 + self.cfg.margin_keep_person_px and x2 <= x3 - self.cfg.margin_keep_person_px:
            return
        # Si no, centramos al centroide
        cx = (x1 + x2) // 2
        self.center_on(cx)

# ---- Stub para PTZ físico (OBSBOT SDK / OSC) ----
class PTZStub:
    """
    Interfaz para control PTZ real. Más adelante, con el SDK de OBSBOT
    u OSC bridge, reemplaza 'center_on' y 'step_patrol' por comandos
    de gimbal. La API se mantiene para no tocar el resto.
    """
    def __init__(self, frame_w: int, frame_h: int, cfg: PatrolConfig):
        self.fw = frame_w
        self.fh = frame_h
        self.cfg = cfg

    def crop_coords(self):
        return 0, 0, self.fw, self.fh

    def step_patrol(self):
        pass

    def center_on(self, target_x: int):
        pass

    def ensure_person_inside(self, bbox):
        pass
