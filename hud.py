# -*- coding: utf-8 -*-
"""
hud_panel_window.py — HUD del Totem en una ventana BAJA (solo la parte negra).

- Dibuja el panel estilo "CAPTURA DE DATOS EVOLUCIÓN I3"
  para usarse como la franja inferior del tótem.
- Calcula el ancho/alto aproximado usando el tamaño físico del tótem:
    Alto total: 79.4 cm
    Ancho:      34 cm
    Alto zona negra: ~25 cm (ajustable)

- Cada 1s hace GET a http://127.0.0.1:8000/dashboard
  y actualiza campos + barrita según el JSON.

Requisitos:
    pip install requests
"""

import tkinter as tk
import requests
from typing import Dict, Any

API_BASE = "http://127.0.0.1:8000"

# ------------------ CONFIG DE TÓTEM (puedes afinar aquí) ------------------

TOTEM_WIDTH_CM = 34.0       # ancho real del tótem
TOTEM_BOTTOM_CM = 25.0      # alto aproximado de la zona negra inferior
SCALE_PX_PER_CM = 20        # escala general en pixeles

WINDOW_WIDTH = int(TOTEM_WIDTH_CM * SCALE_PX_PER_CM)
WINDOW_HEIGHT = int(TOTEM_BOTTOM_CM * SCALE_PX_PER_CM)

# --------------------------------------------------------------------------

CAMPO_ORDER = ["nombre", "empresa", "email", "telefono", "solucion"]

LABELS = {
    "nombre": "NOMBRE:",
    "empresa": "EMPRESA:",
    "email": "EMAIL:",
    "telefono": "TELÉFONO:",
    "solucion": "SOLUCIÓN:",
}


class HudWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Totem – HUD Captura de Datos")
        root.configure(bg="#000000")

        # Ventana = solo la parte negra (franja inferior)
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        root.geometry(f"{self.width}x{self.height}")
        root.resizable(False, False)

        self.canvas = tk.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#000000",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.field_text_ids: Dict[str, int] = {}
        self.progress_fill_id: int | None = None
        self.status_id: int | None = None
        self.bar_left: float = 0.0
        self.bar_right: float = 0.0

        self._draw_static()
        self.poll_dashboard()

    def _draw_static(self):
        c = self.canvas
        w, h = self.width, self.height

        # Marco principal
        margin = 20
        c.create_rectangle(
            margin,
            margin,
            w - margin,
            h - margin,
            outline="#31f0e3",
            width=2,
        )

        # Esquinas naranjas
        corner_len = 28
        offset = 16
        # top-left
        c.create_line(
            margin + offset,
            margin + 8,
            margin + offset + corner_len,
            margin + 8,
            fill="#ff7c00",
            width=3,
        )
        # top-right
        c.create_line(
            w - margin - offset - corner_len,
            margin + 8,
            w - margin - offset,
            margin + 8,
            fill="#ff7c00",
            width=3,
        )
        # bottom-left
        c.create_line(
            margin + offset,
            h - margin - 8,
            margin + offset + corner_len,
            h - margin - 8,
            fill="#ff7c00",
            width=3,
        )
        # bottom-right
        c.create_line(
            w - margin - offset - corner_len,
            h - margin - 8,
            w - margin - offset,
            h - margin - 8,
            fill="#ff7c00",
            width=3,
        )

        # Título más compacto, centrado
        title_y = margin + 26
        c.create_text(
            w / 2,
            title_y,
            text="CAPTURA DE DATOS",
            fill="#42f5e8",
            font=("Consolas", 16, "bold"),
        )
        c.create_text(
            w / 2,
            title_y + 22,
            text="EVOLUCIÓN I3",
            fill="#42f5e8",
            font=("Consolas", 16, "bold"),
        )

        # Caja interna de datos
        inner_top = title_y + 36
        inner_bottom = h - margin - 60
        inner_margin_x = margin + 22
        c.create_rectangle(
            inner_margin_x,
            inner_top,
            w - inner_margin_x,
            inner_bottom,
            outline="#31f0e3",
            width=1,
        )

        # Filas (apretadas para que quepan las 5)
        row_start_y = inner_top + 24
        row_spacing = 28
        label_x = inner_margin_x + 20
        value_x = label_x + 170

        for idx, campo in enumerate(CAMPO_ORDER):
            y = row_start_y + idx * row_spacing
            label = LABELS[campo]
            c.create_text(
                label_x,
                y,
                anchor="w",
                text=label,
                fill="#7cfef3",
                font=("Consolas", 12, "bold"),
            )
            text_id = c.create_text(
                value_x,
                y,
                anchor="w",
                text="-----" if campo != "solucion" else "---",
                fill="#e9ffff",
                font=("Consolas", 12),
            )
            self.field_text_ids[campo] = text_id
            # subrayado
            c.create_line(
                value_x,
                y + 7,
                w - inner_margin_x - 18,
                y + 7,
                fill="#31f0e3",
                width=1,
            )

        # STATUS + barra abajo de todo
        status_y = inner_bottom + 22
        self.status_id = c.create_text(
            margin + 40,
            status_y,
            anchor="w",
            text="STATUS: CAPTURANDO DATOS…",
            fill="#42f5e8",
            font=("Consolas", 11, "bold"),
        )

        bar_y = status_y + 16
        self.bar_left = margin + 40
        self.bar_right = w - margin - 50
        bar_height = 8

        # Barra de fondo
        c.create_rectangle(
            self.bar_left,
            bar_y,
            self.bar_right,
            bar_y + bar_height,
            outline="#000000",
            fill="#222222",
        )

        # Barra de relleno (inicia en 0)
        self.progress_fill_id = c.create_rectangle(
            self.bar_left,
            bar_y,
            self.bar_left,
            bar_y + bar_height,
            outline="",
            fill="#ff7c00",
        )

    def poll_dashboard(self):
        """Consulta /dashboard cada 1s y actualiza la vista."""
        try:
            resp = requests.get(f"{API_BASE}/dashboard", timeout=2)
            if resp.ok:
                data = resp.json()
                self.update_from_dashboard(data)
        except Exception:
            if self.status_id is not None:
                self.canvas.itemconfigure(
                    self.status_id,
                    text="STATUS: SIN CONEXIÓN AL SERVIDOR",
                )
        # volver a preguntar en 1 segundo
        self.root.after(1000, self.poll_dashboard)

    def update_from_dashboard(self, data: Dict[str, Any]):
        fields = data.get("fields", [])
        filled_count = data.get("filled_count", 0)
        total = len(fields) if fields else len(CAMPO_ORDER)

        # Actualizar textos de cada campo
        for f in fields:
            campo_id = f.get("id")
            if campo_id in self.field_text_ids:
                value = f.get("value") or ""
                filled = bool(f.get("filled"))
                if filled and value.strip():
                    text = value.strip()
                else:
                    text = "---" if campo_id == "solucion" else "-----"
                self.canvas.itemconfigure(self.field_text_ids[campo_id], text=text)

        # Status
        if self.status_id is not None:
            if total > 0 and filled_count >= total:
                txt = "STATUS: DATOS COMPLETOS"
            else:
                txt = "STATUS: CAPTURANDO DATOS…"
            self.canvas.itemconfigure(self.status_id, text=txt)

        # Barrita
        if self.progress_fill_id is not None and total > 0:
            pct = max(0.0, min(1.0, filled_count / total))
            new_x2 = self.bar_left + (self.bar_right - self.bar_left) * pct
            x1, y1, _, y2 = self.canvas.coords(self.progress_fill_id)
            self.canvas.coords(self.progress_fill_id, self.bar_left, y1, new_x2, y2)


if __name__ == "__main__":
    root = tk.Tk()
    HudWindow(root)
    root.mainloop()