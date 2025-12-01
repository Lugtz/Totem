# -*- coding: utf-8 -*-
"""
hud_panel_window.py — HUD del Totem en ventana REDIMENSIONABLE (sin trabarse).

- Dibuja el panel estilo "CAPTURA DE DATOS EVOLUCIÓN I3".
- Cada 1s hace GET a http://127.0.0.1:8000/dashboard.
- Puedes redimensionar la ventana con el mouse y el HUD se vuelve a dibujar
  con el nuevo tamaño (sin escalado pesado).

Requisitos:
    pip install requests
"""

import tkinter as tk
import requests
from typing import Dict, Any, Optional

API_BASE = "http://127.0.0.1:8000"

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

        # Tamaño inicial (luego tú lo jalas con el mouse)
        self.width = 700
        self.height = 330
        root.geometry(f"{self.width}x{self.height}")
        root.resizable(True, True)

        self.canvas = tk.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#000000",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Estado interno
        self.field_text_ids: Dict[str, int] = {}
        self.progress_fill_id: Optional[int] = None
        self.status_id: Optional[int] = None
        self.bar_left: float = 0.0
        self.bar_right: float = 0.0
        self._resize_job: Optional[str] = None
        self._last_dashboard: Optional[Dict[str, Any]] = None

        # Dibujamos por primera vez
        self._draw_static(self.width, self.height)

        # Redibujar cuando cambie tamaño (con un pequeño debounce)
        self.canvas.bind("<Configure>", self._on_resize)

        # Empieza el polling del dashboard
        self.poll_dashboard()

    # -------------------- DIBUJO ESTÁTICO --------------------

    def _draw_static(self, w: int, h: int):
        """Dibuja el HUD completo para un tamaño dado (w, h)."""
        c = self.canvas
        c.delete("all")  # limpiamos todo

        # Reiniciar estructuras
        self.field_text_ids = {}
        self.progress_fill_id = None
        self.status_id = None

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

        # Título
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

        # Filas de campos (compacto)
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

        # STATUS + barra
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

        # Barra de relleno (arranca en 0)
        self.progress_fill_id = c.create_rectangle(
            self.bar_left,
            bar_y,
            self.bar_left,
            bar_y + bar_height,
            outline="",
            fill="#ff7c00",
        )

        # Si ya teníamos datos del dashboard, los volvemos a aplicar
        if self._last_dashboard is not None:
            self.update_from_dashboard(self._last_dashboard)

    # -------------------- REDIMENSIONADO --------------------

    def _on_resize(self, event):
        """Cuando cambie el tamaño, re-dibujamos con debounce."""
        new_w = max(event.width, 400)   # mínimo ancho razonable
        new_h = max(event.height, 220)  # mínimo alto razonable
        self.width = new_w
        self.height = new_h

        # evitamos redibujar en cada pixel (debounce)
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)

        self._resize_job = self.root.after(80, self._apply_resize)

    def _apply_resize(self):
        self._resize_job = None
        self.canvas.config(width=self.width, height=self.height)
        self._draw_static(self.width, self.height)

    # ------------------- LÓGICA DEL DASHBOARD -------------------

    def poll_dashboard(self):
        """Consulta /dashboard cada 1s y actualiza la vista."""
        try:
            resp = requests.get(f"{API_BASE}/dashboard", timeout=2)
            if resp.ok:
                data = resp.json()
                self._last_dashboard = data
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