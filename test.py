# -*- coding: utf-8 -*-
"""
test.py — Servidor de prueba para el HUD del Totem (HTML).

Requisitos:
    pip install fastapi uvicorn pydantic

Uso:
    python test.py

Endpoints:
    GET  /            -> HTML del panel HUD
    GET  /dashboard   -> Estado actual (JSON)
    POST /dashboard   -> Actualiza campos (JSON)
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

app = FastAPI()

HUD_HTML = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Totem – Captura de Datos</title>
  <style>
    /* FONDO GENERAL (difuminado entre colores Ei3) */
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      font-family: "Consolas", "Courier New", monospace;
      background: radial-gradient(circle at top left, #015666, #009999 70%, #01242b 100%);
      color: #dfffff;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .hud-screen {
      width: 100%;
      max-width: 700px;
      aspect-ratio: 3 / 4; /* similar proporción a la imagen */
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      box-sizing: border-box;
    }

    /* MARCO PRINCIPAL */
    .hud-panel {
      position: relative;
      width: 100%;
      height: 100%;
      border-radius: 24px;
      border: 1px solid #31f0e3; /* cian brillante */
      box-shadow: 0 0 45px rgba(0, 0, 0, 0.75);
      box-sizing: border-box;
      padding: 26px 30px 22px;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      background: rgba(0, 20, 24, 0.35);
    }

    /* ESQUINAS NARANJA */
    .corner {
      position: absolute;
      width: 20px;
      height: 3px;
      background: #ff7c00;
    }
    .corner.tl { top: 14px; left: 22px; transform: rotate(25deg); }
    .corner.tr { top: 14px; right: 22px; transform: rotate(-25deg); }
    .corner.bl { bottom: 14px; left: 22px; transform: rotate(-25deg); }
    .corner.br { bottom: 14px; right: 22px; transform: rotate(25deg); }

    /* TÍTULO */
    .hud-title {
      text-align: center;
      font-size: 24px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: #42f5e8;
      margin-bottom: 26px;
    }
    .hud-title span {
      display: block;
    }

    /* CAJA INTERNA DE DATOS (como en la imagen) */
    .hud-inner {
      border: 1px solid rgba(49, 240, 227, 0.8);
      border-radius: 8px;
      padding: 20px 24px 18px;
      box-sizing: border-box;
      margin-bottom: 26px;
    }

    .hud-row {
      display: flex;
      align-items: baseline;
      gap: 10px;
      font-size: 18px;
      color: #7cfef3;
      margin-bottom: 14px;
    }

    .hud-label {
      min-width: 155px;
      letter-spacing: 0.20em;
    }

    .hud-value {
      flex: 1;
      color: #e9ffff;
      padding-bottom: 4px;
      border-bottom: 1px solid rgba(49, 240, 227, 0.65);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .hud-value.empty {
      color: rgba(124, 254, 243, 0.55);
      border-bottom-color: rgba(30, 110, 110, 0.8);
    }

    /* STATUS + BARRA */
    .hud-status {
      font-size: 16px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: #42f5e8;
      margin-bottom: 10px;
    }

    .hud-progress-bar {
      width: 70%;
      height: 8px;
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.55);
      overflow: hidden;
      box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.9);
    }

    .hud-progress-fill {
      width: 50%;
      height: 100%;
      background: linear-gradient(90deg, #ff7c00, #ffba66);
      transition: width 0.3s ease-out;
    }
  </style>
</head>
<body>
  <div class="hud-screen">
    <div class="hud-panel">
      <div class="corner tl"></div>
      <div class="corner tr"></div>
      <div class="corner bl"></div>
      <div class="corner br"></div>

      <div class="hud-title">
        <span>CAPTURA DE DATOS</span>
        <span>EVOLUCIÓN I3</span>
      </div>

      <div class="hud-inner">
        <div class="hud-row">
          <div class="hud-label">NOMBRE:</div>
          <div class="hud-value empty" id="v-nombre">-----</div>
        </div>
        <div class="hud-row">
          <div class="hud-label">EMPRESA:</div>
          <div class="hud-value empty" id="v-empresa">-----</div>
        </div>
        <div class="hud-row">
          <div class="hud-label">EMAIL:</div>
          <div class="hud-value empty" id="v-email">-----</div>
        </div>
        <div class="hud-row">
          <div class="hud-label">TELÉFONO:</div>
          <div class="hud-value empty" id="v-telefono">-----</div>
        </div>
        <div class="hud-row" style="margin-bottom: 0;">
          <div class="hud-label">SOLUCIÓN:</div>
          <div class="hud-value empty" id="v-solucion">---</div>
        </div>
      </div>

      <div>
        <div class="hud-status" id="hud-status">
          STATUS: CAPTURANDO DATOS…
        </div>
        <div class="hud-progress-bar">
          <div class="hud-progress-fill" id="progress-fill"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const mapIds = {
      nombre:   "v-nombre",
      empresa:  "v-empresa",
      email:    "v-email",
      telefono: "v-telefono",
      solucion: "v-solucion",
    };

    const progressFill = document.getElementById("progress-fill");
    const statusEl = document.getElementById("hud-status");

    function applyField(id, value, filled) {
      const el = document.getElementById(mapIds[id]);
      if (!el) return;

      if (filled && value && value.trim() !== "") {
        el.textContent = value;
        el.classList.remove("empty");
      } else {
        el.textContent = (id === "solucion") ? "---" : "-----";
        el.classList.add("empty");
      }
    }

    function updateFromDashboard(data) {
      const fields = data.fields || [];
      const total = fields.length || 0;
      const filledCount = data.filled_count || 0;

      fields.forEach(f => {
        if (mapIds[f.id]) {
          applyField(f.id, f.value || "", f.filled);
        }
      });

      const pct = total > 0 ? (filledCount / total) * 100 : 0;
      progressFill.style.width = pct + "%";

      if (filledCount === total && total > 0) {
        statusEl.textContent = "STATUS: DATOS COMPLETOS";
      } else {
        statusEl.textContent = "STATUS: CAPTURANDO DATOS…";
      }
    }

    async function fetchDashboard() {
      try {
        const res = await fetch("/dashboard");
        if (!res.ok) return;
        const data = await res.json();
        updateFromDashboard(data);
      } catch (err) {
        console.error("Error obteniendo dashboard", err);
      }
    }

    // Primer fetch y luego refresco cada 1 segundo
    fetchDashboard();
    setInterval(fetchDashboard, 1000);
  </script>
</body>
</html>"""

# ------------------ ESTADO EN MEMORIA ------------------


class DashboardUpdate(BaseModel):
    nombre: Optional[str] = None
    empresa: Optional[str] = None
    email: Optional[str] = None
    telefono: Optional[str] = None
    solucion: Optional[str] = None


# valores actuales (simulan lo que luego vendrá de los slots del Totem)
CURRENT_VALUES: Dict[str, Optional[str]] = {
    "nombre": None,
    "empresa": None,
    "email": None,
    "telefono": None,
    "solucion": None,
}


def build_dashboard_state() -> Dict[str, Any]:
    fields = []
    filled = 0
    for field_id in ["nombre", "empresa", "email", "telefono", "solucion"]:
        value = CURRENT_VALUES.get(field_id)
        is_filled = bool(value and value.strip())
        if is_filled:
            filled += 1

        label = "SOLUCIÓN" if field_id == "solucion" else field_id.upper()
        fields.append(
            {
                "id": field_id,
                "label": label,
                "value": value if is_filled else None,
                "filled": is_filled,
            }
        )

    missing = [f["id"] for f in fields if not f["filled"]]

    return {
        "session_id": "demo",
        "fields": fields,
        "filled_count": filled,
        "missing_count": len(missing),
        "missing": missing,
    }


# ------------------ ENDPOINTS ------------------


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HUD_HTML)


@app.get("/dashboard", response_class=JSONResponse)
async def get_dashboard():
    return JSONResponse(content=build_dashboard_state())


@app.post("/dashboard", response_class=JSONResponse)
async def update_dashboard(payload: DashboardUpdate):
    data = payload.dict()
    for key, value in data.items():
        if key in CURRENT_VALUES and value is not None:
            txt = value.strip()
            CURRENT_VALUES[key] = txt or None
    return JSONResponse(content=build_dashboard_state())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)