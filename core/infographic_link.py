# -*- coding: utf-8 -*-
"""
core.infographic_link

Obtiene automáticamente la URL pública de la infografía usando la API
local de ngrok (http://127.0.0.1:4040/api/tunnels).

Requisitos:
- ngrok debe estar corriendo, por ejemplo:
    ngrok http 9000
- Servidor HTTP sirviendo la carpeta 'infografias' en el puerto 9000, por ejemplo:
    python -m http.server 9000 -d infografias
"""

import os
import requests

# Endpoint local de ngrok para consultar los túneles activos
NGROK_API_URL = "http://127.0.0.1:4040/api/tunnels"

# Nombre del archivo de la infografía (el mismo que usas en generar_infografia)
INFOGRAFIA_FILENAME = os.getenv("INFOGRAFIA_FILENAME", "infografia_totem.png")


def _get_ngrok_https_base_url() -> str:
    """
    Lee la API local de ngrok y devuelve el primer public_url HTTPS.
    Ejemplo: https://milo-multicentral-pulvinately.ngrok-free.dev
    """
    try:
        resp = requests.get(NGROK_API_URL, timeout=2)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(
            f"No pude consultar la API de ngrok en {NGROK_API_URL}. "
            f"¿Seguro que ngrok está corriendo? Detalle: {e}"
        )

    tunnels = data.get("tunnels", [])
    for t in tunnels:
        public_url = t.get("public_url", "")
        if public_url.startswith("https://"):
            # Quitamos cualquier "/" al final por orden
            return public_url.rstrip("/")

    raise RuntimeError(
        "No encontré ningún túnel HTTPS en ngrok. "
        "Arráncalo con:  ngrok http 9000"
    )


def build_infografia_public_url() -> str:
    """
    Devuelve la URL completa de la infografía, por ejemplo:
        https://XXXX.ngrok-free.dev/infografia_totem.png

    OJO: esto asume que el server HTTP se levantó así:
        python -m http.server 9000 -d infografias
    de modo que la raíz del sitio YA ES la carpeta 'infografias'.
    """
    base = _get_ngrok_https_base_url()
    return f"{base}/{INFOGRAFIA_FILENAME}"
