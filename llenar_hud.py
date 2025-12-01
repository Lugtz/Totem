"""
llenar_hud.py — Script de prueba para ir llenando el HUD paso a paso
mandando JSON al endpoint /dashboard del servidor de prueba.
 
Asegúrate de tener corriendo:
    python test1.py
antes de ejecutar este script.
"""
 
import time
import requests
 
BASE = "http://127.0.0.1:8000"
 
steps = [
    {"nombre": "Luis Pérez"},
    {"empresa": "TKL"},
    {"email": "luis@tkl.com"},
    {"telefono": "555-123-4567"},
    {"solucion": "Ecosistema Zoho Evolución i3"},
]


def enviar_paso(num, payload):
    print(f"\nPaso {num}: mandando -> {payload}")
    try:
        r = requests.post(f"{BASE}/dashboard", json=payload, timeout=5)
        print("Status:", r.status_code)
        try:
            print("JSON:", r.json())
        except Exception:
            print("Respuesta:", r.text)
    except Exception as e:
        print("ERROR al conectar:", repr(e))


if __name__== "__main__":
    print("Iniciando prueba de llenado del HUD...")
    for i, data in enumerate(steps, start=1):
        enviar_paso(i, data)
        time.sleep(2.0)
    print("\nPrueba terminada.")
