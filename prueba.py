# llenar_crm_test.py
# Prueba rápida para llenar el panel CRM del Totem vía HTTP.

import time
import requests

BASE_URL = "http://127.0.0.1:7000/crm"

ejemplos = [
    {
        "nombre": "Luis Pérez",
        "empresa": "TKL Logistics",
        "correo": "luis.perez@tkl.com",
        "telefono": "555-123-4567",
        "diagnostico": "Necesita visibilidad en tiempo real de embarques y alertas automáticas."
    },
    {
        "nombre": "María Gómez",
        "empresa": "Enmedio",
        "correo": "maria.gomez@enmedio.com",
        "telefono": "555-987-6543",
        "diagnostico": "Proyecto CRM omnicanal con automatización de oportunidades y tableros ejecutivos."
    },
    {
        "nombre": "Juan López",
        "empresa": "Evolución i3",
        "correo": "juan.lopez@evolucioni3.com",
        "telefono": "555-111-2233",
        "diagnostico": "Demostración interna de Totem IA3 con generación de propuestas e infografías."
    },
]

def main():
    print("⚙️  Enviando datos de prueba al panel CRM...")
    for i, payload in enumerate(ejemplos, start=1):
        try:
            resp = requests.get(BASE_URL, params=payload, timeout=5)
            print(f"\nCaso #{i}")
            print("URL :", resp.url)
            print("HTTP:", resp.status_code)
            print("Body:", resp.text[:200])
        except Exception as e:
            print(f"❌ Error al conectar: {e}")
        time.sleep(2.0)

if __name__ == "__main__":
    main()
