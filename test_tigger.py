# -*- coding: utf-8 -*-
"""
test_tigger.py
Prueba de conversación con el backend Totem Evolución IA3.

Flujo:
1) Llama a POST /session/start para abrir una sesión nueva.
2) Hace POST /chat/turn con lo que escribas en consola.
3) Muestra la respuesta de IA3, los slots capturados y el audio generado.
"""

import requests

BASE_URL = "http://127.0.0.1:8000"


def main():
    print("🚀 Probando Totem Evolución IA3...\n")

    # ------------------------------------------------------------------
    # 1) Crear sesión en /session/start
    # ------------------------------------------------------------------
    try:
        resp = requests.post(f"{BASE_URL}/session/start", json={})
    except Exception as e:
        print("❌ No se pudo conectar con el backend Totem.")
        print("   Asegúrate de tener corriendo en otra ventana:")
        print("   uvicorn amain:app --host 127.0.0.1 --port 8000 --reload")
        print(f"   Detalle técnico: {e}")
        return

    if resp.status_code != 200:
        print("❌ Error en /session/start:", resp.status_code, resp.text)
        return

    data = resp.json()
    session_id = data.get("session_id")
    campos_requeridos = data.get("campos_requeridos", {})

    print(f"✅ Sesión creada: {session_id}\n")
    print("Campos requeridos para la propuesta:")
    for k, v in campos_requeridos.items():
        print(f"  - {k}: {v}")
    print("\nEscribe algo para comenzar la conversación.")
    print("Deja la línea vacía y presiona ENTER para salir.\n")

    # ------------------------------------------------------------------
    # 2) Loop conversacional con /chat/turn
    # ------------------------------------------------------------------
    while True:
        user_text = input("Tú: ").strip()
        if not user_text:
            print("Saliendo del test...")
            break

        payload = {
            "session_id": session_id,
            "user_text": user_text,
        }

        try:
            r = requests.post(f"{BASE_URL}/chat/turn", json=payload)
        except Exception as e:
            print("❌ Error al llamar /chat/turn:", e)
            continue

        if r.status_code != 200:
            print("❌ Respuesta no OK de /chat/turn:", r.status_code, r.text)
            continue

        res = r.json()

        texto_respuesta = res.get("texto_respuesta")
        slots = res.get("slots", {})
        campos_pendientes = res.get("campos_pendientes", [])
        campos_completos = res.get("campos_completos", False)
        audio_path = res.get("audio_path")

        print("\nIA3:", texto_respuesta)
        print("\nSlots llenos hasta ahora:")
        if slots:
            for k, v in slots.items():
                if v:
                    print(f"  - {k}: {v}")
        else:
            print("  (sin datos todavía)")

        print("\nCampos pendientes:", campos_pendientes)
        print("¿Campos completos?:", campos_completos)
        print("Audio generado:", audio_path)
        print("-" * 60)


if __name__ == "__main__":
    main()
