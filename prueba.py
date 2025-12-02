# -*- coding: utf-8 -*-
"""
prueba.py

Prueba del flujo /session/start + /chat/turn de amain.py
para verificar que se llama crear_lead_en_crm(slots) igual
que en el script de prueba directo.

- NO usamos cámara/YOLO (quitamos startup/shutdown).
- NO llamamos a OpenAI (monkeypatch directo de amain.procesar_turno_dialogo).
- Simulamos un turno final de DESPEDIDA con datos completos.
"""

from fastapi.testclient import TestClient
import amain  # 👈 importante: importamos el módulo completo

# Tomamos la app desde amain
app = amain.app

# -----------------------------------------------------------
# 1) Desactivar eventos de startup/shutdown para evitar YOLO
# -----------------------------------------------------------
app.router.on_startup.clear()
app.router.on_shutdown.clear()

client = TestClient(app)


def main():
    # -------------------------------------------------------
    # 2) Creamos una sesión como hace la cámara
    # -------------------------------------------------------
    resp = client.post("/session/start", json={})
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]
    print("✅ Nueva sesión creada:", session_id)
    print("   Campos requeridos:", data.get("campos_requeridos"))

    # -------------------------------------------------------
    # 3) Monkeypatch de amain.procesar_turno_dialogo
    #    Para que regrese slots COMPLETOS + DESPEDIDA
    #    sin llamar a OpenAI.
    # -------------------------------------------------------
    original_fn = amain.procesar_turno_dialogo

    def fake_procesar_turno(session_id_in: str, texto_usuario: str):
        """
        Simula el resultado del motor de diálogo cuando:
        - Ya tenemos todos los campos obligatorios.
        - Este turno es de DESPEDIDA.
        """
        print(f"[FAKE_DIALOG_ENGINE] session_id={session_id_in}, texto={texto_usuario!r}")

        slots = {
            "nombre": "Hiram",
            "empresa": "Empresa Demo Totem",
            "correo": "totem.prueba+001@evolucioni3.com",
            "telefono": "5551234567",
            "diagnostico": "Implementar Totem Inteligente IA3 en recepción (prueba amain)",
        }
        campos_pendientes = []      # ya no falta nada
        campos_completos = True     # ✅ para que amain lo considere completo
        es_despedida = True         # ✅ para que dispare el CRM

        assistant_text = (
            "Perfecto Hiram, con todo lo que me compartiste ya puedo "
            "preparar la propuesta y registrar tus datos. Muchas gracias, "
            "que tengas un excelente día."
        )

        # OJO: amain._normalizar_respuesta_dialog_engine soporta
        # tupla de 5 elementos donde el 5º es es_despedida
        return assistant_text, slots, campos_pendientes, campos_completos, es_despedida

    # Aplicar monkeypatch DIRECTO SOBRE amain
    amain.procesar_turno_dialogo = fake_procesar_turno

    try:
        # ---------------------------------------------------
        # 4) Mandamos UN solo /chat/turn como si fuera turno final
        # ---------------------------------------------------
        payload = {
            "session_id": session_id,
            "texto_usuario": "Ok, muchas gracias, nos vemos.",
        }

        print("\n>> Llamando /chat/turn con payload:")
        print(payload)

        resp_turn = client.post("/chat/turn", json=payload)
        print("<< STATUS /chat/turn:", resp_turn.status_code)

        body = resp_turn.json()
        print("<< BODY /chat/turn:", body)

        print(
            "\nSi todo está bien, en la consola también deberías ver:\n"
            "- Logs de [get_access_token]\n"
            "- Logs de [crear_lead_en_crm] STATUS 201 (o el código que devuelva Zoho)\n"
            "- Mensaje de 'Lead creado en Zoho CRM correctamente.' desde amain\n"
        )

    finally:
        # ---------------------------------------------------
        # 5) Restaurar la función original del dialog_engine
        # ---------------------------------------------------
        amain.procesar_turno_dialogo = original_fn


if __name__ == "__main__":
    main()
