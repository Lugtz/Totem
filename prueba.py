"""
test_infografia_flujo.py
 
Prueba el FLUJO COMPLETO:
1) Genera texto de infografía con OpenAI.
2) Genera PNG+PDF.
3) Saca URL pública (ngrok o lo que uses).
4) Envía WhatsApp (si enviar_whatsapp=True).
5) Envía Email (si enviar_email_flag=True).
"""
 
from __future__ import annotations
 
from typing import Any, Dict
 
from core.whatsapp_email_infografia import flujo_infografia_whatsapp_email
 
 
def main() -> None:
    # Datos de prueba similares a los que mandará amain
    datos_cliente: Dict[str, Any] = {
        "nombre": "Visitante Zoholics Test",
        "empresa": "Empresa Demo Test",
        "OBJETIVO": "Quiere profesionalizar la operación comercial y financiera.",

    }
 
    # Teléfono y email de prueba
    telefono_prueba = None  # usa el de .env (WOZTELL_TEST_PHONE)
    email_prueba = None     # usa EMAIL_TO_TEST del .env
 
    print(">>> INICIANDO TEST flujo_infografia_whatsapp_email")
    print("datos_cliente =", datos_cliente)
 
    resultado = flujo_infografia_whatsapp_email(
        datos_cliente,
        telefono=telefono_prueba,
        email=email_prueba,
        nombre_archivo="infografia_flujo_test",
        enviar_whatsapp=True,      # pon False si solo quieres probar email
        enviar_email_flag=True,    # pon False si solo quieres probar Whats
    )
 
    print("\n>>> RESULTADO DEL FLUJO:")
    for k, v in resultado.items():
        print(f"  {k}: {v}")
 
 
if __name__ == "__main__":
    main()