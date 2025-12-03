# -*- coding: utf-8 -*-
"""
prueba.py

Script de prueba RÁPIDO para validar _mapear_slots_para_crm()
sin levantar FastAPI ni tocar Zoho ni Woztell.
"""

from pprint import pprint

# Importamos desde amain.py (asegúrate de que amain.py está en esta misma carpeta)
from amain import _mapear_slots_para_crm  # type: ignore


def probar_caso(descripcion: str, slots: dict) -> None:
    print("\n" + "=" * 80)
    print(f"CASO: {descripcion}")
    print("- Slots de entrada:")
    pprint(slots)

    payload = _mapear_slots_para_crm(slots)

    print("\n-> Payload CRM generado:")
    pprint(payload)
    print("=" * 80 + "\n")


def main() -> None:
    # Caso 1: todos los campos “bonitos”
    slots_1 = {
        "nombre": "Irán",
        "empresa": "TKL",
        "correo": "hirm060220@gmail.com",
        "telefono": "449 277 92 68",
        "solucion_a_implementar": "implementar el CRM para manejar mejor los prospectos y automatizar procesos",
    }

    # Caso 2: sin teléfono, sin correo, solo objetivo
    slots_2 = {
        "nombre": "Lourdes",
        "empresa": "Evolución i3",
        "objetivo": "automatizar seguimiento de leads en Zoho CRM",
    }

    # Caso 3: viene OBJETIVO y diagnostico, pero no solucion_a_implementar
    slots_3 = {
        "Name": "Visitante de Expo",
        "Company": "Empresa Demo",
        "Email": "demo@empresa.com",
        "Phone": "555 555 5555",
        "OBJETIVO": "mejorar la experiencia de recepción",
        "diagnostico": "no cuentan con procesos claros de recepción ni registro",
    }

    # Caso 4: slots súper vacíos (debe poner Visitante Totem)
    slots_4 = {}

    probar_caso("1) Todos los campos completos (solucion_a_implementar)", slots_1)
    probar_caso("2) Sin teléfono ni correo (solo objetivo)", slots_2)
    probar_caso("3) Con OBJETIVO y diagnostico explícitos", slots_3)
    probar_caso("4) Slots vacíos (fallback Visitante Totem)", slots_4)


if __name__ == "__main__":
    main()
