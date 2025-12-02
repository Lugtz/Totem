# -*- coding: utf-8 -*-
"""
Prueba de creación de Lead directo en Zoho CRM.
"""

from core.crm_client import crear_lead_en_crm


def main():
    # Datos de prueba (puedes cambiarlos)
    slots = {
        "Name": "Hiram",
        "Company": "Empresa Demo Totem",
        "Email": "totem.prueba+001@evolucioni3.com",
        "Phone": "5551234567",

        "OBJETIVO": "Implementar Totem Inteligente IA3 en recepción (prueba directa Python)",
    }

    resp = crear_lead_en_crm(slots)
    print("Respuesta final CRM:", resp)


if __name__ == "__main__":
    main()
