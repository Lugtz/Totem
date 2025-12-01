# -*- coding: utf-8 -*-
"""
core.crm_client

Crear leads en Zoho CRM directamente desde Python (Totem IA3).
"""

import requests
from typing import Dict, Any

# ========= RELLENA ESTO CON TUS DATOS =========
# ⚠️ OJO: aquí van TUS valores reales:
# - ZOHO_CLIENT_ID      -> el client id del Self Client
# - ZOHO_CLIENT_SECRET  -> el client secret del Self Client
# - ZOHO_REFRESH_TOKEN  -> el valor que salió en "refresh_token : 1000.xxxxx..."
#
# EJEMPLO (NO pegues este literal):
# ZOHO_CLIENT_ID = "1000.ABCDEF..."
# ZOHO_CLIENT_SECRET = "1234567890abcdef..."
# ZOHO_REFRESH_TOKEN = "1000.ZZZZZZ..."
#
ZOHO_CLIENT_ID = "1000.2SWW9OY9TRBSCLNNG8UZ2V4PJVQXWV"
ZOHO_CLIENT_SECRET = "ca65f4d47df77cb20ad17186115129a95f297899a2"
ZOHO_REFRESH_TOKEN = "1000.a786885733245bd99f523ce8a135cea6.94861116dea0f7421a39fa41eb124e35"
ZOHO_DC = "com"  # tu api_domain fue https://www.zohoapis.com -> "com"
# =============================================

ACCOUNTS_BASE = f"https://accounts.zoho.{ZOHO_DC}"
API_BASE = f"https://www.zohoapis.{ZOHO_DC}/crm/v2"


def get_access_token() -> str:
    """
    Usa el REFRESH TOKEN (NO el code) para obtener un access_token nuevo.
    """
    url = f"{ACCOUNTS_BASE}/oauth/v2/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "refresh_token": ZOHO_REFRESH_TOKEN,
    }

    resp = requests.post(url, data=data, timeout=15)

    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    print("[get_access_token] STATUS:", resp.status_code)
    print("[get_access_token] BODY  :", body)

    if resp.status_code != 200 or "access_token" not in body:
        raise RuntimeError(f"Error al obtener access_token: {body}")

    return body["access_token"]


def crear_lead_en_crm(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea un Lead en Zoho CRM usando los datos del dict `slots`.

    Ejemplo de `slots`:
    {
        "Name": "Carlos Demo",
        "Company": "Empresa Demo",
        "Email": "demo.lead@example.com",
        "Phone": "5551234567",
        "OBJETIVO": "Implementar Totem Inteligente IA3 en recepción",
    }

    Para la PRIMERA prueba usaremos solo campos estándar
    (nada de campos personalizados todavía).
    """
    access_token = get_access_token()

    url = f"{API_BASE}/Leads"
    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}",
        "Content-Type": "application/json",
    }

    # Campos mínimos estándar (API Name oficiales de Zoho):
    # - Last_Name  (OBLIGATORIO)
    # - Company
    # - Email
    # - Phone
    # - Lead_Source
    # - Description
    lead = {
        "Last_Name": slots.get("Name", "Visitante Totem"),
        "Company": slots.get("Company", "Visitante Totem"),
        "Email": slots.get("Email"),
        "Phone": slots.get("Phone"),
        "Lead_Source": "Totem IA3",
        "Description": slots.get("OBJETIVO"),
    }

    payload = {
        "data": [lead],
        "trigger": ["workflow", "blueprint"],  # dispara workflows si tienes
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    print("[crear_lead_en_crm] STATUS:", resp.status_code)
    print("[crear_lead_en_crm] BODY  :", resp.text)
    resp.raise_for_status()
    return resp.json()
