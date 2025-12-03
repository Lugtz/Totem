# -*- coding: utf-8 -*-
"""
core.whatsapp_email_infografia

Flujo completo:
1) Llama a OpenAI (infografia_prompt_generator) para generar el TEXTO de la infografía.
2) Genera la infografía (PNG + PDF) con infographic_engine usando ese texto.
3) Obtiene la URL pública de la infografía usando ngrok (core.infographic_link).
4) Envía WhatsApp (TEXTO FIJO + IMAGEN) por Woztell Bot API.
5) Envía la infografía por correo (adjunto PNG) usando SMTP Zoho.

Se expone la función:
    flujo_infografia_whatsapp_email(...)
para que amain.py la use de forma SINCRONA al final de la conversación.
"""

from __future__ import annotations

import os
import json
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from core.infographic_engine import generar_infografia
from core.infographic_link import build_infografia_public_url
from core.infografia_prompt_generator import generar_texto_infografia

# =========================
# Cargar variables de entorno
# =========================
load_dotenv()

# -------------------------
# WhatsApp / Woztell
# -------------------------
WOZTELL_TOKEN = os.getenv("WOZTELL_TOKEN")
WOZTELL_CHANNEL_ID = os.getenv("WOZTELL_CHANNEL_ID")
# Teléfono de prueba (formato 52XXXXXXXXXX)
TELEFONO_DESTINO_DEFAULT = os.getenv("WOZTELL_TEST_PHONE", "5214495097519")

# -------------------------
# Email / SMTP Zoho
# -------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.zoho.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")  # ej. comercial@evolucioni3.com
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "True").lower() == "true"
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Evolución i3")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER or "")
EMAIL_DESTINO_DEFAULT = os.getenv(
    "EMAIL_TO_TEST",
    "valadezgutierrezmaguadalupe@gmail.com",
)

# ====================================================
# FUNCIÓN: WhatsApp (texto fijo Zoholics + imagen IA)
# ====================================================
def enviar_whatsapp_imagen(imagen_url: str, telefono: str) -> None:
    """
    Envía 2 respuestas por Woztell Bot API:
      1) Mensaje de TEXTO (copy Zoholics 2025)
      2) Mensaje de IMAGEN (URL pública de la infografía)
    """
    if not WOZTELL_TOKEN or not WOZTELL_CHANNEL_ID:
        raise RuntimeError("Faltan WOZTELL_TOKEN o WOZTELL_CHANNEL_ID en .env")

    # Endpoint Bot API que YA probamos que funciona
    url = f"https://bot.api.woztell.com/sendResponses?accessToken={WOZTELL_TOKEN}"

    texto = (
        "Hola, soy el asistente inteligente del stand de Evolución i3 en Zoholics 2025.\n"
        "Acabo de generar tu propuesta personalizada con base en lo que analizamos en el tótem 🤖✨\n"
        "\n"
        "Aquí tienes tu infografía con el diagnóstico inicial y las oportunidades clave para tu empresa.\n"
        "\n"
        "Para revisar los ajustes finales y convertirla en un plan ejecutivo listo para implementar, "
        "agenda aquí tu sesión:\n"
        "\n"
        "Gracias por visitarnos. Estamos listos para ayudarte a digitalizar tu operación y aumentar tus resultados 🚀\n"
        "https://crm.zoho.com/bookings/Reuni%C3%B3nde30minutos?rid=8d3bd9521421a88c26b1ebf9b533f56be94824ffc3a3a965a532f723536fabcd15e094adfda5adeac9657be51d3663e3gid4cc9894bf55f0cb0f871993c7fa580efdc4151c82b57b0328c660d2b78474623"
    )

    payload = {
        "channelId": WOZTELL_CHANNEL_ID,
        "recipientId": telefono,  # ej. "5214495097519"
        "response": [
            {
                "type": "TEXT",
                "text": texto,
            },
            {
                "type": "IMAGE",
                "url": imagen_url,
            },
        ],
    }

    print("\n=== DEBUG WhatsApp (TEXTO + IMAGEN) ===")
    print("POST", url.replace(WOZTELL_TOKEN, "*****TOKEN*****"))
    print("payload =")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("========================================")

    headers = {
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    print("STATUS:", resp.status_code)
    print("BODY:", resp.text)

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Error al enviar WhatsApp: {resp.text}")
    else:
        print("✅ WhatsApp (texto + imagen) enviado correctamente.")


# =========================
# FUNCIÓN: Email con infografía adjunta
# =========================
def enviar_email_infografia(png_path: str, email_to: Optional[str] = None) -> None:
    """
    Envía un correo con la infografía PNG adjunta.
    Usa configuración SMTP definida en .env.
    Si email_to es None, se usa EMAIL_DESTINO_DEFAULT.
    """
    destino = email_to or EMAIL_DESTINO_DEFAULT

    if not destino:
        print("⚠️ No hay EMAIL_DESTINO configurado ni email_to. No se envía correo.")
        return

    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS):
        print("⚠️ Datos SMTP incompletos. No se envía correo.")
        return

    from_header = EMAIL_FROM or SMTP_USER
    if SMTP_FROM_NAME:
        from_header = f"{SMTP_FROM_NAME} <{from_header}>"

    msg = EmailMessage()
    msg["From"] = from_header
    msg["To"] = destino
    msg["Subject"] = "Infografía Totem Evolución i3"

    msg.set_content(
        "Hola,\n\n"
        "Adjunto encontrarás la infografía generada por el Tótem Evolución i3.\n\n"
        "Saludos,\nEvolución i3"
    )

    with open(png_path, "rb") as f:
        data = f.read()
        msg.add_attachment(
            data,
            maintype="image",
            subtype="png",
            filename=os.path.basename(png_path),
        )

    print("\n=== DEBUG Email ===")
    print("SMTP_HOST :", SMTP_HOST)
    print("SMTP_PORT :", SMTP_PORT)
    print("SMTP_USER :", SMTP_USER)
    print("FROM      :", from_header)
    print("TO        :", destino)

    context = ssl.create_default_context()

    if SMTP_USE_TLS:
        # Zoho: smtp.zoho.com:587 + STARTTLS
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    else:
        # Caso alternativo: SSL directo (ej. puerto 465)
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

    print("✅ Email enviado correctamente.")


# =====================================================
# FUNCIÓN ORQUESTADORA: para usarla desde amain.py
# =====================================================
def flujo_infografia_whatsapp_email(
    datos_cliente: Dict[str, Any],
    telefono: Optional[str] = None,
    email: Optional[str] = None,
    nombre_archivo: str = "infografia_totem",
    enviar_whatsapp: bool = True,
    enviar_email_flag: bool = True,
) -> Dict[str, Any]:
    """
    Orquesta TODO el flujo:
      1) Llama a OpenAI (generar_texto_infografia) para obtener bloques de texto.
      2) Genera la infografía (PNG + PDF) con generar_infografia(...).
      3) Construye URL pública con build_infografia_public_url().
      4) Envía WhatsApp (TEXTO + IMAGEN) si enviar_whatsapp=True y hay teléfono.
      5) Envía Email con PNG adjunto si enviar_email_flag=True y hay email.

    Retorna un dict con info útil (paths y url) por si quieres loguearlo.
    """
    print(">>> [flujo_infografia_whatsapp_email] INICIO flujo con datos_cliente =", datos_cliente)

    # 1) Texto para infografía vía OpenAI
    bloques = generar_texto_infografia(datos_cliente)
    print(">>> [flujo_infografia_whatsapp_email] bloques generados =", bloques)

    # 2) Adaptar JSON a los slots que espera infographic_engine
    slots = {
        "titulo": f"Diagnóstico inicial - {datos_cliente.get('empresa', 'Evolución IA3 Tótem')}",
        "objetivo": bloques.get("objetivo", ""),
        "alcance": "\n".join(f"- {item}" for item in bloques.get("alcance", [])),
        "beneficios": "\n".join(f"- {item}" for item in bloques.get("beneficios", [])),
        "inversion": bloques.get("inversion_tiempo", ""),
    }

    print(">>> [flujo_infografia_whatsapp_email] slots para infographic_engine =", slots)

    # 3) Generar infografía (PNG + PDF)
    rutas = generar_infografia(slots, nombre_archivo=nombre_archivo)
    png_path = rutas["png"]
    pdf_path = rutas["pdf"]

    print("✅ Infografía exportada:")
    print("   PNG:", png_path)
    print("   PDF:", pdf_path)

    # 4) URL pública de la infografía vía ngrok
    url_publica = build_infografia_public_url()
    print("[flujo_infografia_whatsapp_email] URL pública de la infografía:", url_publica)

    # 5) Enviar WhatsApp (texto fijo + imagen) si aplica
    if enviar_whatsapp:
        tel_final = telefono or TELEFONO_DESTINO_DEFAULT
        if tel_final:
            try:
                enviar_whatsapp_imagen(url_publica, tel_final)
            except Exception as e:
                print(f"⚠️ Error al enviar WhatsApp dentro de flujo_infografia_whatsapp_email: {e!r}")
        else:
            print("⚠️ No hay teléfono disponible para enviar WhatsApp.")

    # 6) Enviar correo con la infografía adjunta si aplica
    if enviar_email_flag:
        try:
            enviar_email_infografia(png_path, email_to=email)
        except Exception as e:
            print(f"⚠️ Error al enviar Email dentro de flujo_infografia_whatsapp_email: {e!r}")

    return {
        "png_path": png_path,
        "pdf_path": pdf_path,
        "url_publica": url_publica,
        "telefono_usado": telefono or TELEFONO_DESTINO_DEFAULT,
        "email_usado": email or EMAIL_DESTINO_DEFAULT,
    }


# =========================
# MAIN: prueba manual
# =========================
def main() -> None:
    # 1) Datos de ejemplo del cliente (luego vendrán del Tótem)
    datos_cliente = {
        "nombre": "Visitante Zoholics",
        "empresa": "Empresa Demo",
        "OBJETIVO": "Quiere profesionalizar su operación comercial y financiera.",
        "problemas": "Falta de control, reportes tardíos y procesos manuales.",
        "necesidades": "Automatizar ventas, facturación y análisis de resultados.",
        # estas soluciones impactan semanas y costo en el prompt
        "soluciones": ["CRM", "Books", "Analytics"],
    }

    flujo_infografia_whatsapp_email(
        datos_cliente,
        telefono=None,           # usa el de .env por defecto
        email=None,              # usa EMAIL_DESTINO_DEFAULT
        nombre_archivo="infografia_totem",
        enviar_whatsapp=True,
        enviar_email_flag=True,
    )


if __name__ == "__main__":
    main()