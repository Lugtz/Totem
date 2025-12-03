# -*- coding: utf-8 -*-
"""
core.whatsapp_email_infografia

Flujo completo:
1) Llama a OpenAI (infografia_prompt_generator) para generar el TEXTO de la infografía.
2) Genera la infografía (PNG + PDF) con infographic_engine usando ese texto.
3) Obtiene la URL pública de la infografía usando ngrok (core.infographic_link).
4) Envía WhatsApp (TEXTO FIJO + IMAGEN) por Woztell Bot API.
5) Envía la infografía por correo (adjunto PNG) usando SMTP Zoho.
"""

from __future__ import annotations

import os
import json
import smtplib
import ssl
from email.message import EmailMessage

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
TELEFONO_DESTINO = os.getenv("WOZTELL_TEST_PHONE", "5214495097519")

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
EMAIL_DESTINO = os.getenv(
    "EMAIL_TO_TEST",
    "valadezgutierrezmaguadalupe@gmail.com"
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
def enviar_email_infografia(png_path: str) -> None:
    """
    Envía un correo con la infografía PNG adjunta.
    Usa configuración SMTP definida en .env
    """
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and EMAIL_DESTINO):
        print("⚠️ Datos SMTP incompletos. No se envía correo.")
        return

    from_header = EMAIL_FROM or SMTP_USER
    if SMTP_FROM_NAME:
        from_header = f"{SMTP_FROM_NAME} <{from_header}>"

    msg = EmailMessage()
    msg["From"] = from_header
    msg["To"] = EMAIL_DESTINO
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
    print("TO        :", EMAIL_DESTINO)

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


# =========================
# MAIN: flujo completo
# =========================
def main() -> None:
    # 1) Datos de ejemplo del cliente (luego vendrán del Tótem)
    datos_cliente = {
        "nombre": "Visitante Zoholics",
        "empresa": "Empresa Demo",
        "objetivo": "Quiere profesionalizar su operación comercial y financiera.",
        "problemas": "Falta de control, reportes tardíos y procesos manuales.",
        "necesidades": "Automatizar ventas, facturación y análisis de resultados.",
        # estas soluciones impactan semanas y costo en el prompt
        "soluciones": ["CRM", "Books", "Analytics"],
    }

    # 2) Pedir a OpenAI el TEXTO de la infografía (JSON)
    print(">>> Generando texto de infografía con OpenAI...")
    bloques = generar_texto_infografia(datos_cliente)

    # 3) Adaptar JSON a los slots que espera infographic_engine
    slots = {
        "titulo": "Diagnóstico inicial - Evolución IA3 Tótem",
        "objetivo": bloques.get("objetivo", ""),
        "alcance": "\n".join(f"- {item}" for item in bloques.get("alcance", [])),
        "beneficios": "\n".join(f"- {item}" for item in bloques.get("beneficios", [])),
        "inversion": bloques.get("inversion_tiempo", ""),
    }

    # 4) Generar infografía (PNG + PDF)
    rutas = generar_infografia(slots, nombre_archivo="infografia_totem")
    png_path = rutas["png"]
    pdf_path = rutas["pdf"]

    print("✅ Infografía exportada:")
    print("   PNG:", png_path)
    print("   PDF:", pdf_path)

    # 5) URL pública de la infografía vía ngrok
    url_publica = build_infografia_public_url()
    print("[MAIN] URL pública de la infografía:", url_publica)

    # 6) Enviar WhatsApp (texto fijo + imagen)
    try:
        enviar_whatsapp_imagen(url_publica, TELEFONO_DESTINO)
    except Exception as e:
        print(f"⚠️ Error al enviar WhatsApp: {e!r}")

    # 7) Enviar correo con la infografía adjunta (opcional)
    try:
        enviar_email_infografia(png_path)
    except Exception as e:
        print(f"⚠️ Error al enviar Email: {e!r}")


if __name__ == "__main__":
    main()
