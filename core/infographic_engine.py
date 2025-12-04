# -*- coding: utf-8 -*-
# core/infographic_engine.py
# Versión refinada visualmente, estilo "IA corporativa" con colores y layout controlado

import os
import random
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

RELATIVE_PATH_PARTS = ["core", "infographic_assets"]

# === Rutas base (usa tu ruta actual) ===
BASE_PATH = Path.cwd().joinpath(*RELATIVE_PATH_PARTS)
LOGO_PATH = os.path.join(BASE_PATH, "logos", "logo ei3 original _ baja.png")
ICON_PATH = os.path.join(BASE_PATH, "icons")
AVATAR_DIR = os.path.join(BASE_PATH, "avatars")

# Tamaño del canvas (infografía base)
WIDTH, HEIGHT = 1536, 1024

# IMPORTANTE: carpeta pública para servir por HTTP/ngrok
# Quedará como: <root>/infografias/infografia_totem.png
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "infografias")

# Ruta a SumatraPDF portable (ajusta si tu exe tiene otro nombre)
SUMATRA_PATH = r"C:\Users\joses\OneDrive\Escritorio\SumatraPDF-3.5.2-64\SumatraPDF-3.5.2-64.exe"

# Nombre EXACTO de la impresora en Windows
PRINTER_NAME = r"HP Color LaserJet MFP M477fdw (3F853F)"

# Paletas de color compatibles con el logo (fondo superior, fondo inferior, color texto)
PALETAS = [
    ((240, 240, 240), (220, 220, 220), (40, 40, 40)),      # Gris claro
    ((255, 255, 255), (245, 245, 245), (30, 30, 30)),      # Blanco total
    ((240, 250, 250), (210, 230, 230), (30, 60, 70)),      # Turquesa muy claro
    ((230, 240, 255), (210, 220, 240), (20, 40, 60)),      # Azul claro
    ((225, 245, 245), (180, 220, 220), (20, 40, 40)),      # Turquesa medio
    ((215, 225, 235), (170, 190, 210), (25, 35, 50)),      # Azul-gris profesional
    ((230, 230, 240), (200, 200, 220), (35, 35, 60)),      # Gris azulado
    ((245, 245, 240), (220, 220, 210), (50, 50, 50)),      # Arena muy clara
]


# ------------------------
# Utilidades de dibujo
# ------------------------
def get_font(size: int, bold: bool = False):
    """Carga Arial o Arial Bold desde el sistema."""
    font_path = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(font_path, size)


def draw_gradient(draw, top, bottom):
    """Degradado vertical suave."""
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        r = int(top[0] * (1 - ratio) + bottom[0] * ratio)
        g = int(top[1] * (1 - ratio) + bottom[1] * ratio)
        b = int(top[2] * (1 - ratio) + bottom[2] * ratio)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))


def text_wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
    """Envuelve texto en líneas que quepan en max_width."""
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    lines = []
    line = ""

    for word in words:
        test = (line + " " + word).strip()
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word

    if line:
        lines.append(line)

    return lines


def draw_body_text_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill,
    max_width: int,
    card_h: int,
    max_lines: int = 5,
):
    """
    Dibuja el texto del cuerpo dentro de la tarjeta:
    - Envuelve por ancho.
    - Limita a max_lines.
    - Centrado verticalmente dentro de la tarjeta.
    """
    # Si viene lista (ej. bullets), la unimos en texto plano
    if isinstance(text, list):
        text = " • ".join(str(t) for t in text)

    lines = text_wrap(draw, str(text), font, max_width)

    if not lines:
        return

    # Si hay más líneas de las permitidas, recortamos y agregamos "…"
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if not lines[-1].endswith("…"):
            lines[-1] = (lines[-1][: max(0, len(lines[-1]) - 1)] + "…").strip()

    line_height = font.size + 6
    total_height = len(lines) * line_height

    # Centrado vertical dentro de la tarjeta
    start_y = y + (card_h - total_height) // 2

    current_y = start_y
    for line in lines:
        draw.text((x, current_y), line, font=font, fill=fill)
        current_y += line_height


def draw_card(draw, img, x, y, title, body, icon_file, text_color):
    """Tarjetas visuales tipo 'glass'."""
    card_w, card_h = 560, 260  # altura fija para control de texto
    radius = 30
    fill_color = (255, 255, 255, 245)
    draw.rounded_rectangle(
        (x, y, x + card_w, y + card_h),
        radius=radius,
        fill=fill_color,
    )

    # Icono
    icon_path = os.path.join(ICON_PATH, icon_file)
    if os.path.exists(icon_path):
        icon = Image.open(icon_path).convert("RGBA").resize((64, 64))
        img.paste(icon, (x + 25, y + 25), icon)

    # Título de la tarjeta
    title_font = get_font(30, bold=True)
    draw.text((x + 110, y + 30), title, font=title_font, fill=text_color)

    # Texto del cuerpo centrado verticalmente
    body_font = get_font(26)
    body_x = x + 30
    body_y = y + 90
    max_width = 500

    draw_body_text_centered(
        draw=draw,
        text=body,
        x=body_x,
        y=body_y,
        font=body_font,
        fill=text_color,
        max_width=max_width,
        card_h=card_h - 90,
        max_lines=5,
    )


def draw_avatar(img):
    """Avatar 3D a la derecha."""
    if not os.path.isdir(AVATAR_DIR):
        return
    archivos = [f for f in os.listdir(AVATAR_DIR) if f.lower().endswith(".png")]
    if not archivos:
        return
    path = os.path.join(AVATAR_DIR, random.choice(archivos))
    avatar = Image.open(path).convert("RGBA")
    wmax, hmax = 300, int(HEIGHT * 0.8)
    scale = min(wmax / avatar.width, hmax / avatar.height)
    avatar = avatar.resize((int(avatar.width * scale), int(avatar.height * scale)))
    x = WIDTH - 280 + (280 - avatar.width) // 2
    y = HEIGHT - avatar.height - 60
    img.paste(avatar, (x, y), avatar)


def draw_logo(img):
    """Logo Evolución i3 en la esquina superior derecha."""
    if os.path.isfile(LOGO_PATH):
        logo = Image.open(LOGO_PATH).convert("RGBA")
        max_w, max_h = 140, 80
        scale = min(max_w / logo.width, max_h / logo.height, 1.0)
        new_w = int(logo.width * scale)
        new_h = int(logo.height * scale)
        logo = logo.resize((new_w, new_h), Image.LANCZOS)
        x = WIDTH - new_w - 40
        y = 30
        img.paste(logo, (x, y), logo)


def elegir_paleta():
    return random.choice(PALETAS)


# ========= 1) CONVERTIR PNG -> PDF MEDIA CARTA =========

def png_a_pdf_mediacarta(path_png: str, path_pdf: str) -> None:
    """
    Convierte un PNG a un PDF tamaño Media Carta (5.5 x 8.5 pulgadas)
    en horizontal (landscape), ocupando toda la página.
    """
    if not os.path.isfile(path_png):
        print(f"⚠️ La imagen NO existe: {path_png}")
        return

    # Media carta (5.5 x 8.5) pero en horizontal: 8.5 de ancho x 5.5 de alto
    width = 8.5 * inch
    height = 5.5 * inch

    c = canvas.Canvas(path_pdf, pagesize=(width, height))

    # Dibujar la imagen ocupando toda la página
    c.drawImage(path_png, 0, 0, width=width, height=height)

    c.showPage()
    c.save()

    print(f"✅ PDF media carta generado: {path_pdf}")


# ========= 2) LIMPIAR COLA DE IMPRESIÓN =========

def limpiar_cola_impresora(printer_name: str) -> None:
    """Borra TODOS los trabajos pendientes de la cola de la impresora indicada."""
    print(f"🧹 Limpiando cola de impresión de: {printer_name!r}")
    cmd = [
        "powershell",
        "-Command",
        (
            f"Get-PrintJob -PrinterName '{printer_name}' "
            "| Remove-PrintJob -Confirm:$false"
        ),
    ]
    try:
        subprocess.run(cmd, check=False)
        print("✅ Cola de impresión limpiada (o ya estaba vacía).")
    except Exception as e:
        print(f"⚠️ No se pudo limpiar la cola de impresión: {e}")


# ========= 3) IMPRIMIR PDF CON SUMATRA (1 COPIA, AJUSTADO) =========

def imprimir_pdf_una_copia(path_pdf: str) -> None:
    """
    Imprime un PDF usando SumatraPDF:
    - 1 sola copia
    - Escalado 'fit' a la hoja configurada en la impresora (media carta).
    """
    print("=== IMPRESIÓN PDF (Sumatra, 1 copia, fit) ===")
    print(f"PDF           : {path_pdf}")
    print(f"Sumatra exe   : {SUMATRA_PATH}")

    if not os.path.isfile(path_pdf):
        print(f"⚠️ El archivo PDF NO existe: {path_pdf}")
        return

    if os.name != "nt":
        print(f"⚠️ Solo implementado para Windows (os.name={os.name})")
        return

    if not os.path.isfile(SUMATRA_PATH):
        print(f"⚠️ SumatraPDF no encontrado en: {SUMATRA_PATH}")
        return

    # 1) Limpiar cola antes de imprimir
    limpiar_cola_impresora(PRINTER_NAME)
    time.sleep(1)

    # 2) Mandar SOLO 1 copia, ajustada a la hoja
    try:
        print('\n--- SumatraPDF -print-to-default -print-settings "fit,1x" ---')
        subprocess.run(
            [
                SUMATRA_PATH,
                "-print-to-default",
                "-exit-on-print",
                "-print-settings",
                "fit,1x",   # ajusta a la hoja + UNA sola copia
                path_pdf,
            ],
            check=True,
            shell=False,
        )
        print("🖨 (Sumatra) PDF enviado a la impresora (1 copia, media carta).")
    except Exception as e:
        print(f"❌ Error al imprimir con SumatraPDF: {e}")


# ========= 4) FUNCIÓN PÚBLICA QUE USA GENERAR_INFOGRAFIA =========

def imprimir_pdf(path_pdf: str) -> None:
    """
    Orquestador:
    - A partir de path_pdf (ej: .../infografia_totem.pdf) deduce el PNG hermano.
    - Genera un PDF en media carta (sufijo _MEDIACARTA.pdf).
    - Imprime ese PDF con Sumatra (1 copia).
    """
    if not path_pdf:
        print("⚠️ imprimir_pdf llamado sin ruta de PDF.")
        return

    base, ext = os.path.splitext(path_pdf)
    path_png = base + ".png"
    path_pdf_mediacarta = base + "_MEDIACARTA.pdf"

    # 1) Convertir PNG -> PDF media carta
    png_a_pdf_mediacarta(path_png, path_pdf_mediacarta)

    # 2) Imprimir ese PDF media carta
    imprimir_pdf_una_copia(path_pdf_mediacarta)


# ------------------------
# Función principal
# ------------------------
def generar_infografia(slots, nombre_archivo="infografia_totem"):
    """
    slots: dict con claves:
      - titulo
      - objetivo
      - alcance
      - beneficios
      - inversion
    nombre_archivo: sin extensión (se guardan .png y .pdf)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = Image.new("RGBA", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)

    # Paleta
    bg_top, bg_bottom, text_color = elegir_paleta()
    draw_gradient(draw, bg_top, bg_bottom)

    # Título (limpio, corto y sin chocar con el logo)
    raw_title = str(slots.get("titulo", "[Sin título]") or "").strip()

    # Quitar prefijo "Proyecto " si viene
    lower = raw_title.lower()
    if lower.startswith("proyecto "):
        title_text = raw_title[9:].strip()
    else:
        title_text = raw_title

    if not title_text:
        title_text = "[Sin título]"

    title_font = get_font(52, bold=True)
    # Reservamos espacio para el logo (aprox 200 px a la derecha)
    max_title_width = WIDTH - 100 - 200  # margen izq 100, margen/logo der ~200

    # Si sigue siendo muy largo, recortamos elegante
    if ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(title_text, font=title_font) > max_title_width:
        original = title_text
        while (
            ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(title_text + "…", font=title_font) > max_title_width
            and len(title_text) > 3
        ):
            if " " in title_text:
                title_text = title_text.rsplit(" ", 1)[0]
            else:
                title_text = title_text[:-1]
        if title_text != original:
            title_text = title_text.rstrip(".") + "…"

    draw.text(
        (100, 60),
        title_text,
        font=title_font,
        fill=text_color,
    )

    # Subtítulo fijo
    subtitle_font = get_font(24)
    draw.text(
        (100, 130),
        "Objetivo, alcance, beneficios e inversión",
        font=subtitle_font,
        fill=text_color,
    )

    # Tarjetas (cuerpo)
    bloques = [
        ("Objetivo", slots.get("objetivo", ""), "objetivo.png"),
        ("Alcance", slots.get("alcance", ""), "archivo.png"),
        ("Beneficios esperados", slots.get("beneficios", ""), "crecimiento.png"),
        ("Inversión y tiempo", slots.get("inversion", ""), "bolsadinero.png"),
    ]

    x0, y0 = 100, 220
    for i, (title, body, icon) in enumerate(bloques):
        col = i % 2
        row = i // 2
        x = x0 + col * 620
        y = y0 + row * 280
        draw_card(draw, img, x, y, title, body, icon, text_color)

    # Footer
    footer_font = get_font(24)
    draw.text(
        (100, HEIGHT - 80),
        "Date la oportunidad, nosotros los resultados",
        font=footer_font,
        fill=text_color,
    )
    draw.text(
        (100, HEIGHT - 45),
        "www.evolucioni3.com",
        font=footer_font,
        fill=text_color,
    )

    draw_logo(img)
    draw_avatar(img)

    path_png = os.path.join(OUTPUT_DIR, f"{nombre_archivo}.png")
    path_pdf = os.path.join(OUTPUT_DIR, f"{nombre_archivo}.pdf")
    img.save(path_png, "PNG")
    img.convert("RGB").save(path_pdf, "PDF", resolution=300.0)

    print("✅ Infografía exportada:")
    print("   PNG:", path_png)
    print("   PDF:", path_pdf)

    result = {
        "png": os.path.abspath(path_png),
        "pdf": os.path.abspath(path_pdf),
    }

    # 🔹 Mandar a imprimir automáticamente el PDF en media carta
    imprimir_pdf(result["pdf"])

    return result
