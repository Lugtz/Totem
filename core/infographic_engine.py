import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =======================
# RUTAS UNIVERSALES
# =======================

# Carpeta donde está este archivo: core/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta de assets: core/infographic_assets
BASE_PATH = os.path.join(CURRENT_DIR, "infographic_assets")

# Subcarpetas dentro de infographic_assets
LOGO_PATH = os.path.join(BASE_PATH, "logos", "logo ei3 original _ baja.png")
ICON_PATH = os.path.join(BASE_PATH, "icons")
AVATAR_DIR = os.path.join(BASE_PATH, "avatars")

# Tamaño del canvas
WIDTH, HEIGHT = 1536, 1024

# Carpeta de salida para las infografías (en la raíz del proyecto)
# Si quieres que salga en la carpeta "infografias" al lado de core/, haz:
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "infografias")
LOGO_PATH = os.path.join(BASE_PATH, "logos", "logo ei3 original _ baja.png")
ICON_PATH = os.path.join(BASE_PATH, "icons")
AVATAR_DIR = os.path.join(BASE_PATH, "avatars")

# Tamaño del canvas
WIDTH, HEIGHT = 1536, 1024

# IMPORTANTE: carpeta pública para servir por HTTP/ngrok
# Quedará como: <root>/infografias/infografia_totem.png
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "infografias")

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


def text_wrap(draw, text, font, max_width):
    words = text.split()
    lines, line = [], ""
    for word in words:
        test = line + word + " "
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
            lines.append(line.strip())
            line = word + " "
    lines.append(line.strip())
    return lines


def draw_text(draw, text, position, font, fill, max_width):
    x, y = position
    for line in text_wrap(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + 6


def draw_card(draw, img, x, y, title, body, icon_file, text_color):
    """Tarjetas visuales tipo 'glass'."""
    card_w, card_h = 560, 260  # reducido para dejar espacio derecho
    radius = 30
    fill_color = (255, 255, 255, 245)
    draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=radius, fill=fill_color)

    # Icono
    icon_path = os.path.join(ICON_PATH, icon_file)
    if os.path.exists(icon_path):
        icon = Image.open(icon_path).convert("RGBA").resize((64, 64))
        img.paste(icon, (x + 25, y + 25), icon)

    # Textos
    draw.text((x + 110, y + 30), title, font=get_font(30, bold=True), fill=text_color)
    draw_text(draw, body, (x + 30, y + 110), font=get_font(26), fill=text_color, max_width=500)


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

    # Título
    draw.text(
        (100, 60),
        slots.get("titulo", "[Sin título]"),
        font=get_font(52, bold=True),
        fill=text_color,
    )
    draw.text(
        (100, 130),
        "Objetivo, alcance, beneficios e inversión",
        font=get_font(24),
        fill=text_color,
    )

    # Tarjetas
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
    draw.text(
        (100, HEIGHT - 80),
        "Date la oportunidad, nosotros los resultados",
        font=get_font(24),
        fill=text_color,
    )
    draw.text(
        (100, HEIGHT - 45),
        "www.evolucioni3.com",
        font=get_font(24),
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

    return {
        "png": os.path.abspath(path_png),
        "pdf": os.path.abspath(path_pdf),
    }
