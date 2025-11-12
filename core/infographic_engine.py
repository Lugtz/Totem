from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import datetime
import textwrap

# ----------------------------------------------------------------------
# 1. RUTAS Y CONFIGURACIÓN
# ----------------------------------------------------------------------

# Rutas oficiales
ROOT_DIR = Path(__file__).parent.parent.resolve()
ASSETS_DIR = ROOT_DIR / "core" / "infographic_assets"
OUTPUT_DIR = ROOT_DIR / "core" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- RECURSOS ---
AVATARES = list((ASSETS_DIR / "avatars").glob("*.png"))
ICONOS = {
    "objetivo": ASSETS_DIR / "icons" / "objetivo.png",
    "alcance": ASSETS_DIR / "icons" / "archivo.png",
    "beneficios": ASSETS_DIR / "icons" / "crecimiento.png",
    "inversion": ASSETS_DIR / "icons" / "bolsadinero.png",  # Se mantiene aunque no se use en esta versión
    "bolsa_dinero": ASSETS_DIR / "icons" / "bolsadinero.png",
}
# RUTA CORREGIDA
LOGO_EI3 = ASSETS_DIR / "logos" / "Logo ei3 con sombra _ baja.png"  # Usado como logo principal en la esquina superior izquierda

# --- COLORES INSTITUCIONALES (Basado en la imagen de referencia) ---
AZUL_PETROLEO = (1, 86, 102)     # Fondo Principal
NARANJA = (255, 124, 0)          # Acento principal (Títulos/Iconos)
BLANCO = (255, 255, 255)         # Texto principal
VERDE_AZULADO = (0, 153, 153)    # Color usado en el header del avatar

# --- FUENTE: ARIA ---
FONT_FILENAME = "arial.ttf"
try:
    # Intenta cargar Arial desde rutas comunes o por nombre
    FONT_BASE = ImageFont.truetype(FONT_FILENAME, 42)
except Exception:
    print(f"⚠ Fuente {FONT_FILENAME} no encontrada, usando fuente por defecto.")
    FONT_BASE = ImageFont.load_default()

def get_font(size):
    return FONT_BASE.font_variant(size=size)

# Tamaños de fuente ajustados (se pueden ajustar más si es necesario)
FONT_TITULO_PRINCIPAL = get_font(60)  # Ajustado para que quepa mejor
FONT_EMPRESA = get_font(30)           # Para "Evolucióni3" en la esquina
FONT_TITULO_SECCION = get_font(40)    # Títulos "Objetivo", "Beneficios"
FONT_CONTENIDO = get_font(28)         # Contenido principal
FONT_FOOTER = get_font(32)            # Pie de página

# Espaciado vertical para contenido general
LINE_SPACING = 12

# Función mejorada para envolver texto y medir su altura (MÉTODO textbbox + COMPENSACIÓN)
def wrap_text_and_measure(text, font, max_width, draw, line_spacing, start_pos=(0, 0), padding_x=0):
    lines = []
    words = text.split(' ')
    current_line = []
    
    # El ancho real para el texto es max_width - (2 * padding_x)
    effective_max_width = max_width - (2 * padding_x)
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        
        # Medir el ancho de la línea de prueba
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width < effective_max_width:
            current_line.append(word)
        else:
            if not current_line:  # Si una palabra es más ancha que la columna, se pone sola
                lines.append(word)
                current_line = []
            else:
                lines.append(' '.join(current_line))
                current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))
        
    wrapped_content = '\n'.join(lines)
    
    if wrapped_content:
        # Usamos el start_pos (bx_text, content_y) para obtener la caja delimitadora real.
        bbox = draw.textbbox((start_pos[0] + padding_x, start_pos[1]), wrapped_content, font=font, spacing=line_spacing)
        total_height = bbox[3] - (start_pos[1])  # Altura real desde el inicio del texto
        total_height += 15  # Pequeña compensación extra para asegurar espacio
    else:
        total_height = 0
    
    return wrapped_content, total_height


# ----------------------------------------------------------------------
# 2. FUNCIÓN PRINCIPAL
# ----------------------------------------------------------------------

def generate_infographic(data: dict):
    # Dimensiones para una proporción 19:9 (ej. 2052x1080 o similar)
    width = 1920  # Ancho 16:9
    height = 1080 
    img = Image.new("RGB", (width, height), AZUL_PETROLEO)
    draw = ImageDraw.Draw(img)

    # --- CONSTANTES DE LAYOUT ---
    AVATAR_AREA_WIDTH = 450      # Ancho para el área del avatar (incluye margen)
    GLOBAL_MARGIN = 50           # Margen general
    INNER_COL_SPACING = 70       # Espacio entre columnas
    TEXT_PADDING = 25            # Margen interno de texto

    # 1. ZONA IZQUIERDA (Header y Avatar)
    draw.rectangle([0, 0, AVATAR_AREA_WIDTH, 150], fill=VERDE_AZULADO) 

    # Logo Principal (EI3, esquina superior izquierda)
    logo_ei3_img = Image.open(LOGO_EI3).convert("RGBA").resize((100, 100))
    img.paste(logo_ei3_img, (GLOBAL_MARGIN, 25), logo_ei3_img)

    # Título Principal
    titulo_texto = data.get("proyecto", "INFOGRAFÍA ESTRATÉGICA")
    content_area_start_x = AVATAR_AREA_WIDTH + GLOBAL_MARGIN
    content_area_end_x = width - GLOBAL_MARGIN
    content_area_center_x = content_area_start_x + (content_area_end_x - content_area_start_x) / 2
    titulo_bbox = draw.textbbox((0, 0), titulo_texto, font=FONT_TITULO_PRINCIPAL)
    titulo_width = titulo_bbox[2] - titulo_bbox[0]
    draw.text((content_area_center_x - titulo_width / 2, 55), titulo_texto, fill=BLANCO, font=FONT_TITULO_PRINCIPAL)
    
    # Logo de Evolucióni3
    empresa_texto = "Evolucióni3"
    empresa_bbox = draw.textbbox((0, 0), empresa_texto, font=FONT_EMPRESA)
    empresa_width = empresa_bbox[2] - empresa_bbox[0]
    draw.text((width - GLOBAL_MARGIN - empresa_width, 55), empresa_texto, fill=BLANCO, font=FONT_EMPRESA)

    # --- 2. Avatar (Lado Izquierdo) ---
    if AVATARES:
        avatar_path = random.choice(AVATARES)
        avatar = Image.open(avatar_path).convert("RGBA").resize((450, 650))
        avatar_y = height - avatar.height - GLOBAL_MARGIN + 100
        img.paste(avatar, (0, avatar_y), avatar)

    # --- 3. Bloques de Contenido (Alineado en 2 columnas) ---
    COLUMNS_COUNT = 2 
    available_content_width = width - AVATAR_AREA_WIDTH - GLOBAL_MARGIN - GLOBAL_MARGIN - INNER_COL_SPACING
    col_width = int(available_content_width / COLUMNS_COUNT)
    first_col_text_start_x = AVATAR_AREA_WIDTH + GLOBAL_MARGIN + TEXT_PADDING

    secciones = [
        {"titulo": "Objetivo", "contenido": data.get("objetivo", "Objetivo no definido."), "icon": ICONOS["objetivo"], "sub_icon": ICONOS["alcance"], "sub_contenido": "\n".join(data.get("alcance", ["Implementación no definida."]))},
        {"titulo": "Beneficios esperados", "contenido": data.get("beneficios", ["Beneficios no definidos."]), "icon": ICONOS["beneficios"]},
    ]

    start_y = 250 
    for idx, sec in enumerate(secciones):
        bx_col_start = AVATAR_AREA_WIDTH + GLOBAL_MARGIN + idx * (col_width + INNER_COL_SPACING)
        bx_text = bx_col_start + TEXT_PADDING
        by = start_y

        icon_size = 40
        icon_x = bx_col_start + TEXT_PADDING
        icon_y = by
        draw.text((icon_x + icon_size + 10, by + 5), sec['titulo'], fill=NARANJA, font=FONT_TITULO_SECCION)
        icon = Image.open(sec['icon']).convert("RGBA").resize((icon_size, icon_size))
        img.paste(icon, (icon_x, icon_y), icon)

        content_y = by + icon_size + 30 
        contenido_str = "\n".join(sec['contenido']) if isinstance(sec['contenido'], list) else sec['contenido']
        wrapped_content, total_height_content = wrap_text_and_measure(
            contenido_str, FONT_CONTENIDO, col_width, draw, LINE_SPACING, (bx_text, content_y), TEXT_PADDING
        )
        draw.multiline_text((bx_text, content_y), wrapped_content, fill=BLANCO, font=FONT_CONTENIDO, spacing=LINE_SPACING)

        if idx == 0:
            end_of_objetivo_y = content_y + total_height_content
            subtitle_y = end_of_objetivo_y + 30 
            sub_icon_size = 40
            sub_icon = Image.open(sec['sub_icon']).convert("RGBA").resize((sub_icon_size, sub_icon_size))
            img.paste(sub_icon, (bx_text, subtitle_y), sub_icon) 
            sub_content_y = subtitle_y + sub_icon_size + 10
            wrapped_sub_content, _ = wrap_text_and_measure(
                sec['sub_contenido'], FONT_CONTENIDO, col_width, draw, LINE_SPACING, (bx_text, sub_content_y), TEXT_PADDING
            )
            draw.multiline_text((bx_text, sub_content_y), wrapped_sub_content, fill=BLANCO, font=FONT_CONTENIDO, spacing=LINE_SPACING)

    # --- 4. Pie de página ---
    draw.line([(0, height - 150), (width, height - 150)], fill=NARANJA, width=5)
    frase = "Date la Oportunidad, nosotros damos los Resultados"
    frase_x = GLOBAL_MARGIN
    draw.text((frase_x, height - 100), frase, fill=BLANCO, font=FONT_FOOTER)
    url = "www.evolucioni3.com"
    url_bbox = draw.textbbox((0, 0), url, font=FONT_CONTENIDO)
    url_width = url_bbox[2] - url_bbox[0]
    draw.text((width - GLOBAL_MARGIN - url_width, height - 100), url, fill=BLANCO, font=FONT_CONTENIDO)

    # Guardar
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_png = OUTPUT_DIR / f"infografia_{timestamp}.png"
    path_pdf = OUTPUT_DIR / f"infografia_{timestamp}.pdf"
    img.save(path_png)
    img.convert("RGB").save(path_pdf, "PDF", resolution=100.0)
    print(f"✅ PNG generado en: {path_png}")
    print(f"✅ PDF generado en: {path_pdf}")
    return path_png, path_pdf


# ----------------------------------------------------------------------
# 3. Alias para compatibilidad con FastAPI
# ----------------------------------------------------------------------
def generate(data: dict):
    """
    Alias para FastAPI que llama internamente a generate_infographic.
    Evita errores de importación al usar /infographic/generate.
    """
    png_path, pdf_path = generate_infographic(data)

    # Convertimos las rutas a string para que sean serializables en JSON
    return {
        "png_path": str(png_path),
        "pdf_path": str(pdf_path)
    }
