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
# Nota: La infografía base NO usa FONDOS, solo color sólido
AVATARES = list((ASSETS_DIR / "avatars").glob("*.png"))
ICONOS = {
    "objetivo": ASSETS_DIR / "icons" / "objetivo.png",
    "alcance": ASSETS_DIR / "icons" / "archivo.png",
    "beneficios": ASSETS_DIR / "icons" / "crecimiento.png",
    "inversion": ASSETS_DIR / "icons" / "bolsadinero.png",
    # Usaremos una versión del icono de inversión para el bloque "dinero"
    "bolsa_dinero": ASSETS_DIR / "icons" / "bolsadinero.png", 
}
LOGO_EI3 = ASSETS_DIR / "logos" / "Logo ei3 con sombra.png" # Se mantiene para el pie de página

# --- COLORES INSTITUCIONALES ---
AZUL_PETROLEO = (1, 86, 102)     # Fondo Principal
NARANJA = (255, 124, 0)          # Acento principal (Títulos/Iconos)
VERDE_AZULADO = (0, 153, 153)    # Acento secundario (No usado, pero disponible)
GRIS_OSCURO = (65, 64, 66)       # Texto de Contenido
BLANCO = (255, 255, 255)         # Fondo de Bloques y Texto en encabezado/pie


# --- FUENTE: CAMBIADA A ARIA ---
# Usamos 'arial.ttf' que es una fuente muy común. Si falla, carga la predeterminada.
FONT_FILENAME = "arial.ttf"
try:
    # Intenta resolver la ruta de Arial dentro de ROOT_DIR o buscarla por nombre
    # Si la encuentras en una ruta conocida o en el sistema, úsala.
    # Para ser robustos en Windows/Linux, usamos una ruta común o el nombre:
    try:
        FONT_PATH = str(Path(ROOT_DIR.anchor) / "Windows" / "Fonts" / FONT_FILENAME)
        FONT_BASE = ImageFont.truetype(FONT_PATH, 42)
    except:
        FONT_BASE = ImageFont.truetype(FONT_FILENAME, 42)
except Exception:
    print(f"⚠ Fuente {FONT_FILENAME} no encontrada, usando fuente por defecto.")
    FONT_BASE = ImageFont.load_default()

def get_font(size):
    return FONT_BASE.font_variant(size=size)

FONT_PROYECTO = get_font(65)
FONT_TITULO_SECCION = get_font(40)
FONT_CONTENIDO = get_font(28)
FONT_FOOTER = get_font(35)


# ----------------------------------------------------------------------
# 2. FUNCIÓN PRINCIPAL
# ----------------------------------------------------------------------

def generate_infographic(data: dict):
    # Crear lienzo de dimensiones de la base (2280x1080)
    width, height = 2280, 1080 
    img = Image.new("RGB", (width, height), AZUL_PETROLEO)
    draw = ImageDraw.Draw(img)

    # --- Zonas de Layout ---
    AVATAR_W = 450
    MARGIN = 70
    
    # 1. ZONA IZQUIERDA (Avatar)
    # Dibujar la esquina superior izquierda si se desea (ej: para un color diferente)
    # draw.rectangle([0, 0, AVATAR_W, height], fill=VERDE_AZULADO) 

    # --- 2. Encabezado ---
    # SIN LOGO INSTITUCIONAL a la izquierda
    
    # LOGO EI3 a la derecha (Empresa)
    logo = Image.open(LOGO_EI3).convert("RGBA").resize((250, 75))
    draw.text((width - logo.width - MARGIN, 80), "Evolucióni3", fill=BLANCO, font=get_font(40))
    # img.paste(logo, (width - logo.width - MARGIN, 60), logo) # Opción de usar la imagen del logo
    
    # Título principal del proyecto (Alineado a la derecha del Avatar)
    titulo_texto = data.get("proyecto", "Alcance para proveedores")
    titulo_x = AVATAR_W + MARGIN
    draw.text((titulo_x, 80), titulo_texto, fill=BLANCO, font=FONT_PROYECTO)


    # --- 3. Avatar (Lado Izquierdo) ---
    if AVATARES:
        avatar_path = random.choice(AVATARES)
        avatar = Image.open(avatar_path).convert("RGBA").resize((450, 750))
        # Posición para que el avatar "salga" ligeramente de la parte inferior
        avatar_y = height - avatar.height - MARGIN + 100 
        img.paste(avatar, (20, avatar_y), avatar)


    # --- 4. Bloques de Contenido (Lado Derecho) ---
    secciones = [
        {"titulo": "Objetivo", "contenido": data.get("objetivo", "Objetivo no definido."), "icon": ICONOS["objetivo"]},
        {"titulo": "Beneficios esperados", "contenido": "\n".join(data.get("beneficios", ["Beneficios no definidos."])), "icon": ICONOS["beneficios"]},
        {"titulo": "Inversión y tiempo", "contenido": f"Inversión: {data.get('inversion', 'N/A')}\nTiempo: {data.get('tiempo', 'N/A')}\nFase de mantenimiento: {data.get('mantenimiento', 'N/A')}", "icon": ICONOS["bolsa_dinero"]},
    ]

    # El alcance se maneja como texto adicional o se integra al objetivo si no hay un bloque claro para él
    alcance_list = data.get("alcance", [])
    alcance_str = "\n".join(alcance_list)
    
    # Modificar el primer bloque para incluir el alcance si el campo "alcance" existe y no es un bloque propio
    if alcance_list and len(alcance_list) > 0:
        secciones[0]['contenido'] += f"\n\nImplementación:\n{alcance_str}"

    
    # Definición del Layout
    start_y = 250
    block_height = height - start_y - 200 # Altura dinámica
    space_x = 40 # Espacio entre bloques
    
    # Ajuste de layout para tres bloques
    col_width = (width - AVATAR_W - MARGIN - space_x * 2) / 3 
    
    # Recorrer y dibujar los 3 bloques principales
    for idx, sec in enumerate(secciones):
        
        # Coordenada X de inicio de cada bloque
        bx = int(titulo_x + idx * (col_width + space_x)) # Asegurar que bx sea entero
        by = start_y
        
        # 4.1. Dibujar el Título del Bloque y el Ícono (en la parte superior del área de texto)
        
        # Calcular ancho del texto del título para centrarlo o alinearlo
        title_bbox = draw.textbbox((bx, by), sec['titulo'], font=FONT_TITULO_SECCION)
        # CORRECCIÓN DE ERROR: Aseguramos que el cálculo sea un entero
        titulo_seccion_w = int(title_bbox[2] - title_bbox[0]) 
        
        # Dibujar Título
        draw.text((bx, by), sec['titulo'], fill=BLANCO, font=FONT_TITULO_SECCION)
        
        # Cargar Ícono (pequeño y de acento)
        icon_small = Image.open(sec['icon']).convert("RGBA").resize((30, 30))
        # CORRECCIÓN DE ERROR: Todos los valores en img.paste deben ser enteros
        icon_x = bx + titulo_seccion_w + 10
        img.paste(icon_small, (icon_x, by + 5), icon_small)
        
        # 4.2. Dibujar el Contenido (Texto)
        content_y_start = by + 60
        
        # Utilizar draw.multiline_text para manejar los saltos de línea y el contenido
        draw.multiline_text((bx, content_y_start), 
                             sec['contenido'], 
                             fill=BLANCO, # Texto en blanco para contraste
                             font=FONT_CONTENIDO, 
                             spacing=15, 
                             align="left")


    # --- 5. Pie de página ---
    frase = "Date la Oportunidad, nosotros damos los Resultados"
    
    # Posición para el lema (centrado en la parte inferior)
    frase_bbox = draw.textbbox((0, 0), frase, font=FONT_FOOTER)
    frase_width = frase_bbox[2] - frase_bbox[0]
    frase_x = (width // 2) - (frase_width // 2) 
    
    draw.text((frase_x, height - 80), frase, fill=BLANCO, font=FONT_FOOTER)
    draw.text((width - 400, height - 80), "www.evolucioni3.com", fill=BLANCO, font=FONT_FOOTER.font_variant(size=25)) # URL de ejemplo


    # Guardar
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_png = OUTPUT_DIR / f"infografia_{timestamp}.png"
    path_pdf = OUTPUT_DIR / f"infografia_{timestamp}.pdf"
    
    # Guardar PNG 
    img.save(path_png) 
    
    # Guardar PDF
    img.convert("RGB").save(path_pdf, "PDF", resolution=100.0)

    print(f"✅ PNG generado en: {path_png}")
    print(f"✅ PDF generado en: {path_pdf}")
    return path_png, path_pdf


# Test independiente con la estructura de datos que me enviaste (lista de strings)
if __name__ == "__main__":
    test_data = {
        "proyecto": "Alcance para proveedores",
        "objetivo": "Integración de una plataforma digital para registro y seguimiento de proveedores.",
        "alcance": [
            "Implementación de un sistema de evaluación periódica basado en criterios de calidad.",
            "Puntualidad y costo-beneficio."
        ],
        "beneficios": [
            "Reducción de tiempos en procesos de compra y contratación.",
            "Mayor transparencia y trazabilidad en las relaciones con proveedores.",
            "Optimización de costos mediante evaluaciones objetivas y selección de mejores ofertas."
        ],
        "inversion": "Estimada: $450,000 MXN (desglose)",
        "tiempo": "6 meses (desarrollo, pruebas, capacitación y puesta en marcha)",
        "mantenimiento": "A partir del mes 7, con mejora continua.",
    }
    generate_infographic(test_data)