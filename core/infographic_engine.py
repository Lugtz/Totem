from PIL import Image, ImageDraw, ImageFont
import os, datetime

def _font(size):
    try:
        return ImageFont.truetype("templates/infographic/fonts/CenturyGothic.ttf", size)
    except:
        return ImageFont.load_default()

def generate(slots: dict):
    os.makedirs("data/tmp", exist_ok=True)
    W, H = 1900, 900  # 19:9
    img = Image.new("RGB", (W, H), (16, 28, 40))
    d = ImageDraw.Draw(img)

    title = "Proyecto Evolución i3"
    subtitle = f"{slots.get('Empresa','Empresa')} • {slots.get('Solucion','Solución')}"

    d.text((60, 40), title, (255,255,255), font=_font(56))
    d.text((60, 110), subtitle, (180,220,255), font=_font(28))

    y, x, colw, rowh = 200, 60, (W-120)//2, 260
    blocks=[("Objetivo", slots.get("Objetivo","Digitalizar procesos con Zoho One")),
            ("Alcance", slots.get("Alcance","CRM, Desk, Forms, Analytics")),
            ("Beneficios", slots.get("Beneficios","Trazabilidad, automatización, reportes")),
            ("Inversión y tiempo", f"{slots.get('Inversion','-')} {slots.get('Moneda','')} • {slots.get('Duracion','-')} semanas")]

    for i,(hdr,txt) in enumerate(blocks):
        bx = x + (i%2)*colw
        by = y + (i//2)*rowh
        d.rounded_rectangle([bx,by,bx+colw-20,by+rowh-20], radius=24, outline=(70,120,180), width=2, fill=(24,40,60))
        d.text((bx+24, by+18), hdr, (255,255,255), font=_font(30))
        d.text((bx+24, by+70), str(txt), (210,225,240), font=_font(24))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png = f"data/tmp/infografia_{ts}.png"
    img.save(png)
    pdf = f"data/tmp/infografia_{ts}.pdf"
    img.convert("RGB").save(pdf, "PDF", resolution=150.0)
    return png, pdf