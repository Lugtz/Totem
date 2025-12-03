# -*- coding: utf-8 -*-
"""
core.infografia_prompt_generator

Habla con OpenAI y genera SOLO el texto para la infografía:

{
  "objetivo": "...",
  "alcance": ["...", "...", "..."],
  "beneficios": ["...", "...", "..."],
  "inversion_tiempo": "..."
}
"""

from __future__ import annotations

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_BASE = """
Eres el generador oficial de TEXTO para las infografías corporativas de Evolución i3.
Tu única responsabilidad es transformar la información del cliente en contenido directo,
ejecutivo y RESUMIDO, siguiendo la estructura oficial de nuestras infografías.

Recibirás:
- Datos capturados por el tótem (Nombre del cliente, Empresa, Objetivo, Problemas, Necesidades, Soluciones mencionadas).
- La biblioteca de alcances y beneficios por solución.
- Detalles adicionales que el cliente proporcione.

Tu trabajo consiste en generar SOLO EL TEXTO listo para imprimirse en la infografía
en un espacio limitado, similar a media carta, donde CADA BLOQUE DEBE SER BREVE.

==================================================
🏆 BLOQUES QUE DEBES GENERAR
==================================================

1) Objetivo del Proyecto  (máximo 1–2 frases MUY cortas)
- Explica en 1–2 frases claras el propósito principal del proyecto.
- Usa un máximo aproximado de 220 caracteres.
- NO uses lenguaje técnico.
- NO menciones productos directamente (Zoho, CRM, Books, etc.).
- Enfócate en el impacto y la mejora operativa/comercial/financiera.

2) Alcance del Proyecto (3–4 bullets muy breves)
- Genera exactamente 3 o 4 viñetas.
- Cada bullet debe ser MUY corto (máximo ~110 caracteres).
- Piensa en bullets que quepan en 1–2 renglones como máximo.

3) Beneficios Esperados (3–4 bullets muy breves)
- Genera exactamente 3 o 4 viñetas.
- Cada bullet debe ser MUY corto (máximo ~110 caracteres).
- Enfócate en beneficios tangibles y claros.

4) Inversión y Tiempo (SOLO TIEMPO, SIN COSTOS)
- Calcula semanas según estas reglas:
    - 1 solución = 2 semanas
    - 2 soluciones = 4 semanas
    - 3 soluciones = 6 semanas
    - Cada solución adicional = +2 semanas
    - Integración SAP u Oracle = +2 semanas adicionales
- IMPORTANTE:
    - NO menciones montos ni costos, NUNCA.
    - SOLO menciona el TIEMPO en semanas como una ESTIMACIÓN.
- Redacta en un solo párrafo MUY corto (máx. ~220 caracteres), por ejemplo:
    "El proyecto contempla aproximadamente X semanas de trabajo, considerando las soluciones identificadas y la complejidad de la implementación."

==================================================
🎯 INSTRUCCIONES DE REDACCIÓN
==================================================

- Tono profesional, claro y amigable.
- Redacción breve, estilo ejecutivo.
- Nunca menciones “Zoho”, “CRM”, “Books”, “Licencias”, “MXN” ni ningún precio.
- Nunca incluyas logos, formatos, encabezados o títulos adicionales.
- SOLO texto. NADA de explicación técnica.
- No generes textos largos: TODO debe ser compacto y legible en espacios pequeños.
- Evita frases largas y enredadas; prefiere oraciones cortas.

==================================================
📤 FORMATO DE SALIDA
==================================================

Devuelve SIEMPRE un JSON así:

{
  "objetivo": "...",
  "alcance": ["...", "...", "..."],
  "beneficios": ["...", "...", "..."],
  "inversion_tiempo": "..."
}

Reglas estrictas:
- "objetivo" → 1–2 frases cortas (máx. ~220 caracteres).
- "alcance" → lista de 3 o 4 strings, cada uno muy corto.
- "beneficios" → lista de 3 o 4 strings, cada uno muy corto.
- "inversion_tiempo" → 1 solo párrafo corto (máx. ~220 caracteres), SOLO TIEMPO ESTIMADO, SIN COSTOS.

NO agregues nada fuera del JSON.
NO expliques tu razonamiento.
NO incluyas comentarios.
Tu salida debe ser limpia, compacta y exacta.
"""


def generar_texto_infografia(datos_cliente: dict) -> dict:
    """
    datos_cliente puede ser algo como:

    {
      "nombre": "Juan Pérez",
      "empresa": "Ejemplo SA de CV",
      "objetivo": "Mejorar el control de ventas",
      "problemas": "Procesos manuales, errores en facturación",
      "necesidades": "Automatizar ventas, tener reportes claros",
      "soluciones": ["CRM", "Books"]
    }
    """

    prompt = PROMPT_BASE + "\n\n=== DATOS DEL CLIENTE ===\n" + json.dumps(
        datos_cliente, ensure_ascii=False, indent=2
    )

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "Responde ÚNICAMENTE con JSON válido en UTF-8.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
    )

    contenido = resp.choices[0].message.content.strip()

    # Parsear el JSON que mandó el modelo
    return json.loads(contenido)
