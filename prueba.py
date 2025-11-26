# -*- coding: utf-8 -*-
"""
test_tts_nacho.py

Prueba rápida de core.tts_engine:
- Genera audio con ElevenLabs
- Lo reproduce
- Manda el texto a Nacho por http://localhost:7000/<texto>
"""

from core.tts_engine import speak


def main():
    # Puedes cambiar este texto de prueba
    texto = "Hola, soy Nacho. Esta es una prueba del motor de voz y de la animación."
    speak(texto)


if __name__ == "__main__":
    main()
