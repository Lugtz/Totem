# core/voz.py
# -*- coding: utf-8 -*-
import os, tempfile, winsound
import pyttsx3

def _elige_voz_latam(engine):
    # intenta seleccionar una voz latino/neutral si existe;
    # si no, se queda la default del sistema
    for v in engine.getProperty('voices'):
        vid = (v.id or "").lower()
        vname = (v.name or "").lower()
        if ("es-mx" in vid) or ("mexico" in vname) or ("latam" in vname) or ("es_419" in vid):
            engine.setProperty('voice', v.id)
            break

def synth_tts(texto: str) -> str:
    """Genera un WAV temporal con pyttsx3 (CPU) y devuelve la ruta del archivo."""
    if not texto:
        texto = " "
    engine = pyttsx3.init()
    _elige_voz_latam(engine)  # si no hay, usará la default instalada
    engine.setProperty('rate', 180)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = tmp.name
    tmp.close()
    engine.save_to_file(texto, wav_path)
    engine.runAndWait()
    return wav_path

def play_audio(wav_path: str):
    """Reproduce el WAV sin bloquear el proceso principal."""
    try:
        if os.path.exists(wav_path):
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            print(f"⚠️ No existe WAV: {wav_path}")
    except Exception as e:
        print(f"⚠️ Error reproduciendo audio: {e}")
