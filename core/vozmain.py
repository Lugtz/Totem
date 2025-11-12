from TTS.api import TTS
import sounddevice as sd
import soundfile as sf
import tempfile, os

# Inicializa el modelo solo una vez (fuera de la función)
modelo = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=modelo, progress_bar=False, gpu=True)


def hablar(texto: str, referencia_voz: str = "1.wav"):
    """
    Genera y reproduce audio con la voz clonada.
    
    Args:
        texto (str): Texto a convertir en voz.
        referencia_voz (str): Ruta al archivo de referencia de voz.
    """
    if not texto.strip():
        return
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        ruta_temp = temp_audio.name

    # Generar voz
    tts.tts_to_file(
        text=texto,
        file_path=ruta_temp,
        speaker_wav=referencia_voz,
        language="es-mx",
        split_sentences=True
    )

    # Reproducir
    data, sr = sf.read(ruta_temp)
    sd.play(data, sr)
    sd.wait()
    os.remove(ruta_temp)


# 🟢 Ejemplo de uso interactivo
if __name__ == "__main__":
    print("🟢 Escribe algo para decir con tu voz (o 'salir' para terminar):")
    while True:
        texto = input("👉 Texto: ").strip()
        if texto.lower() in ["salir", "exit"]:
            print("👋 Cerrando programa...")
            break
        hablar(texto)
