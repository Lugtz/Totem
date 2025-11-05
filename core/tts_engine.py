import threading, time, itertools
import pyttsx3

# TTS local con SAPI (Windows). Genera audio por la bocina del sistema.
# Exponemos un on_viseme callback sencillo con valores 0..1 simulados.

_engine = None
_lock = threading.Lock()
_counter = itertools.count(1)

def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            e = pyttsx3.init()  # SAPI en Windows
            e.setProperty("rate", 175)  # velocidad base
            _engine = e
    return _engine

def speak(text: str, rate_wpm: int = 160, on_viseme=None) -> str:
    utt_id = f"utt_{next(_counter)}"
    def _run():
        eng = _get_engine()
        # Ajustar velocidad si se pide
        try:
            eng.setProperty("rate", rate_wpm)
        except Exception:
            pass
        # Animación simple de mandíbula mientras hablamos (aprox por tiempo)
        # Duración aproximada:  (palabras / wpm) * 60
        words = max(1, len(text.split()))
        dur = (words / max(80, rate_wpm)) * 60.0
        started = time.time()

        def _anim():
            # Mueve mandíbula 10-12 veces por segundo (simulado)
            while time.time() - started < dur:
                if on_viseme:
                    on_viseme(0.75)
                time.sleep(0.05)
                if on_viseme:
                    on_viseme(0.15)
                time.sleep(0.035)
            if on_viseme:
                on_viseme(0.0)

        t2 = threading.Thread(target=_anim, daemon=True)
        t2.start()
        try:
            eng.say(text)
            eng.runAndWait()
        except Exception:
            pass
        finally:
            if on_viseme:
                on_viseme(0.0)

    threading.Thread(target=_run, daemon=True).start()
    return utt_id
