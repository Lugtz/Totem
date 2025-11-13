# -*- coding: utf-8 -*-
# facial_audio_visemes_bgcard_const_select_output_calib_shoulders_static_plusvar.py
# - Visemas desde audio (loopback/mic) con:
#   * Cierre real en BMP/FV
#   * Más variedad de visemas (alófonos) y coarticulación
#   * Transiciones más rápidas (ataque/decay) y micro-cierres entre sílabas
#   * Hombros FIJOS (pose oficial), sin animación
#
# Controles:
#   [ ]  -> bajar/subir intensidad general
#   k/l  -> bajar/subir mouth_boost (apertura)
#   g/h  -> bajar/subir ganancia de entrada
#   c    -> calibrar ruido ambiente
#   rueda -> zoom | click izq+arrastrar -> orbitar

import sys, os, math, time, queue, random
import numpy as np
from collections import deque

# =======================
#   CONSTANTES EDITABLES
# =======================
GLB_PATH  = "nacho.glb"
BG_PATH   = "fondo.png"

AUDIO_MODE   = "loopback"   # "loopback" o "mic"
AUDIO_SR     = 48000
AUDIO_BLOCK  = 1024         # ~21 ms @ 48k → ~47 Hz de análisis

# --- Selección de SALIDA (solo loopback/WASAPI) ---
LIST_AUDIO_DEVICES = True
AUDIO_OUTPUT_INDEX = 22
AUDIO_OUTPUT_NAME  = None

# --- Selección de MIC (modo "mic" o fallback) ---
MIC_INPUT_INDEX = None
MIC_INPUT_NAME  = None

# --- Ganancia de entrada inicial ---
INPUT_GAIN = 2.6  # un poco menor para reducir "boca abierta" por ruido

# --- Detección "hablando" (RMS) ---
RMS_TALK_THRESHOLD = 0.020
RMS_AVG_BLOCKS     = 3

# --- Curva de apertura (logística) y suavizado ---
OPEN_GATE_MULT  = 1.10   # más exigente para abrir
OPEN_DENOM_MULT = 6.0
OPEN_SIG_ALPHA  = 3.2
OPEN_SIG_BIAS   = 0.58   # sesgo a permanecer más cerrado
OPEN_EMA_ALPHA  = 0.28

# --- Dinámica de visemas / naturalidad ---
ATTACK_ALPHA    = 0.55   # sube rápido al nuevo visema
RELEASE_ALPHA   = 0.25   # baja más suave
HOLD_MIN_MS_ACTIVE = 45  # sostén mínimo cuando hay energía
HOLD_MIN_MS_SOFT  = 70   # sostén mínimo cuando hay poca energía
HOLD_MAX_MS       = 220  # nunca sostener más que esto (evita "pocos por segundo")
MICRO_CLOSURE_DUR = 0.060  # cierre BMP corto entre sílabas
MICRO_CLOSURE_COOLDOWN = 0.22
VARIANT_PROB     = 0.15  # prob. de alófono para variación natural
SIB_DEESS_JAW    = 0.75  # reduce mandíbula en sibilantes (S/SH/TS)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------- AUDIO (sounddevice) ----------
try:
    import sounddevice as sd
    _HAS_SD = True
except Exception:
    sd = None
    _HAS_SD = False

def _list_wasapi_outputs():
    try:
        apis = sd.query_hostapis()
        wasapi_idx = next((i for i, a in enumerate(apis) if "WASAPI" in a.get("name","").upper()), None)
        if wasapi_idx is None:
            print("⚠ No se encontró hostapi WASAPI.")
            return
        devs = sd.query_devices()
        print("\n🔊 Dispositivos WASAPI (salida) disponibles:")
        for i, d in enumerate(devs):
            if d.get("hostapi") == wasapi_idx and d.get("max_output_channels", 0) > 0:
                print(f"  [{i:02d}] {d['name']}  | ch_out={d['max_output_channels']}  sr_def={int(d.get('default_samplerate',0))}")
        print("")
    except Exception as e:
        print(f"⚠ No se pudo enumerar dispositivos WASAPI: {e}")

def _pick_device_by_name_or_index(kind, prefer_index=None, prefer_name_substr=None):
    devs = sd.query_devices()

    if kind == 'output':
        apis = sd.query_hostapis()
        wasapi_idx = next((i for i,a in enumerate(apis) if "WASAPI" in a.get("name","").upper()), None)
        if wasapi_idx is None:
            raise RuntimeError("WASAPI no disponible en este sistema")
        if prefer_index is not None and 0 <= prefer_index < len(devs):
            d = devs[prefer_index]
            if d.get("hostapi") == wasapi_idx and d.get("max_output_channels",0) > 0:
                return prefer_index, d
            raise RuntimeError(f"El índice {prefer_index} no es salida WASAPI válida.")
        if prefer_name_substr:
            key = prefer_name_substr.lower()
            for i, d in enumerate(devs):
                if d.get("hostapi") == wasapi_idx and d.get("max_output_channels",0) > 0:
                    if key in d.get("name","").lower():
                        return i, d
        out_idx = apis[wasapi_idx].get("default_output_device", -1)
        if out_idx is not None and out_idx >= 0:
            d = devs[out_idx]
            if d.get("max_output_channels",0) > 0:
                return out_idx, d
        for i, d in enumerate(devs):
            if d.get("hostapi") == wasapi_idx and d.get("max_output_channels",0) > 0:
                return i, d
        raise RuntimeError("No hay dispositivos de salida WASAPI disponibles.")

    elif kind == 'input':
        if prefer_index is not None and 0 <= prefer_index < len(devs):
            d = devs[prefer_index]
            if d.get("max_input_channels",0) > 0:
                return prefer_index, d
            raise RuntimeError(f"El índice {prefer_index} no es entrada válida.")
        if prefer_name_substr:
            key = prefer_name_substr.lower()
            for i, d in enumerate(devs):
                if d.get("max_input_channels",0) > 0 and key in d.get("name","").lower():
                    return i, d
        try:
            d = sd.query_devices(kind="input")
            default_name = d.get("name","")
            for i, di in enumerate(devs):
                if di.get("name","") == default_name and di.get("max_input_channels",0) > 0:
                    return i, di
        except Exception:
            pass
        for i, d in enumerate(devs):
            if d.get("max_input_channels",0) > 0:
                return i, d
        raise RuntimeError("No hay dispositivos de entrada disponibles.")
    else:
        raise ValueError("kind debe ser 'output' o 'input'")

class AudioVisemeDriver:
    def __init__(self, mode="loopback", samplerate=None, blocksize=1024,
                 prefer_out_index=None, prefer_out_name=None,
                 prefer_mic_index=None, prefer_mic_name=None,
                 init_gain=INPUT_GAIN):
        if not _HAS_SD:
            raise RuntimeError("sounddevice no instalado. `pip install sounddevice`")

        self.mode = mode
        self.blocksize = int(blocksize)
        self.running = False
        self.q = queue.Queue(maxsize=16)  # mayor buffer
        self.last = {"key":"REST", "open":0.0, "ts":time.time()}
        self._noise_ema = 1e-3
        self._rms_ema   = 1e-3
        self._sr = None
        self._stream = None
        self._have_loopback = False
        self._loopback_channels = 1
        self.gain = float(init_gain)

        self._talk_hist = deque(maxlen=RMS_AVG_BLOCKS)
        self.is_talking = False
        self._last_key  = "REST"
        self._open_ema  = 0.0

        # control de tasa / cierres
        self._hold_until = 0.0
        self._micro_closure_until = 0.0
        self._last_change_ts = 0.0
        self._last_rms = 0.0
        self._last_energy_drop_ts = 0.0

        try:
            if self.mode == "loopback":
                ok_stereo = False
                try:
                    ok_stereo = self._open_stereo_mix_loopback(samplerate)
                except Exception as e:
                    print(f"⚠ No se pudo usar 'Stereo Mix' como loopback: {e}")

                if not ok_stereo:
                    if LIST_AUDIO_DEVICES:
                        _list_wasapi_outputs()
                    dev_index, dev_info = _pick_device_by_name_or_index('output', prefer_out_index, prefer_out_name)
                    ch_out = int(dev_info.get("max_output_channels", 2)) or 2
                    self._loopback_channels = max(1, ch_out)
                    self._sr = int(dev_info.get("default_samplerate", 48000)) if not samplerate else int(samplerate)
                    print(f"🎚 Usando WASAPI output idx={dev_index}: {dev_info['name']} (ch_out={self._loopback_channels}, sr_def={int(dev_info.get('default_samplerate',0))})")

                    ws = None
                    try:
                        ws = sd.WasapiSettings(loopback=True)
                    except Exception as e:
                        print(f"⚠ WasapiSettings(loopback=True) no disponible: {e}")
                        ws = None

                    if ws is not None:
                        try:
                            self._stream = sd.InputStream(
                                samplerate=self._sr,
                                blocksize=self.blocksize,
                                dtype="float32",
                                channels=self._loopback_channels,
                                device=dev_index,
                                extra_settings=ws,
                                callback=self._cb
                            )
                            self._have_loopback = True
                        except Exception as e:
                            print(f"⚠ Falló abrir loopback WASAPI: {e}")

                    if not self._have_loopback:
                        print("⚠ Loopback no disponible. Cambiando a MIC.")
                        self._open_mic(prefer_mic_index, prefer_mic_name, samplerate)

            elif self.mode == "mic":
                self._open_mic(prefer_mic_index, prefer_mic_name, samplerate)
            else:
                self._stream = None

        except Exception as e:
            print(f"⚠ Error en configuración de audio ({e}). Intentando mic genérico ...")
            try:
                self._open_mic(None, None, samplerate)
            except Exception as e2:
                raise RuntimeError(f"Sin loopback ni mic disponibles: {e2}") from e

    def _open_stereo_mix_loopback(self, samplerate):
        devs = sd.query_devices()
        loopback_idx = None
        for i, dev in enumerate(devs):
            name = (dev.get("name","") or "").lower()
            if dev.get("max_input_channels",0) > 0 and ("stereo mix" in name or "stereomix" in name or "what u hear" in name or "mix" in name):
                loopback_idx = i
                print(f"🎧 Loopback por 'Stereo Mix' idx={i}: {dev['name']}")
                break

        if loopback_idx is None:
            print("⚠ No se encontró dispositivo tipo 'Stereo Mix'.")
            return False

        dev_info = devs[loopback_idx]
        ch_in = max(1, int(dev_info.get("max_input_channels",1)))
        self._sr = int(dev_info.get("default_samplerate", 48000)) if not samplerate else int(samplerate)
        self._stream = sd.InputStream(
            samplerate=self._sr,
            blocksize=self.blocksize,
            dtype="float32",
            channels=min(ch_in, 2),
            device=loopback_idx,
            callback=self._cb
        )
        self.mode = "loopback"
        self._have_loopback = True
        return True

    def _open_mic(self, prefer_mic_index, prefer_mic_name, samplerate):
        dev_index, dev_info = _pick_device_by_name_or_index('input', prefer_mic_index, prefer_mic_name)
        self._sr = int(dev_info.get("default_samplerate", 48000)) if not samplerate else int(samplerate)
        ch_in = max(1, int(dev_info.get("max_input_channels", 1)))
        print(f"🎙 Usando MIC idx={dev_index}: {dev_info['name']} (ch_in={ch_in}, sr_def={int(dev_info.get('default_samplerate',0))})")
        self._stream = sd.InputStream(
            samplerate=self._sr,
            blocksize=self.blocksize,
            dtype="float32",
            channels=min(ch_in, 2),
            device=dev_index,
            callback=self._cb
        )
        self.mode = "mic"

    @property
    def samplerate(self):
        return self._sr

    def start(self):
        if self._stream and not self.running:
            self._stream.start()
            self.running = True

    def stop(self):
        if self._stream and self.running:
            self._stream.stop()
            self.running = False

    def has_data(self):
        return not self.q.empty()

    def pop_latest(self):
        if self.q.empty():
            return self.last
        d = self.last
        while not self.q.empty():
            try: d = self.q.get_nowait()
            except queue.Empty: break
        self.last = d
        return d

    def set_gain(self, g):
        self.gain = float(clamp(g, 0.2, 20.0))
        print(f"🔈 INPUT_GAIN = {self.gain:.2f}")

    def calibrate_quick(self):
        self._noise_ema = max(self._noise_ema, 1.2 * self._rms_ema)
        print(f"🧰 Calibración rápida: noise_ema = {self._noise_ema:.6f} | rms_ema = {self._rms_ema:.6f}")

    # ---- Clasificador de visemas con variedad y control de tasa ----
    def _classify(self, x, sr):
        # FFT
        win = np.hanning(x.size).astype(np.float32)
        X = np.fft.rfft(x * win)
        mag = np.abs(X) + 1e-12
        freqs = np.fft.rfftfreq(x.size, d=1.0/sr)
        total_mag = float(np.sum(mag))
        if total_mag <= 0.0:
            total_mag = 1e-12

        centroid = float(np.sum(freqs * mag) / total_mag)
        sgn = np.sign(x)
        zcr = float(np.mean(np.abs(np.diff(sgn)) > 0))
        flatness = float(np.exp(np.mean(np.log(mag))) / (np.mean(mag) + 1e-12))
        flatness = float(clamp(flatness, 0.0, 1.0))
        cum = np.cumsum(mag)
        thr = 0.85 * cum[-1]
        idx = int(np.searchsorted(cum, thr))
        rolloff85 = float(freqs[min(idx, len(freqs)-1)])

        def band_ratio(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.sum(mag[mask]) / total_mag) if np.any(mask) else 0.0

        r_low  = band_ratio(0, 800)
        r_mid  = band_ratio(800, 2000)
        r_high = band_ratio(2000, min(sr*0.45, 6000))

        # Apertura estimada (logística)
        rms = float(np.sqrt(np.mean(x*x) + 1e-12))
        self._rms_ema = 0.90*self._rms_ema + 0.10*rms
        if rms < 3.0*self._noise_ema:
            self._noise_ema = 0.995*self._noise_ema + 0.005*rms
        self._noise_ema = max(self._noise_ema, 1e-5)

        self._talk_hist.append(rms)
        rms_prom = float(np.mean(self._talk_hist)) if len(self._talk_hist)>0 else rms
        self.is_talking = (rms_prom >= RMS_TALK_THRESHOLD)

        denom = max(self._noise_ema * OPEN_DENOM_MULT, 1e-6)
        norm  = max(0.0, rms - OPEN_GATE_MULT * self._noise_ema) / denom
        open_lin = float(clamp(norm, 0.0, 1.5))
        mouth_open = 1.0 / (1.0 + math.exp(-OPEN_SIG_ALPHA * (open_lin - OPEN_SIG_BIAS)))
        if not self.is_talking:
            mouth_open *= 0.12  # más cerrado cuando no habla

        # micro-cierres entre sílabas (detecta caídas de energía)
        now = time.time()
        if rms < self._last_rms * 0.55 and (now - self._last_change_ts) > 0.08:
            self._last_energy_drop_ts = now
        self._last_rms = rms

        # Clasificación base
        if mouth_open < 0.06:
            key = "BMP"
        else:
            if r_low > 0.48 and (r_low - max(r_mid, r_high)) > 0.08:
                key = "O" if centroid < 850 else "U"
            elif r_mid > 0.40 and (r_mid - max(r_low, r_high)) > 0.05:
                # separa A/E y agrega alófonos
                if rolloff85 < 1600:
                    key = "AH"    # A más central/oscura
                elif rolloff85 < 1900:
                    key = "AE"    # entre A y E
                else:
                    key = "E"
            elif r_high > 0.40:
                # fricativas/altas
                if zcr > 0.30 and flatness > 0.60 and rolloff85 > 3200:
                    key = "SH"   # sibilante ancha
                elif zcr > 0.28 and rolloff85 > 3000:
                    key = "S"
                else:
                    key = "I"
            else:
                # fallback por centroid
                if   centroid < 900:  key = "O"
                elif centroid < 1300: key = "OU"  # O→U transición
                elif centroid < 1700: key = "AE"
                elif centroid < 2300: key = "E"
                elif centroid < 3200: key = "IH"  # I más relajada
                else:                 key = "S"

        # cierres y retención mínima
        min_hold = HOLD_MIN_MS_ACTIVE if self.is_talking else HOLD_MIN_MS_SOFT
        if now < self._hold_until:
            key = self._last_key
        else:
            self._hold_until = now + (min_hold/1000.0)

        # inserta micro-cierre BMP si hay caída de energía reciente
        if (now - self._last_energy_drop_ts) < MICRO_CLOSURE_DUR and (now - self._last_change_ts) > MICRO_CLOSURE_COOLDOWN:
            key = "BMP"

        # alófonos/variación
        key = self._maybe_variant(key, r_low, r_mid, r_high, centroid, rolloff85, flatness, zcr)

        # hysteresis de cierre/evitar rebotes
        if mouth_open < 0.10:
            key = "BMP"
        elif self._last_key != key and mouth_open < 0.15:
            key = self._last_key

        changed = (key != self._last_key)
        if changed:
            self._last_change_ts = now
            self._hold_until = min(self._hold_until + (HOLD_MIN_MS_ACTIVE/1000.0), now + (HOLD_MAX_MS/1000.0))
        self._last_key = key

        self._open_ema = (1.0 - OPEN_EMA_ALPHA) * self._open_ema + OPEN_EMA_ALPHA * mouth_open
        return key, mouth_open

    def _maybe_variant(self, key, r_low, r_mid, r_high, centroid, rolloff85, flatness, zcr):
        if random.random() > VARIANT_PROB:
            return key
        # vecinos por similaridad para mayor naturalidad
        neighbors = {
            "A": ["AH","AE","DT"],
            "AH":["A","AE"],
            "AE":["A","E"],
            "E": ["EH","AE","I"],
            "EH":["E","I"],
            "I": ["IH","E"],
            "IH":["I","E"],
            "O": ["OE","OU","U"],
            "OE":["O","E"],
            "OU":["O","U"],
            "U": ["OU","O","W"],
            "S": ["SH","TS"],
            "SH":["S","CH"],
            "CH":["TS","S"],
            "DT":["L","R"],
            "R": ["RR","DT"],
            "RR":["R","DT"],
            "W": ["U","O","OU"],
            "Y": ["I","E"],
            "FV":["BMP","S"]
        }
        cand = neighbors.get(key, None)
        if not cand: return key
        # pesa por rasgos espectrales (sibilancia, vocal abierta/cerrada)
        if key in ("S","SH","TS"):
            return "SH" if (r_high>0.45 and flatness>0.6) else random.choice(cand)
        if key in ("O","U","OU"):
            return "OU" if rolloff85>1400 else ("O" if centroid<850 else "U")
        if key in ("E","I","IH","EH","AE"):
            if centroid>2400: return "I"
            if centroid>2000: return "EH"
            if centroid>1600: return "E"
            return "AE"
        return random.choice(cand)

    def _cb(self, indata, frames, time_info, status):
        if status:
            pass
        if indata is None or len(indata) == 0:
            return
        arr = np.asarray(indata, dtype=np.float32)
        x = arr.mean(axis=1) if (arr.ndim == 2 and arr.shape[1] > 1) else arr.reshape(-1)
        x = x * self.gain
        if x.size == 0:
            return

        key, mouth_open = self._classify(x, self._sr or AUDIO_SR)

        d = {"key": key, "open": mouth_open, "ts": time.time()}
        try:
            self.q.put_nowait(d)
        except queue.Full:
            try: _ = self.q.get_nowait()
            except queue.Empty: pass
            try: self.q.put_nowait(d)
            except queue.Full: pass

# ---------- Panda3D ----------
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.task import Task
from panda3d.core import (
    ClockObject, AmbientLight, DirectionalLight, VBase4,
    NodePath, TextNode, AntialiasAttrib, CardMaker, TransparencyAttrib
)
globalClock = ClockObject.getGlobalClock()

class TextVisemeDemo(ShowBase):
    def __init__(self, glb_path, bg_path=None, audio_mode=AUDIO_MODE, sr=AUDIO_SR, block=AUDIO_BLOCK):
        ShowBase.__init__(self)
        self.setBackgroundColor(0.15, 0.15, 0.15)
        self.disableMouse()
        self.render.setAntialias(AntialiasAttrib.MAuto)

        try:
            import simplepbr
            simplepbr.init()
            print("✓ simplepbr activo")
        except Exception as e:
            print(f"⚠ simplepbr no disponible ({e}). Uso ShaderAuto")
            self.render.setShaderAuto()

        self.camera.setPos(0, -5, 2)
        self.camera.lookAt(0, 0, 1.5)

        amb = AmbientLight("amb"); amb.setColor(VBase4(0.35,0.35,0.35,1))
        key = DirectionalLight("key"); key.setColor(VBase4(0.85,0.85,0.85,1))
        fill = DirectionalLight("fill"); fill.setColor(VBase4(0.25,0.25,0.25,1))
        self.render.setLight(self.render.attachNewNode(amb))
        key_np = self.render.attachNewNode(key); key_np.setHpr(35,-55,0); self.render.setLight(key_np)
        fill_np = self.render.attachNewNode(fill); fill_np.setHpr(-60,-20,0); self.render.setLight(fill_np)

        self.bg_card = None
        self.bg_tex = None
        self._bg_dist = 12.0
        if bg_path:
            self._setup_background_card(bg_path)
            self.accept("window-event", self._on_window_event)

        try:
            self.actor = Actor(glb_path)
        except Exception as e:
            print(f"Error al cargar modelo: {e}\n(Instala: pip install panda3d-gltf)")
            sys.exit(1)
        if not self.actor.getNode(0):
            print(f"\n❌ ERROR: No se pudo cargar '{glb_path}'.")
            sys.exit(1)
        try:
            self.actor.clearColor(); self.actor.clearColorScale()
        except Exception:
            pass
        self.actor.reparentTo(self.render)

        # HUD
        self.status_display = TextNode('status_display')
        try:
            font = self.loader.loadFont('cmss12')
            if font: self.status_display.setFont(font)
        except Exception:
            pass
        self.status_display.setAlign(TextNode.A_left)
        self.status_node = self.aspect2d.attachNewNode(self.status_display)
        self.status_node.setScale(0.07)
        self.status_node.setPos(-self.getAspectRatio()+0.1, 0, 0.9)

        # Escalas / perillas
        self.MOVEMENT_SCALE = 1.0
        self.intensity   = 0.60
        self.mouth_boost = 1.45  # menos "boca abierta" por defecto

        # Atenuaciones
        self.jaw_atten  = 0.60 * self.MOVEMENT_SCALE
        self.chin_atten = 0.60 * self.MOVEMENT_SCALE

        # ====== HUESOS ======
        self.upper_L = self._find(["lip.T.L","lip.T.L.001"])
        self.upper_R = self._find(["lip.T.R","lip.T.R.001"])
        self.lower_L = self._find(["lip.B.L","lip.B.L.001"])
        self.lower_R = self._find(["lip.B.R","lip.B.R.001"])
        self.jaw     = self._find(["jaw"])
        self.tongue  = self._find(["tongue","tongue.001","tongue.002"])
        self.teeth_T = self._find(["teeth.T"])
        self.teeth_B = self._find(["teeth.B"])
        self.chin_center = self._find(["chin"])
        self.chin_L      = self._find(["chin.L"])
        self.chin_R      = self._find(["chin.R"])
        self.cheek_B_L   = self._find(["cheek.B.L"])
        self.cheek_B_R   = self._find(["cheek.B.R"])
        self.jaw_L_001   = self._find(["jaw.L.001"])
        self.jaw_R_001   = self._find(["jaw.R.001"])
        self.brow_L = self._find(["brow.B.L","brow.T.L","brow.B.L.001","brow.T.L.001"])
        self.brow_R = self._find(["brow.B.R","brow.T.R","brow.B.R.001","brow.T.R.001"])
        self.upper_arm_L = self._find(["upper_arm.L"])
        self.upper_arm_R = self._find(["upper_arm.R"])

        # Bases
        self.base_hpr = {}
        self.base_pos = {}
        for group in [self.upper_L,self.upper_R,self.lower_L,self.lower_R,
                      self.jaw,self.tongue,self.teeth_T,self.teeth_B,
                      self.chin_center,self.chin_L,self.chin_R,
                      self.cheek_B_L,self.cheek_B_R,
                      self.jaw_L_001,self.jaw_R_001,
                      self.brow_L,self.brow_R,
                      self.upper_arm_L,self.upper_arm_R]:
            for name,j in group:
                if name not in self.base_hpr: self.base_hpr[name] = j.getHpr()
                if name not in self.base_pos: self.base_pos[name] = j.getPos()

        # ====== HOMBROS FIJOS (sin animación) - POSE OFICIAL ======
        self.SHO_L_ROT_X = 0.0
        self.SHO_L_ROT_Y = 10.0
        self.SHO_L_ROT_Z = 80.0
        self.SHO_L_SIGN_X = +1.0
        self.SHO_L_SIGN_Y = +1.0
        self.SHO_L_SIGN_Z = +1.0

        self.SHO_R_ROT_X = 0.0
        self.SHO_R_ROT_Y = 10.0
        self.SHO_R_ROT_Z = -80.0
        self.SHO_R_SIGN_X = +1.0
        self.SHO_R_SIGN_Y = +1.0
        self.SHO_R_SIGN_Z = +1.0

        self._set_shoulders_static()

        # ====== PÁRPADOS (anillos) ======
        top_L_names = ["lid.T.L.003","lid.T.L.002","lid.T.L.001","lid.T.L"]
        bot_L_names = ["lid.B.L.003","lid.B.L.002","lid.B.L.001","lid.B.L"]
        top_R_names = [n.replace(".L", ".R") for n in top_L_names]
        bot_R_names = [n.replace(".L", ".R") for n in bot_L_names]
        lid_weights = [0.35, 0.60, 0.85, 1.00]

        def _bind_with_weights(names, weights):
            out=[]
            for i, n in enumerate(names):
                pair = self._find([n])
                if pair:
                    name, j = pair[0]
                    out.append((name, j, weights[i]))
                    if name not in self.base_hpr: self.base_hpr[name] = j.getHpr()
                    if name not in self.base_pos: self.base_pos[name] = j.getPos()
            return out

        self.lid_top_L = _bind_with_weights(top_L_names, lid_weights)
        self.lid_top_R = _bind_with_weights(top_R_names, lid_weights)
        self.lid_bot_L = _bind_with_weights(bot_L_names, lid_weights)
        self.lid_bot_R = _bind_with_weights(bot_R_names, lid_weights)

        # Parámetros párpados
        self.lid_top_z_gain = 0.0032
        self.lid_top_max_dz = 0.0060
        self.lid_bot_z_gain = 0.0016
        self.lid_bot_max_dz = 0.0035
        self.lid_top_x_gain = 0.0020
        self.lid_top_max_dx = 0.0030
        self.lid_bot_x_gain = 0.0012
        self.lid_bot_max_dx = 0.0020
        self.squeeze_threshold = 0.70
        self.squeeze_roll_gain = 0.6 * self.MOVEMENT_SCALE
        self.squeeze_cheek_dz  = 0.0012 * self.MOVEMENT_SCALE

        # ====== TABLA VISEMAS (con 'seal' para cerrar labios) ======
        self.VTABLE = self._build_viseme_table()

        # Ganancias labios/mandíbula
        self.pitch_gain = 5.2
        self.roll_gain  = 4.0
        self.top_ratio  = 1.65
        self.bot_ratio  = 1.25
        self.lip_z_gain = 0.0028
        self.max_lip_dz = 0.018
        self.jaw_z_gain = 0.0032
        self.max_jaw_dz = 0.016
        self.teeth_bot_follow = 1.00
        self.close_dz_gain  = 0.0022
        self.max_close_dz   = 0.008

        # Fuerza extra de sellado
        self.seal_lip_gain = 0.0065 * self.MOVEMENT_SCALE
        self.seal_jaw_mult = 0.90

        # AUDIO
        self.audio = None
        if _HAS_SD:
            try:
                self.audio = AudioVisemeDriver(
                    mode=audio_mode,
                    samplerate=sr,
                    blocksize=block,
                    prefer_out_index=AUDIO_OUTPUT_INDEX,
                    prefer_out_name=AUDIO_OUTPUT_NAME,
                    prefer_mic_index=MIC_INPUT_INDEX,
                    prefer_mic_name=MIC_INPUT_NAME,
                    init_gain=INPUT_GAIN
                )
                self.audio.start()
                mode_txt = "loopback" if self.audio._have_loopback and self.audio.mode=="loopback" else self.audio.mode
                print(f"🎧 Audio mode: {mode_txt} @ {self.audio.samplerate} Hz (block {block})")
            except Exception as e:
                print(f"⚠ Audio deshabilitado: {e}")
                self.audio = None
        else:
            print("⚠ sounddevice no disponible: sin audio.")

        # Controles y cámara
        self.accept("wheel_up", self._zoom_in)
        self.accept("wheel_down", self._zoom_out)
        self.accept("mouse1", self._start_rotate)
        self.accept("mouse1-up", self._stop_rotate)
        self.accept("escape", sys.exit)
        self.accept("[", lambda: self._set_intensity(self.intensity - 0.05))
        self.accept("]", lambda: self._set_intensity(self.intensity + 0.05))
        self.accept("k", lambda: self._set_mouth_boost(self.mouth_boost - 0.05))
        self.accept("l", lambda: self._set_mouth_boost(self.mouth_boost + 0.05))
        self.accept("g", lambda: self._set_gain_rel(-0.2))
        self.accept("h", lambda: self._set_gain_rel(+0.2))
        self.accept("c", self._calibrate_noise)
        self.is_rotating=False; self.last_mouse=(0,0)
        self.cam_dist=5; self.cam_angle_x=15; self.cam_angle_y=0

        self.taskMgr.add(self._update_camera, "camera")
        self.taskMgr.add(self._animate, "animate")

        self._update_status(prefix=f"GLB: {os.path.basename(glb_path)} | Fondo: {os.path.basename(bg_path) if bg_path else 'N/A'}")

    # ----- Fondo card pegado a cámara -----
    def _setup_background_card(self, image_path: str):
        if not os.path.exists(image_path):
            print(f"⚠ No se encontró el fondo: {image_path}")
            return
        self.bg_tex = self.loader.loadTexture(image_path)
        if not self.bg_tex:
            print(f"❌ No se pudo cargar textura: {image_path}")
            return

        cm = CardMaker("bg_card")
        cm.setFrame(-1, 1, -1, 1)
        self.bg_card = self.camera.attachNewNode(cm.generate())
        self.bg_card.setPos(0, self._bg_dist, 0)
        self.bg_card.setTwoSided(True)
        self.bg_card.setTransparency(TransparencyAttrib.MAlpha)
        self.bg_card.setTexture(self.bg_tex)
        self.bg_card.setDepthTest(False)
        self.bg_card.setDepthWrite(False)
        self.bg_card.setBin("background", 0)
        self._layout_bg_card()
        print(f"🖼 Fondo activo (card en cámara): {image_path} ({self.bg_tex.getXSize()}x{self.bg_tex.getYSize()})")

    def _layout_bg_card(self):
        if not self.bg_card or not self.bg_tex:
            return
        lens = self.cam.node().getLens()
        hfov, vfov = lens.getFov()
        dist = self._bg_dist
        width  = 2.0 * dist * math.tan(math.radians(hfov * 0.5))
        height = 2.0 * dist * math.tan(math.radians(vfov * 0.5))
        view_ar = max(1e-6, width / height)
        img_w = max(self.bg_tex.getXSize(), 1)
        img_h = max(self.bg_tex.getYSize(), 1)
        img_ar = img_w / img_h
        sx = sz = 1.0
        if img_ar > view_ar: sx = img_ar / view_ar
        else: sz = view_ar / img_ar
        self.bg_card.setScale((width * 0.5) * sx, 1.0, (height * 0.5) * sz)

    def _on_window_event(self, window):
        if window is self.win:
            self._layout_bg_card()

    # ---------- Tabla de visemas ----------
    def _build_viseme_table(self):
        # Agregamos 'seal' (0..1) para forzar cierre real de labios
        V = {}
        # Vocales y alófonos
        V["REST"]=dict(top=0.0,  bot=0.0,  roll=0.0,  tongue=0.0, jaw_open=0.0,  menton_close=0.0,  lip_close_mult=1.0, seal=0.0)
        V["A"]   =dict(top=+2.3, bot=+7.8, roll=+0.2, tongue=+1.0, jaw_open=0.90, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["AH"]  =dict(top=+2.0, bot=+6.8, roll=+0.1, tongue=+0.9, jaw_open=0.80, menton_close=0.12, lip_close_mult=1.00, seal=0.0)
        V["AE"]  =dict(top=+2.0, bot=+5.8, roll=+0.5, tongue=+0.8, jaw_open=0.65, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["E"]   =dict(top=+1.8, bot=+5.2, roll=+0.8, tongue=+0.8, jaw_open=0.55, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["EH"]  =dict(top=+1.6, bot=+4.5, roll=+1.0, tongue=+0.8, jaw_open=0.48, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["I"]   =dict(top=+1.2, bot=+3.6, roll=+1.6, tongue=+0.6, jaw_open=0.35, menton_close=0.00, lip_close_mult=1.00, seal=0.0)
        V["IH"]  =dict(top=+1.0, bot=+3.0, roll=+1.2, tongue=+0.5, jaw_open=0.28, menton_close=0.00, lip_close_mult=1.00, seal=0.0)
        V["O"]   =dict(top=+1.4, bot=+4.4, roll=-1.0, tongue=+0.4, jaw_open=0.55, menton_close=1.10, lip_close_mult=0.30, seal=0.0)
        V["OE"]  =dict(top=+1.5, bot=+4.2, roll=-0.6, tongue=+0.5, jaw_open=0.52, menton_close=0.90, lip_close_mult=0.40, seal=0.0)
        V["OU"]  =dict(top=+1.2, bot=+3.8, roll=-1.1, tongue=+0.4, jaw_open=0.50, menton_close=0.85, lip_close_mult=0.50, seal=0.0)
        V["U"]   =dict(top=+1.0, bot=+3.2, roll=-1.3, tongue=+0.3, jaw_open=0.50, menton_close=0.80, lip_close_mult=1.00, seal=0.0)
        # Labiales/dentales
        V["BMP"] =dict(top=0.0,  bot=0.0,  roll=0.0,  tongue=0.0, jaw_open=0.0,  menton_close=0.50, lip_close_mult=1.00, seal=1.0)
        V["FV"]  =dict(top=+0.8, bot=-0.4, roll=+0.2, tongue=0.0, jaw_open=0.12, menton_close=0.45, lip_close_mult=1.00, seal=0.65)
        # Linguales/otras
        V["L"]   =dict(top=+1.1, bot=+2.6, roll=+0.3, tongue=+2.6, jaw_open=0.35, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["DT"]  =dict(top=+1.0, bot=+2.1, roll=+0.3, tongue=+0.7, jaw_open=0.25, menton_close=0.10, lip_close_mult=1.00, seal=0.15)
        V["R"]   =dict(top=+1.0, bot=+2.4, roll=+0.4, tongue=+0.9, jaw_open=0.30, menton_close=0.05, lip_close_mult=1.00, seal=0.0)
        V["RR"]  =dict(top=+1.0, bot=+2.6, roll=+0.4, tongue=+1.0, jaw_open=0.33, menton_close=0.08, lip_close_mult=1.00, seal=0.0)
        V["CH"]  =dict(top=+0.9, bot=+2.6, roll=+0.2, tongue=+0.5, jaw_open=0.25, menton_close=0.30, lip_close_mult=1.00, seal=0.0)
        V["TS"]  =dict(top=+0.9, bot=+2.0, roll=+0.4, tongue=+0.6, jaw_open=0.22, menton_close=0.20, lip_close_mult=1.00, seal=0.0)
        # Fricativas
        V["S"]   =dict(top=+0.8, bot=+1.6, roll=+0.6, tongue=+0.2, jaw_open=0.18, menton_close=0.00, lip_close_mult=1.00, seal=0.10)
        V["SH"]  =dict(top=+0.8, bot=+1.8, roll=+0.8, tongue=+0.3, jaw_open=0.16, menton_close=0.00, lip_close_mult=1.00, seal=0.12)
        # Semivocales
        V["W"]   =dict(top=+1.0, bot=+3.0, roll=-1.0, tongue=+0.5, jaw_open=0.45, menton_close=0.70, lip_close_mult=1.00, seal=0.0)
        V["Y"]   =dict(top=+1.1, bot=+2.8, roll=+1.2, tongue=+0.7, jaw_open=0.28, menton_close=0.05, lip_close_mult=1.00, seal=0.0)
        V["MID"] =dict(top=+1.4, bot=+4.0, roll=+0.5, tongue=+0.6, jaw_open=0.50, menton_close=0.20, lip_close_mult=1.00, seal=0.0)
        return V

    def _viseme_params(self, key: str):
        if not hasattr(self, "VTABLE") or self.VTABLE is None:
            self.VTABLE = self._build_viseme_table()
        return dict(self.VTABLE.get(key, self.VTABLE["MID"]))

    # ---------- Aplicación parámetros → huesos ----------
    def _apply_params(self, params, open_amount):
        s = self.MOVEMENT_SCALE
        intensity = self.intensity
        mb = self.mouth_boost

        # ATAQUE/RELEASE: mueve el "open_amount" hacia params["jaw_open"]
        desired = open_amount * params.get("jaw_open", 0.0)
        if not hasattr(self, "_open_state"): self._open_state = 0.0
        alpha = ATTACK_ALPHA if desired > self._open_state else RELEASE_ALPHA
        self._open_state = (1.0 - alpha)*self._open_state + alpha*desired

        top_v = s * (mb * (intensity * self.pitch_gain * self.top_ratio * params["top"]))
        bot_v = s * (mb * (intensity * self.pitch_gain * self.bot_ratio * params["bot"]))
        roll_amt  = s * (intensity * self.roll_gain * params["roll"])

        lip_close_mult = params.get("lip_close_mult", 1.0)
        dz_up   = clamp(self.lip_z_gain * top_v, -self.max_lip_dz,  self.max_lip_dz) * lip_close_mult
        dz_down = clamp(self.lip_z_gain * bot_v, -self.max_lip_dz,  self.max_lip_dz) * lip_close_mult

        # Labios
        for name,j in self.upper_L + self.upper_R:
            bp = self.base_pos[name]; j.setPos(bp[0], bp[1], bp[2] + dz_up)
        for name,j in self.lower_L + self.lower_R:
            bp = self.base_pos[name]; j.setPos(bp[0], bp[1], bp[2] - dz_down)

        # Roll lateral
        for name,j in self.upper_L + self.lower_L:
            h,p,r = self.base_hpr[name]; j.setHpr(h, p, r + roll_amt)
        for name,j in self.upper_R + self.lower_R:
            h,p,r = self.base_hpr[name]; j.setHpr(h, p, r - roll_amt)

        # Mandíbula (abre según estado con de-esser en sibilantes)
        jaw_open = params.get("jaw_open", 0.0) * self._open_state
        jaw_dz   = s * clamp(self.jaw_z_gain * (bot_v * jaw_open), -self.max_jaw_dz, self.max_jaw_dz)
        jaw_dz  *= self.jaw_atten

        # de-esser (S, SH, TS): menos mandíbula para no abrir de más
        if hasattr(self, "curr_key") and self.curr_key in ("S","SH","TS"):
            jaw_dz *= SIB_DEESS_JAW

        # SELLADO (BMP / FV)
        seal = params.get("seal", 0.0)
        if seal > 0.0:
            jaw_dz *= (1.0 - self.seal_jaw_mult * seal)
            extra = self.seal_lip_gain * seal * (1.0 - 0.35*open_amount)
            extra = clamp(extra, 0.0, 0.012)
            for name,j in self.upper_L + self.upper_R:
                bpx, bpy, bpz = j.getPos()
                j.setPos(bpx, bpy, bpz - extra)
            for name,j in self.lower_L + self.lower_R:
                bpx, bpy, bpz = j.getPos()
                j.setPos(bpx, bpy, bpz + extra)

        # Aplicar mandíbula / dientes
        for name,j in self.jaw:
            bp = self.base_pos[name]
            j.setPos(bp[0], bp[1], bp[2] - jaw_dz)

        for name,j in self.teeth_T:
            j.setPos(self.base_pos[name]); j.setHpr(self.base_hpr[name])
        for name,j in self.teeth_B:
            bp = self.base_pos[name]
            j.setPos(bp[0], bp[1], bp[2] - self.teeth_bot_follow * jaw_dz)
            j.setHpr(self.base_hpr[name])

        # Mentón/mejillas
        close_k = params.get("menton_close", 0.0)
        shape_amt = 0.5*(abs(top_v) + abs(bot_v))
        close_amt = s * clamp(self.close_dz_gain * shape_amt * close_k, 0.0, self.max_close_dz)
        close_amt *= self.chin_atten
        for name,j in (self.chin_L + self.chin_R + self.cheek_B_L + self.cheek_B_R):
            bp = self.base_pos[name]
            j.setPos(bp[0], bp[1], bp[2] + close_amt)
        for name,j in self.chin_center:
            j.setPos(self.base_pos[name])

        # Lengua
        for name,j in self.tongue:
            bh,bp,br = self.base_hpr[name]
            j.setHpr(bh, bp + s * intensity * params.get("tongue",0.0), br)

    def _micro_update_brows(self, t):
        if not (self.brow_L or self.brow_R): return
        dp = 1.0 * self.MOVEMENT_SCALE * math.sin(2*math.pi*0.6*t)
        for name,j in self.brow_L + self.brow_R:
            bh,bp,br = self.base_hpr[name]
            j.setHpr(bh, bp + dp, br)

    def _update_lids_translate_zx(self, t):
        if not (self.lid_top_L or self.lid_top_R or self.lid_bot_L or self.lid_bot_R):
            return
        s = self.MOVEMENT_SCALE
        now = globalClock.getFrameTime()

        if not hasattr(self, "next_blink"):
            self.next_blink = now + random.uniform(2.0, 5.0)
            self.blink_dur = random.uniform(0.10, 0.14)
            self.blinking = False

        blink_amt = 0.0
        if now >= self.next_blink and not self.blinking:
            self.blinking = True
            self.blink_start = now
        if self.blinking:
            u = (now - self.blink_start) / self.blink_dur
            if u >= 1.0:
                self.blinking = False
                self.next_blink = now + random.uniform(2.0, 5.0)
            else:
                blink_amt = 1.0 - (2.0*u - 1.0)**2

        micro_z = s * (0.00035 * math.sin(13.0 * t))
        micro_x = s * (0.00015 * math.sin(13.0 * 0.83 * t + 0.5))
        lat = (blink_amt ** 1.35)

        for (name, j, w) in self.lid_top_L:
            bpx, bpy, bpz = self.base_pos[name]
            dz = -1.0 * clamp((0.0032 * blink_amt * w)*s + micro_z*w, -0.0060*s, 0.0060*s)
            dx = -1.0 * clamp((0.0020 * lat * w)*s + micro_x*w, -0.0030*s, 0.0030*s)
            j.setPos(bpx + dx, bpy, bpz + dz); j.setHpr(self.base_hpr[name])
        for (name, j, w) in self.lid_top_R:
            bpx, bpy, bpz = self.base_pos[name]
            dz = -1.0 * clamp((0.0032 * blink_amt * w)*s + micro_z*w, -0.0060*s, 0.0060*s)
            dx = +1.0 * clamp((0.0020 * lat * w)*s + micro_x*w, -0.0030*s, 0.0030*s)
            j.setPos(bpx + dx, bpy, bpz + dz); j.setHpr(self.base_hpr[name])

        for (name, j, w) in self.lid_bot_L:
            bpx, bpy, bpz = self.base_pos[name]
            dz = +1.0 * clamp((0.0016 * blink_amt * w)*s + 0.5*micro_z*w, -0.0035*s, 0.0035*s)
            dx = -1.0 * clamp((0.0012 * lat * w)*s + 0.5*micro_x*w, -0.0020*s, 0.0020*s)
            j.setPos(bpx + dx, bpy, bpz + dz); j.setHpr(self.base_hpr[name])
        for (name, j, w) in self.lid_bot_R:
            bpx, bpy, bpz = self.base_pos[name]
            dz = +1.0 * clamp((0.0016 * blink_amt * w)*s + 0.5*micro_z*w, -0.0035*s, 0.0035*s)
            dx = +1.0 * clamp((0.0012 * lat * w)*s + 0.5*micro_x*w, -0.0020*s, 0.0020*s)
            j.setPos(bpx + dx, bpy, bpz + dz); j.setHpr(self.base_hpr[name])

        if blink_amt > 0.70:
            k = (blink_amt - 0.70) / 0.30
            k = clamp(k, 0.0, 1.0)
            sq_roll = 0.6 * self.MOVEMENT_SCALE * k
            sq_lift = 0.0012 * self.MOVEMENT_SCALE * k
            for name,j in self.upper_L + self.lower_L:
                h,p,r = self.base_hpr[name]; j.setHpr(h, p, r + sq_roll)
            for name,j in self.upper_R + self.lower_R:
                h,p,r = self.base_hpr[name]; j.setHpr(h, p, r - sq_roll)
            for name,j in (self.cheek_B_L + self.cheek_B_R):
                bp = self.base_pos[name]
                j.setPos(bp[0], bp[1], bp[2] + sq_lift)

    def _set_shoulders_static(self):
        # Aplica SOLO una vez la pose fija de hombros
        for name, j in self.upper_arm_L:
            bh, bp, br = self.base_hpr[name]
            H = bh + (self.SHO_L_ROT_X * self.SHO_L_SIGN_X)
            P = bp + (self.SHO_L_ROT_Y * self.SHO_L_SIGN_Y)
            R = br + (self.SHO_L_ROT_Z * self.SHO_L_SIGN_Z)
            j.setHpr(H, P, R)
        for name, j in self.upper_arm_R:
            bh, bp, br = self.base_hpr[name]
            H = bh + (self.SHO_R_ROT_X * self.SHO_R_SIGN_X)
            P = bp + (self.SHO_R_ROT_Y * self.SHO_R_SIGN_Y)
            R = br + (self.SHO_R_ROT_Z * self.SHO_R_SIGN_Z)
            j.setHpr(H, P, R)

    def _animate(self, task):
        vis_key = "REST"; vis_open = 0.0
        if hasattr(self, "audio") and self.audio is not None:
            d = self.audio.pop_latest()
            vis_key = d.get("key", "REST")
            vis_open = float(d.get("open", 0.0))

            v_target = self._viseme_params(vis_key)
            v_rest   = self._viseme_params("REST")

            # Mezcla dependiente del "open"
            target = {k: (1.0 - vis_open) * v_rest[k] + vis_open * v_target[k] for k in v_rest.keys()}

            # Ataque/Release por parámetro
            if not hasattr(self, 'curr_params'):
                self.curr_params = dict(v_rest)
            blended={}
            for k in v_rest.keys():
                a = ATTACK_ALPHA if target[k] > self.curr_params[k] else RELEASE_ALPHA
                blended[k] = (1.0 - a) * self.curr_params[k] + a * target[k]
            self.curr_params = blended
            self.curr_key = vis_key
            self._apply_params(self.curr_params, vis_open)
        else:
            v_rest = self._viseme_params("REST")
            if not hasattr(self, 'curr_params'):
                self.curr_params = dict(v_rest)
            for k in v_rest.keys():
                self.curr_params[k] = 0.85*self.curr_params[k] + 0.15*v_rest[k]
            self.curr_key = "REST"
            self._apply_params(self.curr_params, 0.0)

        t = globalClock.getFrameTime()
        self._micro_update_brows(t)
        self._update_lids_translate_zx(t)
        self._update_status(prefix=f"{vis_key} | open={vis_open:.2f}")
        return Task.cont

    def _find(self, names):
        out=[]
        for n in names:
            j = NodePath()
            try:
                j = self.actor.controlJoint(None, "metarig", n)
            except: pass
            if j.isEmpty():
                try: j = self.actor.controlJoint(None, "modelRoot", n)
                except: pass
            if not j.isEmpty():
                out.append((n,j))
        return out

    def _update_status(self, prefix=None):
        g = getattr(self.audio,'gain',INPUT_GAIN) if hasattr(self, "audio") and self.audio else INPUT_GAIN
        talking = False
        if hasattr(self, "audio") and self.audio:
            talking = getattr(self.audio, "is_talking", False)
        msg = (
            f"{prefix or ''}  |  Intensidad: {getattr(self,'intensity',0.6):.2f}"
            f" | MouthBoost: {getattr(self,'mouth_boost',1.45):.2f}"
            f" | Gain: {g:.2f} | Talking: {'ON' if talking else 'off'}"
        )
        self.status_display.setText(msg)
        self.status_display.setTextColor(VBase4(0.85,0.95,1,1))

    # Controles
    def _set_intensity(self, v):
        self.intensity = clamp(v, 0.30, 1.20); self._update_status()

    def _set_mouth_boost(self, v):
        self.mouth_boost = clamp(v, 0.80, 2.50); self._update_status()

    def _set_gain_rel(self, delta):
        if hasattr(self, "audio") and self.audio:
            self.audio.set_gain(self.audio.gain + delta)
        self._update_status()

    def _calibrate_noise(self):
        if hasattr(self, "audio") and self.audio:
            self.audio.calibrate_quick()

    # Cámara
    def _zoom_in(self): self.cam_dist = max(2,getattr(self,'cam_dist',5)-0.5)
    def _zoom_out(self): self.cam_dist = min(10,getattr(self,'cam_dist',5)+0.5)
    def _start_rotate(self):
        if self.mouseWatcherNode.hasMouse():
            self.is_rotating=True
            m=self.mouseWatcherNode.getMouse(); self.last_mouse=(m.getX(),m.getY())
    def _stop_rotate(self): self.is_rotating=False
    def _update_camera(self, task):
        if self.is_rotating and self.mouseWatcherNode.hasMouse():
            m=self.mouseWatcherNode.getMouse()
            dx=m.getX()-self.last_mouse[0]; dy=m.getY()-self.last_mouse[1]
            self.cam_angle_y += dx*100
            self.cam_angle_x = clamp(getattr(self,'cam_angle_x',15) - dy*100, -20, 60)
            self.last_mouse=(m.getX(),m.getY())
        x=self.cam_dist*math.sin(math.radians(self.cam_angle_y))
        y=-self.cam_dist*math.cos(math.radians(self.cam_angle_y))
        z=self.cam_dist*math.sin(math.radians(self.cam_angle_x))*0.2 + 1.8
        self.camera.setPos(x,y,z); self.camera.lookAt(0,0,1.5)
        return Task.cont

if __name__ == "__main__":
    app = TextVisemeDemo(
        GLB_PATH,
        bg_path=BG_PATH,
        audio_mode=AUDIO_MODE,
        sr=AUDIO_SR,
        block=AUDIO_BLOCK
    )
    app.run()
