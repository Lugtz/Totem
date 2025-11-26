# -*- coding: utf-8 -*-
"""
Visor GLB + VISEMAS por TEXTO (sin audio)
- Escribes texto en la terminal y el personaje lo "lee".
- Mueve labios superiores/inferiores y mandíbula en Z.
- Para visemas tipo O/U/W, agrega movimiento lateral (X) para redondear la boca.
"""

import sys, os, math, time, random, threading
from collections import deque

# ---------- CONSTANTES BÁSICAS ----------
GLB_PATH = "nacho.glb"
BG_PATH  = "fondo.png"

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------- Panda3D ----------
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.task import Task
from panda3d.core import (
    ClockObject, AmbientLight, DirectionalLight, VBase4,
    NodePath, TextNode, AntialiasAttrib, CardMaker, TransparencyAttrib
)

# Loader glTF/GLB
try:
    import gltf  # registra el loader de glTF para .glb
except Exception as e:
    print(f"⚠ No se pudo importar 'gltf' (panda3d-gltf): {e}")

globalClock = ClockObject.getGlobalClock()

# instancia global opcional para usar como "función"
APP_INSTANCE = None


class TextVisemeDemo(ShowBase):
    def __init__(self, glb_path, bg_path=None):
        ShowBase.__init__(self)
        self.setBackgroundColor(0.1, 0.1, 0.1)
        self.disableMouse()
        self.render.setAntialias(AntialiasAttrib.MAuto)

        # ========= sombreado / PBR =========
        # Intentamos usar simplepbr (corrige iluminación PBR y gamma)
        try:
            import simplepbr
            simplepbr.init(max_lights=2)
            print("✓ simplepbr activo")
        except Exception as e:
            print(f"⚠ simplepbr no disponible ({e}). Uso ShaderAuto")
            self.render.setShaderAuto()

        # ========= ILUMINACIÓN SUAVIZADA =========
        # Luces mucho más suaves para que Nacho no se vea blanco quemado.
        amb = AmbientLight("amb")
        amb.setColor(VBase4(0.25, 0.25, 0.25, 1))
        amb_np = self.render.attachNewNode(amb)
        self.render.setLight(amb_np)

        key = DirectionalLight("key")
        key.setColor(VBase4(0.65, 0.65, 0.65, 1))  # luz principal más suave
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(40, -45, 0)   # arriba-frontal
        self.render.setLight(key_np)

        fill = DirectionalLight("fill")
        fill.setColor(VBase4(0.35, 0.40, 0.45, 1))  # relleno frío y débil
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(-60, -15, 0)
        self.render.setLight(fill_np)

        # Cámara
        self.camera.setPos(0, -5, 2)
        self.camera.lookAt(0, 0, 1.5)

        # Fondo pegado a cámara (opcional)
        self.bg_card = None
        self.bg_tex  = None
        self._bg_dist = 12.0
        if bg_path:
            self._setup_background_card(bg_path)
            self.accept("window-event", self._on_window_event)

        # Carga modelo
        try:
            self.actor = Actor(glb_path)
        except Exception as e:
            print(f"❌ Error al cargar modelo '{glb_path}': {e}\n(Instala: pip install panda3d-gltf)")
            sys.exit(1)
        if not self.actor.getNode(0):
            print(f"❌ No se pudo cargar '{glb_path}'. Verifica ruta/nombre.")
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

        # Parámetros globales de movimiento
        self.MOVEMENT_SCALE = 1.0
        self.intensity      = 0.60
        self.mouth_boost    = 1.45

        self.jaw_atten  = 0.60 * self.MOVEMENT_SCALE
        self.chin_atten = 0.60 * self.MOVEMENT_SCALE

        # ====== HUESOS LABIOS / MANDÍBULA / CARA ======
        self.upper_L = self._find(["lip.T.L","lip.T.L.001"])
        self.upper_R = self._find(["lip.T.R","lip.T.R.001"])
        self.lower_L = self._find(["lip.B.L","lip.B.L.001"])
        self.lower_R = self._find(["lip.B.R","lip.B.R.001"])
        self.jaw     = self._find(["jaw"])

        # no animaremos lengua, solo labios + mandíbula (pero la referenciamos por si acaso)
        self.tongue  = self._find(["tongue","tongue.001","tongue.002"])

        self.teeth_T = self._find(["teeth.T"])
        self.teeth_B = self._find(["teeth.B"])

        self.chin_center = self._find(["chin"])
        self.chin_001    = self._find(["chin.001"])
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

        # Bases de posición / rotación
        self.base_hpr = {}
        self.base_pos = {}
        for group in [self.upper_L,self.upper_R,self.lower_L,self.lower_R,
                      self.jaw,self.tongue,self.teeth_T,self.teeth_B,
                      self.chin_center,self.chin_001,self.chin_L,self.chin_R,
                      self.cheek_B_L,self.cheek_B_R,
                      self.jaw_L_001,self.jaw_R_001,
                      self.brow_L,self.brow_R,
                      self.upper_arm_L,self.upper_arm_R]:
            for name,j in group:
                if name not in self.base_hpr: self.base_hpr[name] = j.getHpr()
                if name not in self.base_pos: self.base_pos[name] = j.getPos()

        # ====== HOMBROS EN POSE FIJA ======
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

        # Mezcla al parpadear (squeeze)
        self.squeeze_threshold = 0.70
        self.squeeze_roll_gain = 0.6 * self.MOVEMENT_SCALE
        self.squeeze_cheek_dz  = 0.0012 * self.MOVEMENT_SCALE

        # ====== TABLA DE VISEMAS ======
        self.VTABLE = self._build_viseme_table()

        # Ganancias labios/mandíbula (rango SUAVE)
        self.pitch_gain = 5.2
        self.roll_gain  = 4.0
        self.top_ratio  = 1.65
        self.bot_ratio  = 1.25

        self.lip_z_gain = 0.0018
        self.max_lip_dz = 0.012
        self.jaw_z_gain = 0.0022
        self.max_jaw_dz = 0.010
        self.teeth_bot_follow = 1.00
        self.close_dz_gain  = 0.0015
        self.max_close_dz   = 0.006

        # Fuerza extra de sellado (BMP/FV)
        self.seal_lip_gain = 0.0065 * self.MOVEMENT_SCALE
        self.seal_jaw_mult = 0.90

        # ====== CONTROL POR TEXTO ======
        self.curr_key      = "REST"
        self.target_key    = "REST"
        self.curr_params   = self._viseme_params("REST")
        self.next_change_t = 0.0

        self.viseme_seq   = deque()
        self._viseme_lock = threading.Lock()

        self._build_char_viseme_map()

        # --- IMPORTANTE: ya NO arrancamos hilo de input aquí ---
        # Si quisieras usar input por consola, puedes lanzar el hilo fuera:
        # t = threading.Thread(target=self._input_thread, daemon=True)
        # t.start()

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

        self.is_rotating=False; self.last_mouse=(0,0)
        self.cam_dist=5; self.cam_angle_x=15; self.cam_angle_y=0

        self.taskMgr.add(self._update_camera, "camera")
        self.taskMgr.add(self._animate, "animate")

        self._update_status(prefix=f"GLB: {os.path.basename(glb_path)} | Modo: TEXTO")

        print("\n🧠 Nacho listo.")
        print("Puedes llamarlo desde otro módulo usando enqueue_text() o speak_text().\n")

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
        print(f"🖼 Fondo activo: {image_path} ({self.bg_tex.getXSize()}x{self.bg_tex.getYSize()})")

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
        V = {}
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
        V["BMP"] =dict(top=0.0,  bot=0.0,  roll=0.0,  tongue=0.0, jaw_open=0.0,  menton_close=0.50, lip_close_mult=1.00, seal=1.0)
        V["FV"]  =dict(top=+0.8, bot=-0.4, roll=+0.2, tongue=0.0, jaw_open=0.12, menton_close=0.45, lip_close_mult=1.00, seal=0.65)
        V["L"]   =dict(top=+1.1, bot=+2.6, roll=+0.3, tongue=+2.6, jaw_open=0.35, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["DT"]  =dict(top=+1.0, bot=+2.1, roll=+0.3, tongue=+0.7, jaw_open=0.25, menton_close=0.10, lip_close_mult=1.00, seal=0.15)
        V["R"]   =dict(top=+1.0, bot=+2.4, roll=+0.4, tongue=+0.9, jaw_open=0.30, menton_close=0.05, lip_close_mult=1.00, seal=0.0)
        V["RR"]  =dict(top=+1.0, bot=+2.6, roll=+0.4, tongue=+1.0, jaw_open=0.33, menton_close=0.08, lip_close_mult=1.00, seal=0.0)
        V["CH"]  =dict(top=+0.9, bot=+2.6, roll=+0.2, tongue=+0.5, jaw_open=0.25, menton_close=0.30, lip_close_mult=1.00, seal=0.0)
        V["TS"]  =dict(top=+0.9, bot=+2.0, roll=+0.4, tongue=+0.6, jaw_open=0.22, menton_close=0.20, lip_close_mult=1.00, seal=0.0)
        V["S"]   =dict(top=+0.8, bot=+1.6, roll=+0.6, tongue=+0.2, jaw_open=0.18, menton_close=0.00, lip_close_mult=1.00, seal=0.10)
        V["SH"]  =dict(top=+0.8, bot=+1.8, roll=+0.8, tongue=+0.3, jaw_open=0.16, menton_close=0.00, lip_close_mult=1.00, seal=0.12)
        V["W"]   =dict(top=+1.0, bot=+3.0, roll=-1.0, tongue=+0.5, jaw_open=0.45, menton_close=0.70, lip_close_mult=1.00, seal=0.0)
        V["Y"]   =dict(top=+1.1, bot=+2.8, roll=+1.2, tongue=+0.7, jaw_open=0.28, menton_close=0.05, lip_close_mult=1.00, seal=0.0)
        V["MID"] =dict(top=+1.4, bot=+4.0, roll=+0.5, tongue=+0.6, jaw_open=0.50, menton_close=0.20, lip_close_mult=1.00, seal=0.0)
        return V

    def _viseme_params(self, key: str):
        if not hasattr(self, "VTABLE") or self.VTABLE is None:
            self.VTABLE = self._build_viseme_table()
        return dict(self.VTABLE.get(key, self.VTABLE["MID"]))

    # ---------- MAPEO CARACTER → VISEMA ----------
    def _build_char_viseme_map(self):
        self.char_to_viseme = {
            'a':"A",'á':"A",
            'e':"E",'é':"E",
            'i':"I",'í':"I",
            'o':"O",'ó':"O",
            'u':"U",'ú':"U",

            'b':"BMP",'p':"BMP",'m':"BMP",
            'f':"FV",'v':"FV",

            's':"S",'z':"S",
            'c':"S",  # aproximación

            'j':"SH",'x':"SH",'g':"SH",

            't':"DT",'d':"DT",
            'l':"L",
            'r':"R",
            'y':"Y",
            'w':"W",
            'ñ':"I",

            'k':"DT",'q':"DT",'n':"DT",
            'h':"REST"
        }

    def _text_to_viseme_sequence(self, text: str):
        seq = deque()
        vowels = set("aeiouáéíóú")
        speed_factor = 4 # 2x más rápido

        for ch in text.lower():
            if ch.isspace():
                seq.append(("REST", 0.06 / speed_factor))   # 0.03
                continue
            key = self.char_to_viseme.get(ch, "MID")
            if ch in vowels:
                dur = 0.13 / speed_factor                  # 0.065
            elif key in ("BMP", "FV"):
                dur = 0.10 / speed_factor                  # 0.05
            else:
                dur = 0.08 / speed_factor                  # 0.04
            seq.append((key, dur))

        # pequeño descanso al final
        seq.append(("REST", 0.18 / speed_factor))          # 0.09
        return seq



    # ============ NUEVO: API pública para hablar por texto ============
    def enqueue_text(self, text: str, clear_queue: bool = False):
        """
        Encola una frase para que Nacho la 'lea' con visemas.
        Si clear_queue=True, vacía primero la cola actual.
        """
        text = (text or "").strip()
        if not text:
            return
        seq = self._text_to_viseme_sequence(text)
        with self._viseme_lock:
            if clear_queue:
                self.viseme_seq.clear()
            self.viseme_seq.extend(seq)

    def speak_text(self, text: str, clear_queue: bool = False):
        """
        Alias de convenience: igual que enqueue_text().
        """
        self.enqueue_text(text, clear_queue=clear_queue)
    # ================================================================

    def _input_thread(self):
        print("Escribe texto y Nacho lo 'lee' con la boca (sin audio).")
        print("Ejemplo:  hola, cómo estás\n")
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            # Reutilizamos la API nueva:
            self.enqueue_text(line)

    # ---------- Aplicación parámetros → huesos ----------
    def _apply_params(self, params, open_amount):
        s = self.MOVEMENT_SCALE
        intensity = self.intensity
        mb = self.mouth_boost

        # Estado interno de "apertura" de mandíbula
        desired = open_amount * params.get("jaw_open", 0.0)
        if not hasattr(self, "_open_state"):
            self._open_state = 0.0
        ATTACK_ALPHA = 0.55
        RELEASE_ALPHA = 0.25
        alpha = ATTACK_ALPHA if desired > self._open_state else RELEASE_ALPHA
        self._open_state = (1.0 - alpha)*self._open_state + alpha*desired

        top_v = s * (mb * (intensity * self.pitch_gain * self.top_ratio * params["top"]))
        bot_v = s * (mb * (intensity * self.pitch_gain * self.bot_ratio * params["bot"]))

        # Factor de redondeo lateral (O/U/W) usando roll negativo
        roll = params.get("roll", 0.0)
        round_factor = max(0.0, -roll)   # solo cuando roll < 0
        lip_lat   = s * intensity * 0.0018 * round_factor
        lip_lat   = clamp(lip_lat, 0.0, 0.0030)
        cheek_lat = lip_lat * 0.6

        lip_close_mult = params.get("lip_close_mult", 1.0)
        dz_up   = clamp(self.lip_z_gain * top_v, -self.max_lip_dz,  self.max_lip_dz) * lip_close_mult
        dz_down = clamp(self.lip_z_gain * bot_v, -self.max_lip_dz,  self.max_lip_dz) * lip_close_mult

        # Labios (Z + lateral X para O/U/W)
        for name,j in self.upper_L:
            bp = self.base_pos[name]
            j.setPos(bp[0] + lip_lat, bp[1], bp[2] + dz_up)
        for name,j in self.upper_R:
            bp = self.base_pos[name]
            j.setPos(bp[0] - lip_lat, bp[1], bp[2] + dz_up)
        for name,j in self.lower_L:
            bp = self.base_pos[name]
            j.setPos(bp[0] + lip_lat, bp[1], bp[2] - dz_down)
        for name,j in self.lower_R:
            bp = self.base_pos[name]
            j.setPos(bp[0] - lip_lat, bp[1], bp[2] - dz_down)

        # Aseguramos rotaciones base en labios (no roll por visema)
        for name,j in self.upper_L + self.upper_R + self.lower_L + self.lower_R:
            j.setHpr(self.base_hpr[name])

        # Mandíbula (abre en Z)
        jaw_open = params.get("jaw_open", 0.0) * self._open_state
        jaw_dz   = s * clamp(self.jaw_z_gain * (bot_v * jaw_open), -self.max_jaw_dz, self.max_jaw_dz)
        jaw_dz  *= self.jaw_atten

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
            j.setPos(self.base_pos[name])
            j.setHpr(self.base_hpr[name])
        for name,j in self.teeth_B:
            bp = self.base_pos[name]
            j.setPos(bp[0], bp[1], bp[2] - self.teeth_bot_follow * jaw_dz)
            j.setHpr(self.base_hpr[name])

        # Mentón y mejillas (cierre + algo de lateral)
        close_k   = params.get("menton_close", 0.0)
        shape_amt = 0.5*(abs(top_v) + abs(bot_v))
        close_amt = s * clamp(self.close_dz_gain * shape_amt * close_k, 0.0, self.max_close_dz)
        close_amt *= self.chin_atten

        left_group  = self.chin_L + self.cheek_B_L
        right_group = self.chin_R + self.cheek_B_R

        for name,j in left_group:
            bp = self.base_pos[name]
            j.setPos(bp[0] + cheek_lat, bp[1], bp[2] + close_amt)
        for name,j in right_group:
            bp = self.base_pos[name]
            j.setPos(bp[0] - cheek_lat, bp[1], bp[2] + close_amt)

        for name,j in self.chin_center + self.chin_001:
            bp = self.base_pos[name]
            j.setPos(bp[0], bp[1], bp[2] + close_amt*0.8)

    def _micro_update_brows(self, t):
        if not (self.brow_L or self.brow_R):
            return
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
            self.blink_dur  = random.uniform(0.10, 0.14)
            self.blinking   = False

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

        # squeeze suave al pico del parpadeo (usa labios/cheeks)
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

    # ---------- BUCLE PRINCIPAL DE ANIMACIÓN ----------
    def _animate(self, task):
        now = globalClock.getFrameTime()

        # Gestionamos secuencia de visemas por tiempo
        if now >= self.next_change_t:
            with self._viseme_lock:
                if self.viseme_seq:
                    self.target_key, dur = self.viseme_seq.popleft()
                    self.next_change_t = now + float(dur)
                else:
                    self.target_key = "REST"
                    self.next_change_t = now + 0.08

        v_target = self._viseme_params(self.target_key)
        v_rest   = self._viseme_params("REST")

        # open_amount usa jaw_open del visema (vocales abren más)
        open_amount = v_target.get("jaw_open", 0.0)

        # Interpolación suave de parámetros (ataque/release)
        ATTACK_ALPHA = 0.55
        RELEASE_ALPHA = 0.25
        blended = {}
        for k in v_rest.keys():
            tgt = v_target[k]
            cur = self.curr_params.get(k, v_rest[k])
            a = ATTACK_ALPHA if tgt > cur else RELEASE_ALPHA
            blended[k] = (1.0 - a) * cur + a * tgt
        self.curr_params = blended
        self.curr_key    = self.target_key

        # Aplicar parámetros a huesos
        self._apply_params(self.curr_params, open_amount)

        t = now
        self._micro_update_brows(t)
        self._update_lids_translate_zx(t)

        self._update_status(prefix=f"{self.curr_key}")
        return Task.cont

    # ---------- UTILIDADES HUESOS ----------
    def _find(self, names):
        out=[]
        for n in names:
            j = NodePath()
            try:
                j = self.actor.controlJoint(None, "metarig", n)
            except: pass
            if j.isEmpty():
                try:
                    j = self.actor.controlJoint(None, "modelRoot", n)
                except: pass
            if not j.isEmpty():
                out.append((n,j))
        return out

    def _update_status(self, prefix=None):
        cola = 0
        try:
            cola = len(self.viseme_seq)
        except Exception:
            cola = 0
        msg = (
            f"{prefix or ''}  | Intensidad: {self.intensity:.2f}"
            f" | MouthBoost: {self.mouth_boost:.2f}"
            f" | Cola visemas: {cola}"
        )
        self.status_display.setText(msg)
        self.status_display.setTextColor(VBase4(0.85,0.95,1,1))

    # Controles de intensidad
    def _set_intensity(self, v):
        self.intensity = clamp(v, 0.30, 1.20)
        self._update_status()

    def _set_mouth_boost(self, v):
        self.mouth_boost = clamp(v, 0.80, 2.50)
        self._update_status()

    # Cámara
    def _zoom_in(self):
        self.cam_dist = max(2, getattr(self,'cam_dist',5)-0.5)
    def _zoom_out(self):
        self.cam_dist = min(10, getattr(self,'cam_dist',5)+0.5)
    def _start_rotate(self):
        if self.mouseWatcherNode.hasMouse():
            self.is_rotating=True
            m=self.mouseWatcherNode.getMouse()
            self.last_mouse=(m.getX(),m.getY())
    def _stop_rotate(self):
        self.is_rotating=False
    def _update_camera(self, task):
        if self.is_rotating and self.mouseWatcherNode.hasMouse():
            m=self.mouseWatcherNode.getMouse()
            dx=m.getX()-self.last_mouse[0]
            dy=m.getY()-self.last_mouse[1]
            self.cam_angle_y += dx*100
            self.cam_angle_x = clamp(getattr(self,'cam_angle_x',15) - dy*100, -20, 60)
            self.last_mouse=(m.getX(),m.getY())
        x=self.cam_dist*math.sin(math.radians(self.cam_angle_y))
        y=-self.cam_dist*math.cos(math.radians(self.cam_angle_y))
        z=self.cam_dist*math.sin(math.radians(self.cam_angle_x))*0.2 + 1.8
        self.camera.setPos(x,y,z)
        self.camera.lookAt(0,0,1.5)
        return Task.cont


# ---------- API SENCILLA PARA OTROS MÓDULOS ----------
def start_viewer(glb_path: str = GLB_PATH, bg_path: str = BG_PATH):
    """
    Crea el visor global y entra en el loop de Panda3D.
    Normalmente la llamas en un hilo aparte:
        threading.Thread(target=start_viewer, daemon=True).start()
    """
    global APP_INSTANCE
    APP_INSTANCE = TextVisemeDemo(glb_path, bg_path)
    APP_INSTANCE.run()

def speak(text: str, clear_queue: bool = False):
    """
    Función global que otros módulos pueden llamar:

        from ui_nacho import speak
        speak("Hola, soy Nacho")

    """
    if APP_INSTANCE is None:
        print("⚠ El visor aún no está inicializado. Llama primero a start_viewer().")
        return
    APP_INSTANCE.enqueue_text(text, clear_queue=clear_queue)


# ---------- MODO STANDALONE (con input por consola) ----------
if __name__ == "__main__":
    app = TextVisemeDemo(
        GLB_PATH,
        bg_path=BG_PATH
    )
    # Solo en modo standalone queremos leer de la consola:
    t = threading.Thread(target=app._input_thread, daemon=True)
    t.start()
    APP_INSTANCE = app
    app.run()
