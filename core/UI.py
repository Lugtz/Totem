# -*- coding: utf-8 -*-
"""
Visor GLB + VISEMAS por TEXTO (sin audio) + Panel CRM inferior
- Arriba: Nacho 3D con visemas.
- Abajo: tarjeta tipo CRM que se puede actualizar con update_crm().
"""

import sys, os, math, time, random, threading
from collections import deque

# --- NUEVO: imports para servidor HTTP ---
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote

# ---------- CONSTANTES BÁSICAS ----------
GLB_PATH = "nacho.glb"
BG_PATH  = "fondo.png"
HTTP_PORT = 7000  # puerto para el servidor HTTP

# --- COLORES DEL PANEL CRM (estilo Zoho) ---
CRM_APP_BG    = (0.0039, 0.337, 0.4, 1.0)   # fondo general inferior (azul petróleo)
CRM_CARD_BG   = (1.0, 1.0, 1.0, 1.0)        # tarjeta blanca
CRM_LABEL_FG  = (0.40, 0.40, 0.40, 1.0)     # etiquetas grises
CRM_VALUE_FG  = (0.13, 0.13, 0.13, 1.0)     # texto negro
CRM_EMAIL_FG  = (0.83, 0.18, 0.18, 1.0)     # email en rojo
CRM_TITLE_FG  = (0.20, 0.20, 0.20, 1.0)     # título

# sombreado y borde suave de la tarjeta
CRM_SHADOW_BG = (0.0, 0.0, 0.0, 0.12)
CRM_BORDER_BG = (0.92, 0.96, 0.98, 1.0)

# Fracción de altura de pantalla que ocupa el CRM (33%)
PANEL_HEIGHT_FRACTION = 0.33  # 33% de la altura total (-1 a 1 en aspect2d)


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
        try:
            import simplepbr
            simplepbr.init(max_lights=2)
            print("✓ simplepbr activo")
        except Exception as e:
            print(f"⚠ simplepbr no disponible ({e}). Uso ShaderAuto")
            self.render.setShaderAuto()

        # ========= ILUMINACIÓN SUAVIZADA =========
        amb = AmbientLight("amb")
        amb.setColor(VBase4(0.25, 0.25, 0.25, 1))
        amb_np = self.render.attachNewNode(amb)
        self.render.setLight(amb_np)

        key = DirectionalLight("key")
        key.setColor(VBase4(0.65, 0.65, 0.65, 1))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(40, -45, 0)
        self.render.setLight(key_np)

        fill = DirectionalLight("fill")
        fill.setColor(VBase4(0.35, 0.40, 0.45, 1))
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

        # Mantener aspecto correcto del modelo al redimensionar
        self._force_aspect_from_window()
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
            self.actor.clearColor()
            self.actor.clearColorScale()
        except Exception:
            pass
        self.actor.reparentTo(self.render)

        # HUD
        self.ui_font = None
        self.status_display = TextNode('status_display')
        try:
            font = self.loader.loadFont('cmss12')
            if font:
                self.status_display.setFont(font)
                self.ui_font = font
        except Exception:
            pass
        self.status_display.setAlign(TextNode.A_left)
        self.status_node = self.aspect2d.attachNewNode(self.status_display)
        self.status_node.setScale(0.07)
        self.status_node.setPos(-self.getAspectRatio() + 0.1, 0, 0.9)

        # Panel CRM
        self._init_crm_panel()

        # Parámetros globales de movimiento
        self.MOVEMENT_SCALE = 1.0
        self.intensity      = 0.60
        self.mouth_boost    = 1.45

        self.jaw_atten  = 0.60 * self.MOVEMENT_SCALE
        self.chin_atten = 0.60 * self.MOVEMENT_SCALE

        # ====== HUESOS LABIOS / MANDÍBULA / CARA ======
        self.upper_L = self._find(["lip.T.L", "lip.T.L.001"])
        self.upper_R = self._find(["lip.T.R", "lip.T.R.001"])
        self.lower_L = self._find(["lip.B.L", "lip.B.L.001"])
        self.lower_R = self._find(["lip.B.R", "lip.B.R.001"])
        self.jaw     = self._find(["jaw"])

        self.tongue  = self._find(["tongue", "tongue.001", "tongue.002"])
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

        self.brow_L = self._find(["brow.B.L", "brow.T.L", "brow.B.L.001", "brow.T.L.001"])
        self.brow_R = self._find(["brow.B.R", "brow.T.R", "brow.B.R.001", "brow.T.R.001"])

        self.upper_arm_L = self._find(["upper_arm.L"])
        self.upper_arm_R = self._find(["upper_arm.R"])

        self.forearm_L = self._find(["forearm.L", "lower_arm.L"])
        self.forearm_R = self._find(["forearm.R", "lower_arm.R"])
        self.hand_L    = self._find(["hand.L", "palm.L"])
        self.hand_R    = self._find(["hand.R", "palm.R"])

        self.spine_bones = self._find(["spine", "spine.001", "spine.002", "spine.003"])
        self.head_bones  = self._find(["head", "head.001", "face"])

        # Offset aleatorio para gestos
        self._gesture_offset = random.uniform(0.0, math.pi * 2.0)
        self._gesture_mode = "REST"
        self._gesture_mode_t = 0.0
        self._last_gesture_mode = "REST"

        # Bases de posición / rotación
        self.base_hpr = {}
        self.base_pos = {}
        for group in [
            self.upper_L, self.upper_R, self.lower_L, self.lower_R,
            self.jaw, self.tongue, self.teeth_T, self.teeth_B,
            self.chin_center, self.chin_001, self.chin_L, self.chin_R,
            self.cheek_B_L, self.cheek_B_R,
            self.jaw_L_001, self.jaw_R_001,
            self.brow_L, self.brow_R,
            self.upper_arm_L, self.upper_arm_R,
            self.forearm_L, self.forearm_R,
            self.hand_L, self.hand_R,
            self.spine_bones, self.head_bones,
        ]:
            for name, j in group:
                if name not in self.base_hpr:
                    self.base_hpr[name] = j.getHpr()
                if name not in self.base_pos:
                    self.base_pos[name] = j.getPos()

        # ====== HOMBROS EN POSE FIJA ======
        self.SHO_L_ROT_X = 0.0
        self.SHO_L_ROT_Y = 0.0
        self.SHO_L_ROT_Z = 70.0
        self.SHO_L_SIGN_X = +1.0
        self.SHO_L_SIGN_Y = +1.0
        self.SHO_L_SIGN_Z = +1.0

        self.SHO_R_ROT_X = 0.0
        self.SHO_R_ROT_Y = 0.0
        self.SHO_R_ROT_Z = -70.0
        self.SHO_R_SIGN_X = +1.0
        self.SHO_R_SIGN_Y = +1.0
        self.SHO_R_SIGN_Z = +1.0

        self._set_shoulders_static()

        # ====== PÁRPADOS ======
        top_L_names = ["lid.T.L.003", "lid.T.L.002", "lid.T.L.001", "lid.T.L"]
        bot_L_names = ["lid.B.L.003", "lid.B.L.002", "lid.B.L.001", "lid.B.L"]
        top_R_names = [n.replace(".L", ".R") for n in top_L_names]
        bot_R_names = [n.replace(".L", ".R") for n in bot_L_names]
        lid_weights = [0.35, 0.60, 0.85, 1.00]

        def _bind_with_weights(names, weights):
            out = []
            for i, n in enumerate(names):
                pair = self._find([n])
                if pair:
                    name, j = pair[0]
                    out.append((name, j, weights[i]))
                    if name not in self.base_hpr:
                        self.base_hpr[name] = j.getHpr()
                    if name not in self.base_pos:
                        self.base_pos[name] = j.getPos()
            return out

        self.lid_top_L = _bind_with_weights(top_L_names, lid_weights)
        self.lid_top_R = _bind_with_weights(top_R_names, lid_weights)
        self.lid_bot_L = _bind_with_weights(bot_L_names, lid_weights)
        self.lid_bot_R = _bind_with_weights(bot_R_names, lid_weights)

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

        # ====== TABLA DE VISEMAS ======
        self.VTABLE = self._build_viseme_table()

        # Ganancias labios/mandíbula
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

        self.seal_lip_gain = 0.0065 * self.MOVEMENT_SCALE
        self.seal_jaw_mult = 0.90

        # CONTROL POR TEXTO
        self.curr_key      = "REST"
        self.target_key    = "REST"
        self.curr_params   = self._viseme_params("REST")
        self.next_change_t = 0.0

        self.viseme_seq   = deque()
        self._viseme_lock = threading.Lock()

        self._build_char_viseme_map()

        # Controles
        self.accept("wheel_up", self._zoom_in)
        self.accept("wheel_down", self._zoom_out)
        self.accept("mouse1", self._start_rotate)
        self.accept("mouse1-up", self._stop_rotate)
        self.accept("escape", sys.exit)
        self.accept("[", lambda: self._set_intensity(self.intensity - 0.05))
        self.accept("]", lambda: self._set_intensity(self.intensity + 0.05))
        self.accept("k", lambda: self._set_mouth_boost(self.mouth_boost - 0.05))
        self.accept("l", lambda: self._set_mouth_boost(self.mouth_boost + 0.05))
        self.accept("arrow_up", self._view_up)
        self.accept("arrow_down", self._view_down)

        self.is_rotating = False
        self.last_mouse = (0, 0)
        self.cam_dist = 5
        self.cam_angle_x = 15
        self.cam_angle_y = 0
        self.view_offset_z = 0.0

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
        if img_ar > view_ar:
            sx = img_ar / view_ar
        else:
            sz = view_ar / img_ar
        self.bg_card.setScale((width * 0.5) * sx, 1.0, (height * 0.5) * sz)

    def _force_aspect_from_window(self):
        if not self.win:
            return
        props = self.win.getProperties()
        w = props.getXSize()
        h = props.getYSize()
        if w <= 0 or h <= 0:
            return
        aspect = float(w) / float(h)
        lens = self.cam.node().getLens()
        lens.setAspectRatio(aspect)

    def _on_window_event(self, window):
        if window is self.win:
            self._force_aspect_from_window()
            self._layout_bg_card()

    # ---------- Tabla de visemas ----------
    def _build_viseme_table(self):
        V = {}
        V["REST"] = dict(top=0.0,  bot=0.0,  roll=0.0,  tongue=0.0, jaw_open=0.0,  menton_close=0.0,  lip_close_mult=1.0, seal=0.0)
        V["A"]   = dict(top=+2.3, bot=+7.8, roll=+0.2, tongue=+1.0, jaw_open=0.90, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["AH"]  = dict(top=+2.0, bot=+6.8, roll=+0.1, tongue=+0.9, jaw_open=0.80, menton_close=0.12, lip_close_mult=1.00, seal=0.0)
        V["AE"]  = dict(top=+2.0, bot=+5.8, roll=+0.5, tongue=+0.8, jaw_open=0.65, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["E"]   = dict(top=+1.8, bot=+5.2, roll=+0.8, tongue=+0.8, jaw_open=0.55, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["EH"]  = dict(top=+1.6, bot=+4.5, roll=+1.0, tongue=+0.8, jaw_open=0.48, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["I"]   = dict(top=+1.2, bot=+3.6, roll=+1.6, tongue=+0.6, jaw_open=0.35, menton_close=0.00, lip_close_mult=1.00, seal=0.0)
        V["IH"]  = dict(top=+1.0, bot=+3.0, roll=+1.2, tongue=+0.5, jaw_open=0.28, menton_close=0.00, lip_close_mult=1.00, seal=0.0)
        V["O"]   = dict(top=+1.4, bot=+4.4, roll=-1.0, tongue=+0.4, jaw_open=0.55, menton_close=1.10, lip_close_mult=0.30, seal=0.0)
        V["OE"]  = dict(top=+1.5, bot=+4.2, roll=-0.6, tongue=+0.5, jaw_open=0.52, menton_close=0.90, lip_close_mult=0.40, seal=0.0)
        V["OU"]  = dict(top=+1.2, bot=+3.8, roll=-1.1, tongue=+0.4, jaw_open=0.50, menton_close=0.85, lip_close_mult=0.50, seal=0.0)
        V["U"]   = dict(top=+1.0, bot=+3.2, roll=-1.3, tongue=+0.3, jaw_open=0.50, menton_close=0.80, lip_close_mult=1.00, seal=0.0)
        V["BMP"] = dict(top=0.0,  bot=0.0,  roll=0.0,  tongue=0.0, jaw_open=0.0,  menton_close=0.50, lip_close_mult=1.00, seal=1.0)
        V["FV"]  = dict(top=+0.8, bot=-0.4, roll=+0.2, tongue=0.0, jaw_open=0.12, menton_close=0.45, lip_close_mult=1.00, seal=0.65)
        V["L"]   = dict(top=+1.1, bot=+2.6, roll=+0.3, tongue=+2.6, jaw_open=0.35, menton_close=0.10, lip_close_mult=1.00, seal=0.0)
        V["DT"]  = dict(top=+1.0, bot=+2.1, roll=+0.3, tongue=+0.7, jaw_open=0.25, menton_close=0.10, lip_close_mult=1.00, seal=0.15)
        V["R"]   = dict(top=+1.0, bot=+2.4, roll=+0.4, tongue=+0.9, jaw_open=0.30, menton_close=0.05, lip_close_mult=1.00, seal=0.0)
        V["RR"]  = dict(top=+1.0, bot=+2.6, roll=+0.4, tongue=+1.0, jaw_open=0.33, menton_close=0.08, lip_close_mult=1.00, seal=0.0)
        V["CH"]  = dict(top=+0.9, bot=+2.6, roll=+0.2, tongue=+0.5, jaw_open=0.25, menton_close=0.30, lip_close_mult=1.00, seal=0.0)
        V["TS"]  = dict(top=+0.9, bot=+2.0, roll=+0.4, tongue=+0.6, jaw_open=0.22, menton_close=0.20, lip_close_mult=1.00, seal=0.0)
        V["S"]   = dict(top=+0.8, bot=+1.6, roll=+0.6, tongue=+0.2, jaw_open=0.18, menton_close=0.00, lip_close_mult=1.00, seal=0.10)
        V["SH"]  = dict(top=+0.8, bot=+1.8, roll=+0.8, tongue=+0.3, jaw_open=0.16, menton_close=0.00, lip_close_mult=1.00, seal=0.12)
        V["W"]   = dict(top=+1.0, bot=+3.0, roll=-1.0, tongue=+0.5, jaw_open=0.45, menton_close=0.70, lip_close_mult=1.00, seal=0.0)
        V["Y"]   = dict(top=+1.1, bot=+2.8, roll=+1.2, tongue=+0.7, jaw_open=0.28, menton_close=0.05, lip_close_mult=1.00, seal=0.0)
        V["MID"] = dict(top=+1.4, bot=+4.0, roll=+0.5, tongue=+0.6, jaw_open=0.50, menton_close=0.20, lip_close_mult=1.00, seal=0.0)
        return V

    def _viseme_params(self, key: str):
        if not hasattr(self, "VTABLE") or self.VTABLE is None:
            self.VTABLE = self._build_viseme_table()
        return dict(self.VTABLE.get(key, self.VTABLE["MID"]))

    # ---------- MAPEO CARACTER → VISEMA ----------
    def _build_char_viseme_map(self):
        self.char_to_viseme = {
            'a': "A", 'á': "A",
            'e': "E", 'é': "E",
            'i': "I", 'í': "I",
            'o': "O", 'ó': "O",
            'u': "U", 'ú': "U",

            'b': "BMP", 'p': "BMP", 'm': "BMP",
            'f': "FV", 'v': "FV",

            's': "S", 'z': "S",
            'c': "S",

            'j': "SH", 'x': "SH", 'g': "SH",

            't': "DT", 'd': "DT",
            'l': "L",
            'r': "R",
            'y': "Y",
            'w': "W",
            'ñ': "I",

            'k': "DT", 'q': "DT", 'n': "DT",
            'h': "REST"
        }

    def _text_to_viseme_sequence(self, text: str):
        seq = deque()
        vowels = set("aeiouáéíóú")
        speed_factor = 2.35  # rápido pero entendible

        for ch in text.lower():
            if ch.isspace():
                seq.append(("REST", 0.06 / speed_factor))
                continue
            key = self.char_to_viseme.get(ch, "MID")
            if ch in vowels:
                dur = 0.13 / speed_factor
            elif key in ("BMP", "FV"):
                dur = 0.10 / speed_factor
            else:
                dur = 0.08 / speed_factor
            seq.append((key, dur))

        seq.append(("REST", 0.18 / speed_factor))
        return seq

    # ============ API pública para hablar por texto ============
    def enqueue_text(self, text: str, clear_queue: bool = False):
        text = (text or "").strip()
        if not text:
            return
        seq = self._text_to_viseme_sequence(text)
        with self._viseme_lock:
            if clear_queue:
                self.viseme_seq.clear()
            self.viseme_seq.extend(seq)

    def speak_text(self, text: str, clear_queue: bool = False):
        self.enqueue_text(text, clear_queue=clear_queue)

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
            self.enqueue_text(line)

    # ---------- Aplicar parámetros a huesos ----------
    def _apply_params(self, params, open_amount):
        s = self.MOVEMENT_SCALE
        intensity = self.intensity
        mb = self.mouth_boost

        desired = open_amount * params.get("jaw_open", 0.0)
        if not hasattr(self, "_open_state"):
            self._open_state = 0.0
        ATTACK_ALPHA = 0.55
        RELEASE_ALPHA = 0.25
        alpha = ATTACK_ALPHA if desired > self._open_state else RELEASE_ALPHA
        self._open_state = (1.0 - alpha) * self._open_state + alpha * desired

        top_v = s * (mb * (intensity * self.pitch_gain * self.top_ratio * params["top"]))
        bot_v = s * (mb * (intensity * self.pitch_gain * self.bot_ratio * params["bot"]))

        roll = params.get("roll", 0.0)
        round_factor = max(0.0, -roll)
        lip_lat   = s * intensity * 0.0018 * round_factor
        lip_lat   = clamp(lip_lat, 0.0, 0.0030)
        cheek_lat = lip_lat * 0.6

        lip_close_mult = params.get("lip_close_mult", 1.0)
        dz_up   = clamp(self.lip_z_gain * top_v, -self.max_lip_dz,  self.max_lip_dz) * lip_close_mult
        dz_down = clamp(self.lip_z_gain * bot_v, -self.max_lip_dz,  self.max_lip_dz) * lip_close_mult

        for name, j in self.upper_L:
            bp = self.base_pos[name]
            j.setPos(bp[0] + lip_lat, bp[1], bp[2] + dz_up)
        for name, j in self.upper_R:
            bp = self.base_pos[name]
            j.setPos(bp[0] - lip_lat, bp[1], bp[2] + dz_up)
        for name, j in self.lower_L:
            bp = self.base_pos[name]
            j.setPos(bp[0] + lip_lat, bp[1], bp[2] - dz_down)
        for name, j in self.lower_R:
            bp = self.base_pos[name]
            j.setPos(bp[0] - lip_lat, bp[1], bp[2] - dz_down)

        for name, j in self.upper_L + self.upper_R + self.lower_L + self.lower_R:
            j.setHpr(self.base_hpr[name])

        jaw_open = params.get("jaw_open", 0.0) * self._open_state
        jaw_dz   = s * clamp(self.jaw_z_gain * (bot_v * jaw_open), -self.max_jaw_dz, self.max_jaw_dz)
        jaw_dz  *= self.jaw_atten

        seal = params.get("seal", 0.0)
        if seal > 0.0:
            jaw_dz *= (1.0 - self.seal_jaw_mult * seal)
            extra = self.seal_lip_gain * seal * (1.0 - 0.35 * open_amount)
            extra = clamp(extra, 0.0, 0.012)
            for name, j in self.upper_L + self.upper_R:
                bpx, bpy, bpz = j.getPos()
                j.setPos(bpx, bpy, bpz - extra)
            for name, j in self.lower_L + self.lower_R:
                bpx, bpy, bpz = j.getPos()
                j.setPos(bpx, bpy, bpz + extra)

        for name, j in self.jaw:
            bp = self.base_pos[name]
            j.setPos(bp[0], bp[1], bp[2] - jaw_dz)

        for name, j in self.teeth_T:
            j.setPos(self.base_pos[name])
            j.setHpr(self.base_hpr[name])
        for name, j in self.teeth_B:
            bp = self.base_pos[name]
            j.setPos(bp[0], bp[1], bp[2] - self.teeth_bot_follow * jaw_dz)
            j.setHpr(self.base_hpr[name])

        close_k   = params.get("menton_close", 0.0)
        shape_amt = 0.5 * (abs(top_v) + abs(bot_v))
        close_amt = s * clamp(self.close_dz_gain * shape_amt * close_k, 0.0, self.max_close_dz)
        close_amt *= self.chin_atten

        left_group  = self.chin_L + self.cheek_B_L
        right_group = self.chin_R + self.cheek_B_R

        for name, j in left_group:
            bp = self.base_pos[name]
            j.setPos(bp[0] + cheek_lat, bp[1], bp[2] + close_amt)
        for name, j in right_group:
            bp = self.base_pos[name]
            j.setPos(bp[0] - cheek_lat, bp[1], bp[2] + close_amt)

        for name, j in self.chin_center + self.chin_001:
            bp = self.base_pos[name]
            j.setPos(bp[0], bp[1], bp[2] + close_amt * 0.8)

    def _micro_update_brows(self, t):
        if not (self.brow_L or self.brow_R):
            return
        dp = 1.0 * self.MOVEMENT_SCALE * math.sin(2 * math.pi * 0.6 * t)
        for name, j in self.brow_L + self.brow_R:
            bh, bp, br = self.base_hpr[name]
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
                blink_amt = 1.0 - (2.0 * u - 1.0) ** 2

        micro_z = s * (0.00035 * math.sin(13.0 * t))
        micro_x = s * (0.00015 * math.sin(13.0 * 0.83 * t + 0.5))
        lat = (blink_amt ** 1.35)

        for (name, j, w) in self.lid_top_L:
            bpx, bpy, bpz = self.base_pos[name]
            dz = -1.0 * clamp((0.0032 * blink_amt * w) * s + micro_z * w, -0.0060 * s, 0.0060 * s)
            dx = -1.0 * clamp((0.0020 * lat * w) * s + micro_x * w, -0.0030 * s, 0.0030 * s)
            j.setPos(bpx + dx, bpy, bpz + dz)
            j.setHpr(self.base_hpr[name])
        for (name, j, w) in self.lid_top_R:
            bpx, bpy, bpz = self.base_pos[name]
            dz = -1.0 * clamp((0.0032 * blink_amt * w) * s + micro_z * w, -0.0060 * s, 0.0060 * s)
            dx = +1.0 * clamp((0.0020 * lat * w) * s + micro_x * w, -0.0030 * s, 0.0030 * s)
            j.setPos(bpx + dx, bpy, bpz + dz)
            j.setHpr(self.base_hpr[name])

        for (name, j, w) in self.lid_bot_L:
            bpx, bpy, bpz = self.base_pos[name]
            dz = +1.0 * clamp((0.0016 * blink_amt * w) * s + 0.5 * micro_z * w, -0.0035 * s, 0.0035 * s)
            dx = -1.0 * clamp((0.0012 * lat * w) * s + 0.5 * micro_x * w, -0.0020 * s, 0.0020 * s)
            j.setPos(bpx + dx, bpy, bpz + dz)
            j.setHpr(self.base_hpr[name])
        for (name, j, w) in self.lid_bot_R:
            bpx, bpy, bpz = self.base_pos[name]
            dz = +1.0 * clamp((0.0016 * blink_amt * w) * s + 0.5 * micro_z * w, -0.0035 * s, 0.0035 * s)
            dx = +1.0 * clamp((0.0012 * lat * w) * s + 0.5 * micro_x * w, -0.0020 * s, 0.0020 * s)
            j.setPos(bpx + dx, bpy, bpz + dz)
            j.setHpr(self.base_hpr[name])

        if blink_amt > 0.70:
            k = (blink_amt - 0.70) / 0.30
            k = clamp(k, 0.0, 1.0)
            sq_roll = 0.6 * self.MOVEMENT_SCALE * k
            sq_lift = 0.0012 * self.MOVEMENT_SCALE * k
            for name, j in self.upper_L + self.lower_L:
                h, p, r = self.base_hpr[name]
                j.setHpr(h, p, r + sq_roll)
            for name, j in self.upper_R + self.lower_R:
                h, p, r = self.base_hpr[name]
                j.setHpr(h, p, r - sq_roll)
            for name, j in (self.cheek_B_L + self.cheek_B_R):
                bp = self.base_pos[name]
                j.setPos(bp[0], bp[1], bp[2] + sq_lift)

    def _update_body_language(self, t: float, speak_level: float):
        """
        Lenguaje corporal desde la columna hasta las manos.

        Versión relajada:
        - Cabeza más tranquila (menos amplitud y menor frecuencia).
        - Brazos con movimientos un poco más amplios pero MUCHO más lentos y prolongados.
        - Brazos y antebrazos siguen usando eje principal P (pitch) para que se sienta más "eje Z visual".
        """

        # ------ CONFIG DE EJES SOLO PARA BRAZOS ------
        ARM_UPPER_MAIN_AXIS      = "P"  # eje principal H/P/R
        ARM_UPPER_SECOND_AXIS    = "R"
        ARM_FOREARM_MAIN_AXIS    = "P"
        ARM_FOREARM_SECOND_AXIS  = "R"
        ARM_HAND_MAIN_AXIS       = "P"
        ARM_HAND_SECOND_AXIS     = "R"

        # --- CONFIG DE VELOCIDADES Y RANGOS ---
        # Brazos muy lentos y suaves
        ARM_SPEED = 0.03
        # frecuencia de la onda principal de brazos (baja -> más lento)
        ARM_RANGE        = 0.70      # un poco más de amplitud que antes
        FOREARM_SPEED    = 0.05        # antebrazos algo más lentos
        SMALL_NOISE_FREQ = 0.10        # ruidito fino (manos) más lento también

        MOVEMENT_FLIP = 1

        s = self.MOVEMENT_SCALE
        speak_level = clamp(speak_level, 0.0, 1.0)
        now = t

        # ---------------- MODO DE GESTO ----------------
        if speak_level < 0.18:
            self._gesture_mode = "REST"
        else:
            if now >= self._gesture_mode_t:
                modes = ["OPEN_PALMS", "STEADY_R", "STEADY_L"]
                choices = [m for m in modes if m != self._last_gesture_mode]
                if not choices:
                    choices = modes
                self._gesture_mode = random.choice(choices)
                self._last_gesture_mode = self._gesture_mode
                # Cambios de gesto más prolongados
                self._gesture_mode_t = now + random.uniform(3.5, 5.0)

        mode = self._gesture_mode

        # ---------------- BASE CUERPO / COLUMNA ----------------
        # Más lento que antes para que todo se sienta relajado
        body_t  = t * 0.06 + self._gesture_offset
        small_t = t * SMALL_NOISE_FREQ

        sway_yaw   = MOVEMENT_FLIP * 1.0 * s * math.sin(body_t)
        sway_pitch = 0.7 * s * math.sin(body_t * 0.7 + 0.8)
        lift       = 0.0020 * s * math.sin(body_t * 0.9)

        for name, j in self.spine_bones:
            bh, bp, br    = self.base_hpr[name]
            bpx, bpy, bpz = self.base_pos[name]
            j.setHpr(bh + sway_yaw * 0.4, bp + sway_pitch * 0.6, br)
            j.setPos(bpx, bpy, bpz + lift)

        # ---------------- CABEZA MÁS RELAJADA ----------------
        # Menor amplitud y menor frecuencia
        head_t = t * (0.05 + 0.06 * speak_level) + self._gesture_offset * 0.3

        head_nod_amp  = (0.5 + 0.9 * speak_level) * s      # antes ~1.0+1.4
        head_turn_amp = (0.4 * speak_level) * s            # antes 0.7*speak_level

        head_nod  = head_nod_amp * math.sin(head_t)
        head_turn = MOVEMENT_FLIP * head_turn_amp * math.sin(head_t * 0.8 + 1.5)

        for name, j in self.head_bones:
            bh, bp, br = self.base_hpr[name]
            j.setHpr(bh + head_turn, bp + head_nod, br)

        # ---------------- RESPIRACIÓN / PECHO ----------------
        chest_open = (2.5 + 3.0 * speak_level) * s
        breath     = (1.0 + 0.7 * speak_level) * s * math.sin(body_t * 0.9)

        # ---------------- PARÁMETROS GLOBALES DE HABLA ----------------
        AMP       = 8.0                               # un poco menos agresivo
        base_talk = (2.5 + 4.0 * speak_level) * s     # antes 3.5 + 6.0
        talk_amp  = AMP * base_talk

        # Factores por modo de gesto
        if mode == "REST":
            l_factor = 0.12
            r_factor = 0.12
        elif mode == "OPEN_PALMS":
            l_factor = 0.55
            r_factor = 0.55
        elif mode == "STEADY_R":
            l_factor = 0.25
            r_factor = 0.80
        elif mode == "STEADY_L":
            l_factor = 0.80
            r_factor = 0.25
        else:
            l_factor = r_factor = 0.22

        # ---------------- ONDA LENTA PARA BRAZOS ----------------
        # Periodo ~30 segundos aprox → muy suave y prolongado
        arms_t   = t * ARM_SPEED + self._gesture_offset * 0.8
        arms_t_L = arms_t
        arms_t_R = arms_t + math.pi * 0.8

        arm_wave_scalar_L = math.sin(arms_t_L)
        arm_wave_scalar_R = math.sin(arms_t_R)

        arm_noise_L = 0.05 * math.sin(arms_t_L * 0.7 + 1.3)
        arm_noise_R = 0.05 * math.sin(arms_t_R * 0.9 + 2.1)

        arm_wave_L = MOVEMENT_FLIP * clamp(arm_wave_scalar_L * l_factor + arm_noise_L, -1.0, 1.0)
        arm_wave_R = MOVEMENT_FLIP * clamp(arm_wave_scalar_R * r_factor + arm_noise_R, -1.0, 1.0)

        # ---------------- HOMBROS (eje principal P → se siente "Z") ----------------
        for side, upper_arm, base_rot, wave in (
            ("L", self.upper_arm_L,
             (self.SHO_L_ROT_X, self.SHO_L_ROT_Y, self.SHO_L_ROT_Z),
             arm_wave_L),
            ("R", self.upper_arm_R,
             (self.SHO_R_ROT_X, self.SHO_R_ROT_Y, self.SHO_R_ROT_Z),
             arm_wave_R),
        ):
            rot_x, rot_y, rot_z = base_rot
            for name, j in upper_arm:
                bh, bp, br = self.base_hpr[name]
                if side == "L":
                    base_H = bh + rot_x * self.SHO_L_SIGN_X
                    base_P = bp + rot_y * self.SHO_L_SIGN_Y
                    base_R = br + rot_z * self.SHO_L_SIGN_Z
                else:
                    base_H = bh + rot_x * self.SHO_R_SIGN_X
                    base_P = bp + rot_y * self.SHO_R_SIGN_Y
                    base_R = br + rot_z * self.SHO_R_SIGN_Z

                off_main = (breath * 0.18 + talk_amp * 0.35 * wave)
                off_sec  = (chest_open * 0.12 + talk_amp * 0.25 * wave)

                off_main *= ARM_RANGE
                off_sec  *= ARM_RANGE

                H, P, R = base_H, base_P, base_R

                # eje principal (P) → brazo sube/baja hacia delante/atrás muy suave
                if ARM_UPPER_MAIN_AXIS == "H":
                    H += off_main
                elif ARM_UPPER_MAIN_AXIS == "P":
                    P += off_main
                elif ARM_UPPER_MAIN_AXIS == "R":
                    R += off_main

                # eje secundario
                if ARM_UPPER_SECOND_AXIS == "H":
                    H += off_sec
                elif ARM_UPPER_SECOND_AXIS == "P":
                    P += off_sec
                elif ARM_UPPER_SECOND_AXIS == "R":
                    R += off_sec

                j.setHpr(H, P, R)

        # ---------------- ANTEBRAZOS (CODO) ----------------
        forearm_base = t * FOREARM_SPEED
        max_deg      = 35.0 * ARM_RANGE * speak_level  # amplitud controlada

        for side, forearm in (("L", self.forearm_L), ("R", self.forearm_R)):
            phase = 0.0 if side == "L" else math.pi * 0.7
            wave  = math.sin(forearm_base + phase)

            for name, j in forearm:
                bh, bp, br = self.base_hpr[name]

                if speak_level < 0.15:
                    j.setHpr(bh, bp, br)
                    continue

                angle = max_deg * wave
                if side == "L":
                    angle = -angle * 0.9
                else:
                    angle = angle * 1.0

                H, P, R = bh, bp, br

                if ARM_FOREARM_MAIN_AXIS == "H":
                    H += angle
                elif ARM_FOREARM_MAIN_AXIS == "P":
                    P += angle
                elif ARM_FOREARM_MAIN_AXIS == "R":
                    R += angle

                sec_angle = angle * 0.25
                if ARM_FOREARM_SECOND_AXIS == "H":
                    H += sec_angle
                elif ARM_FOREARM_SECOND_AXIS == "P":
                    P += sec_angle
                elif ARM_FOREARM_SECOND_AXIS == "R":
                    R += sec_angle

                j.setHpr(H, P, R)

        # ---------------- MANOS ----------------
        for side, wave, hand in (
            ("L", arm_wave_L, self.hand_L),
            ("R", arm_wave_R, self.hand_R),
        ):
            for name, j in hand:
                bh, bp, br = self.base_hpr[name]

                base_roll_extra = 5.0 * s
                if speak_level > 0.25:
                    base_roll_extra += 3.0 * s * ARM_RANGE
                if mode == "OPEN_PALMS":
                    base_roll_extra += 5.0 * s * ARM_RANGE

                base_main = (talk_amp * 0.30 * wave) * ARM_RANGE
                base_sec  = 0.0

                twist_noise = (2.0 * s * math.sin(
                    small_t + (0.5 if side == "L" else -0.5)
                )) * ARM_RANGE

                H, P, R = bh, bp, br + base_roll_extra + twist_noise

                if ARM_HAND_MAIN_AXIS == "H":
                    H += base_main
                elif ARM_HAND_MAIN_AXIS == "P":
                    P += base_main
                elif ARM_HAND_MAIN_AXIS == "R":
                    R += base_main

                if ARM_HAND_SECOND_AXIS == "H":
                    H += base_sec
                elif ARM_HAND_SECOND_AXIS == "P":
                    P += base_sec
                elif ARM_HAND_SECOND_AXIS == "R":
                    R += base_sec

                j.setHpr(H, P, R)

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

    # ---------- BUCLE PRINCIPAL ----------
    def _animate(self, task):
        now = globalClock.getFrameTime()

        if now >= getattr(self, "next_change_t", 0.0):
            with self._viseme_lock:
                if self.viseme_seq:
                    self.target_key, dur = self.viseme_seq.popleft()
                    self.next_change_t = now + float(dur)
                else:
                    self.target_key = "REST"
                    self.next_change_t = now + 0.08

        v_target = self._viseme_params(self.target_key)
        v_rest   = self._viseme_params("REST")

        open_amount = v_target.get("jaw_open", 0.0)

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

        self._apply_params(self.curr_params, open_amount)

        t = now
        self._micro_update_brows(t)
        self._update_lids_translate_zx(t)

        speak_level = getattr(self, "_open_state", 0.0)
        self._update_body_language(t, speak_level)

        self._update_status(prefix=f"{self.curr_key}")
        return Task.cont

    # ---------- UTILIDADES HUESOS ----------
    def _find(self, names):
        out = []
        for n in names:
            j = NodePath()
            try:
                j = self.actor.controlJoint(None, "metarig", n)
            except:
                pass
            if j.isEmpty():
                try:
                    j = self.actor.controlJoint(None, "modelRoot", n)
                except:
                    pass
            if not j.isEmpty():
                out.append((n, j))
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
        self.status_display.setTextColor(VBase4(0.85, 0.95, 1, 1))

    # ---------- PANEL CRM EN LA PARTE INFERIOR ----------
    def _init_crm_panel(self):
        """
        Panel tipo Zoho CRM en la parte baja (ocupa aprox. 33% de la altura total).

        Campos visibles (en pantalla):
        - Nombre
        - Empresa
        - Correo
        - Teléfono
        - Diagnóstico
        """

        self.crm_root = self.aspect2d.attachNewNode("crm_root")

        # Fondo de la franja inferior (base: altura 0.9, de -0.45 a 0.45)
        cm_bg = CardMaker("crm_bg")
        cm_bg.setFrame(-1.0, 1.0, -0.45, 0.45)
        bg_np = self.crm_root.attachNewNode(cm_bg.generate())
        bg_np.setColor(*CRM_APP_BG)

        # Sombra
        cm_shadow = CardMaker("crm_shadow")
        cm_shadow.setFrame(-0.93, 0.93, -0.43, 0.43)
        shadow_np = self.crm_root.attachNewNode(cm_shadow.generate())
        shadow_np.setColor(*CRM_SHADOW_BG)
        shadow_np.setPos(0.02, 0, -0.01)

        # Borde suave
        cm_border = CardMaker("crm_border")
        cm_border.setFrame(-0.90, 0.90, -0.40, 0.40)
        border_np = self.crm_root.attachNewNode(cm_border.generate())
        border_np.setColor(*CRM_BORDER_BG)

        # Tarjeta blanca
        cm_card = CardMaker("crm_card")
        cm_card.setFrame(-0.88, 0.88, -0.38, 0.38)
        card_np = self.crm_root.attachNewNode(cm_card.generate())
        card_np.setColor(*CRM_CARD_BG)

        # Título
        title_y = 0.24

        icon_node = TextNode("crm_icon")
        icon_node.setText("")
        icon_node.setAlign(TextNode.A_left)
        if self.ui_font is not None:
            icon_node.setFont(self.ui_font)
        icon_node.setTextColor(*CRM_TITLE_FG)
        icon_np = self.crm_root.attachNewNode(icon_node)
        icon_np.setPos(-0.82, 0, title_y)
        icon_np.setScale(0.06)

        title_node = TextNode("crm_title")
        title_node.setText("Información del prospecto")
        title_node.setAlign(TextNode.A_left)
        if self.ui_font is not None:
            title_node.setFont(self.ui_font)
        title_node.setTextColor(*CRM_TITLE_FG)
        title_np = self.crm_root.attachNewNode(title_node)
        title_np.setPos(-0.76, 0, title_y + 0.005)
        title_np.setScale(0.055)

        # Campos: NOMBRE, EMPRESA, CORREO, TELÉFONO, DIAGNÓSTICO
        self._crm_fields = [
            ("Nombre",      "nombre"),
            ("Empresa",     "empresa"),
            ("Correo",      "correo"),
            ("Teléfono",    "telefono"),
            ("Diagnóstico", "diagnostico"),
        ]

        self._crm_value_nodes = {}

        label_x = -0.82
        value_x = -0.20
        start_y = 0.12
        dy      = 0.12

        for i, (label_text, key) in enumerate(self._crm_fields):
            y = start_y - i * dy

            lbl = TextNode(f"crm_lbl_{key}")
            lbl.setText(label_text)
            lbl.setAlign(TextNode.A_left)
            if self.ui_font is not None:
                lbl.setFont(self.ui_font)
            lbl.setTextColor(*CRM_LABEL_FG)
            lbl_np = self.crm_root.attachNewNode(lbl)
            lbl_np.setPos(label_x, 0, y)
            lbl_np.setScale(0.050)

            val_node = TextNode(f"crm_val_{key}")
            val_node.setText("-")
            val_node.setAlign(TextNode.A_left)
            if self.ui_font is not None:
                val_node.setFont(self.ui_font)

            if key == "correo":
                val_node.setTextColor(*CRM_EMAIL_FG)
            else:
                val_node.setTextColor(*CRM_VALUE_FG)

            val_np = self.crm_root.attachNewNode(val_node)
            val_np.setPos(value_x, 0, y)
            val_np.setScale(0.050)

            self._crm_value_nodes[key] = val_np

        # ===== Escala para que el panel ocupe el 33% de la altura =====
        panel_half_base   = 0.45
        panel_half_target = PANEL_HEIGHT_FRACTION  # 0.33 => 33% de 2 = 0.66 total

        scale_z  = panel_half_target / panel_half_base
        center_z = -1.0 + panel_half_target  # centrado justo sobre el borde inferior

        self.crm_root.setScale(1.0, 1.0, scale_z)
        self.crm_root.setZ(center_z)

    def update_lead_panel(self, data: dict):
        """
        Actualiza los textos del panel CRM.

        data esperado (en español):
        {
            "nombre":      "...",
            "empresa":     "...",
            "correo":      "...",
            "telefono":    "...",
            "diagnostico": "..."
        }

        También acepta algunos alias en inglés (compatibles con amain.py):
        - "name"     -> nombre
        - "company"  -> empresa
        - "email"    -> correo
        - "proposal" -> diagnostico
        """
        if not hasattr(self, "_crm_value_nodes"):
            return

        data = data or {}

        normalizado = {
            "nombre":      data.get("nombre")      or data.get("name"),
            "empresa":     data.get("empresa")     or data.get("company"),
            "correo":      data.get("correo")      or data.get("email"),
            "telefono":    data.get("telefono"),
            "diagnostico": data.get("diagnostico") or data.get("proposal"),
        }

        for key, node_np in self._crm_value_nodes.items():
            val = normalizado.get(key, "-")
            if val in (None, ""):
                val = "-"
            tn = node_np.node()
            tn.setText(str(val))

    # Controles intensidad
    def _set_intensity(self, v):
        self.intensity = clamp(v, 0.30, 1.20)
        self._update_status()

    def _set_mouth_boost(self, v):
        self.mouth_boost = clamp(v, 0.80, 2.50)
        self._update_status()

    # Cámara
    def _zoom_in(self):
        self.cam_dist = max(2, getattr(self, 'cam_dist', 5) - 0.5)

    def _zoom_out(self):
        self.cam_dist = min(10, getattr(self, 'cam_dist', 5) + 0.5)

    def _view_up(self):
        cur = getattr(self, "view_offset_z", 0.0)
        self.view_offset_z = clamp(cur + 0.10, -1.0, 2.0)

    def _view_down(self):
        cur = getattr(self, "view_offset_z", 0.0)
        self.view_offset_z = clamp(cur - 0.10, -1.0, 2.0)

    def _start_rotate(self):
        if self.mouseWatcherNode.hasMouse():
            self.is_rotating = True
            m = self.mouseWatcherNode.getMouse()
            self.last_mouse = (m.getX(), m.getY())

    def _stop_rotate(self):
        self.is_rotating = False

    def _update_camera(self, task):
        if self.is_rotating and self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            dx = m.getX() - self.last_mouse[0]
            dy = m.getY() - self.last_mouse[1]
            self.cam_angle_y += dx * 100
            self.cam_angle_x = clamp(
                getattr(self, 'cam_angle_x', 15) - dy * 100,
                -20, 60
            )
            self.last_mouse = (m.getX(), m.getY())

        x = self.cam_dist * math.sin(math.radians(self.cam_angle_y))
        y = -self.cam_dist * math.cos(math.radians(self.cam_angle_y))

        base_target_z = 1.5
        view_offset = getattr(self, "view_offset_z", 0.0)
        target_z = base_target_z + view_offset

        z = self.cam_dist * math.sin(math.radians(self.cam_angle_x)) * 0.2 + target_z

        self.camera.setPos(x, y, z)
        self.camera.lookAt(0, 0, target_z)
        return Task.cont


# ---------- API SENCILLA PARA OTROS MÓDULOS ----------
def start_viewer(glb_path: str = GLB_PATH, bg_path: str = BG_PATH):
    global APP_INSTANCE
    APP_INSTANCE = TextVisemeDemo(glb_path, bg_path)
    APP_INSTANCE.run()


def speak(text: str, clear_queue: bool = False):
    if APP_INSTANCE is None:
        print("⚠ El visor aún no está inicializado. Llama primero a start_viewer().")
        return
    APP_INSTANCE.enqueue_text(text, clear_queue=clear_queue)


def update_crm(lead_data: dict):
    """
    Actualiza el panel CRM:

        update_crm({
            "nombre":      "Nombre del cliente",
            "empresa":     "Empresa S.A.",
            "correo":      "cliente@dominio.com",
            "telefono":    "55-1234-5678",
            "diagnostico": "Descripción corta del diagnóstico"
        })

    También puedes seguir mandando:
        name, company, email, proposal
    y se normalizan a las claves en español.
    """
    if APP_INSTANCE is None:
        print("⚠ El visor aún no está inicializado. Llama primero a start_viewer().")
        return
    APP_INSTANCE.update_lead_panel(lead_data)


# ---------- SERVIDOR HTTP EN PUERTO 7000 ----------
class NachoRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global APP_INSTANCE

        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path or "/"

        # --- NUEVO: modo CRM ------------------------------------
        if path.startswith("/crm"):
            if APP_INSTANCE is None:
                self._send_response(503, "Nacho aún no está listo (APP_INSTANCE es None).")
                return

            # Tomamos parámetros en español, con fallback a los que manda amain.py
            data = {
                "nombre":      qs.get("nombre", [""])[0]      or qs.get("name", [""])[0],
                "empresa":     qs.get("empresa", [""])[0]     or qs.get("company", [""])[0],
                "correo":      qs.get("correo", [""])[0]      or qs.get("email", [""])[0],
                "telefono":    qs.get("telefono", [""])[0],
                "diagnostico": qs.get("diagnostico", [""])[0] or qs.get("proposal", [""])[0],
            }

            try:
                APP_INSTANCE.update_lead_panel(data)
                self._send_response(200, "OK, CRM actualizado")
            except Exception as e:
                msg = f"Error al actualizar CRM: {e}"
                print(msg)
                self._send_response(500, msg)
            return
        # -------------------------------------------------------

        # Modo texto normal (lo que ya tenías)
        if "t" in qs and qs["t"]:
            text = qs["t"][0]
        else:
            text = unquote(path.lstrip("/"))

        text = (text or "").strip()

        if not text:
            self._send_response(
                400,
                "Debes enviar texto en la URL, ej: /Hola%20soy%20Nacho o ?t=Hola"
            )
        elif APP_INSTANCE is None:
            self._send_response(503, "Nacho aún no está listo (APP_INSTANCE es None).")
        else:
            APP_INSTANCE.enqueue_text(text)
            self._send_response(200, f"OK, Nacho dirá: {text}")

    def log_message(self, format, *args):
        # Silenciar logs de HTTPServer en consola
        return

    def _send_response(self, code, msg: str):
        try:
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(msg.encode("utf-8", errors="ignore"))
        except Exception:
            pass


def start_http_server(port: int = HTTP_PORT):
    server = HTTPServer(("0.0.0.0", port), NachoRequestHandler)
    print(f"🌐 Servidor HTTP de Nacho escuchando en http://localhost:{port}")
    print("   Ejemplo:  http://localhost:7000/Hola%20soy%20Nacho")
    print("   O:        http://localhost:7000/?t=Hola%20soy%20Nacho")
    print("   CRM:      http://localhost:7000/crm?nombre=Lulu&empresa=Ei3&correo=test%40mail.com")
    server.serve_forever()


# ---------- MODO STANDALONE ----------
if __name__ == "__main__":
    app = TextVisemeDemo(
        GLB_PATH,
        bg_path=BG_PATH
    )
    APP_INSTANCE = app

    t = threading.Thread(target=app._input_thread, daemon=True)
    t.start()

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    app.run()
