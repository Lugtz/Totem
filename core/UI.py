# -*- coding: utf-8 -*-
# facial_fullface_speed_control_metarig.py — Animación facial avanzada con detección de audio
# Uso:
#   pip install panda3d panda3d-gltf simplepbr sounddevice numpy
#   python facial_fullface_speed_control_metarig.py --glb tu_modelo.glb

from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import ClockObject, AmbientLight, DirectionalLight, VBase4, NodePath, TextNode 
from direct.task import Task
import math, random, sys, argparse
import sounddevice as sd
import numpy as np
import threading
from collections import deque
import time

globalClock = ClockObject.getGlobalClock()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# --- Ajustes de detección de audio ---
SAMPLE_RATE = 48000
BLOCK_DURATION = 0.05
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
RMS_THRESHOLD = 0.02
AVERAGE_BLOCKS = 3

class FacialSpeedDemo(ShowBase):
    def __init__(self, glb_path):
        ShowBase.__init__(self)
        self.setBackgroundColor(0.15, 0.15, 0.15)
        self.disableMouse()

        # === Render sólido para glTF ===
        try:
            import simplepbr
            simplepbr.init()
            print("✓ simplepbr activo")
        except Exception:
            self.render.setShaderAuto()
            print("⚠ simplepbr no disponible; usando ShaderAuto")

        self.camera.setPos(0, -5, 2)
        self.camera.lookAt(0, 0, 1.5)

        # --- Luces ---
        amb = AmbientLight("amb"); amb.setColor(VBase4(0.30, 0.30, 0.30, 1))
        key = DirectionalLight("key"); key.setColor(VBase4(0.75, 0.75, 0.75, 1))
        fill = DirectionalLight("fill"); fill.setColor(VBase4(0.20, 0.20, 0.20, 1))
        self.render.setLight(self.render.attachNewNode(amb))
        key_np = self.render.attachNewNode(key); key_np.setHpr(35, -55, 0); self.render.setLight(key_np)
        fill_np = self.render.attachNewNode(fill); fill_np.setHpr(-60, -20, 0); self.render.setLight(fill_np)

        # --- Modelo ---
        try:
            self.actor = Actor(glb_path)
        except Exception as e:
            print(f"Error al cargar modelo: {e}\n(Instala: pip install panda3d-gltf)")
            sys.exit(1)
        if not self.actor.getNode(0):
            print(f"\n❌ ERROR CRÍTICO: No se pudo cargar el archivo '{glb_path}'.")
            sys.exit(1)
        try:
            self.actor.clearColor()
            self.actor.clearColorScale()
        except Exception:
            pass
        self.actor.reparentTo(self.render)

        # --- Variables ---
        self.t = 0.0
        self.speed_factor = 1.0
        self.is_talking = False

        # --- Interfaz de estado ---
        status_font = self.loader.loadFont('cmss12')
        self.status_display = TextNode('status_display')
        self.status_display.setFont(status_font)
        self.status_display.setAlign(TextNode.A_left)
        self.status_node = self.aspect2d.attachNewNode(self.status_display)
        self.status_node.setScale(0.07)
        self.status_node.setPos(-self.getAspectRatio() + 0.1, 0, 0.9)
        self.status_display.setText("Estado: REPOSO")
        self.status_display.setTextColor(VBase4(1, 1, 1, 1))

        # --- Mapeo de huesos ---
        self.upper_L = self._find(["lip.T.L","lip.T.L.001"])
        self.upper_R = self._find(["lip.T.R","lip.T.R.001"])
        self.lower_L = self._find(["lip.B.L","lip.B.L.001"])
        self.lower_R = self._find(["lip.B.R","lip.B.R.001"])
        self.eyebrow_L = self._find(["brow.B.L","brow.B.L.001","brow.T.L","brow.T.L.001"])
        self.eyebrow_R = self._find(["brow.B.R","brow.B.R.001","brow.T.R","brow.T.R.001"])
        self.lid_L = self._find(["lid.T.L","lid.T.L.001","lid.B.L","lid.B.L.001"])
        self.lid_R = self._find(["lid.T.R","lid.T.R.001","lid.B.R","lid.B.R.001"])
        self.cheeks = self._find(["cheek.T.L","cheek.T.R","cheek.B.L","cheek.B.R"])
        self.forehead = self._find(["forehead.L","forehead.R"])
        self.nose = self._find(["nose"])
        self.jaw = self._find(["jaw"])
        self.chin = self._find(["chin"])
        self.teeth = self._find(["teeth.T","teeth.B"])
        self.tongue = self._find(["tongue"])

        # Guardar base HPR y fase aleatoria
        self.base_hpr = {}
        self.phase = {}
        all_groups = [self.upper_L,self.upper_R,self.lower_L,self.lower_R,
                      self.eyebrow_L,self.eyebrow_R,self.lid_L,self.lid_R,
                      self.cheeks,self.forehead,self.nose,self.jaw,self.chin,self.teeth,self.tongue]
        for group in all_groups:
            for name,j in group:
                self.base_hpr[name] = j.getHpr()
                self.phase[name] = random.random()*math.tau

        self.top_ratio = 0.35
        self.bot_ratio = 0.90
        self.pitch_talk = 6.0
        self.roll_talk = 6.0

        # --- Controles ---
        self.accept("f", lambda: setattr(self, 'is_talking', True))
        self.accept("f-up", lambda: setattr(self, 'is_talking', False))
        self.accept("+", self.increase_speed)
        self.accept("-", self.decrease_speed)
        self.accept("wheel_up", self._zoom_in)
        self.accept("wheel_down", self._zoom_out)
        self.accept("mouse1", self._start_rotate)
        self.accept("mouse1-up", self._stop_rotate)
        self.is_rotating = False
        self.last_mouse = (0,0)
        self.cam_dist = 5
        self.cam_angle_x = 15
        self.cam_angle_y = 0
        self.taskMgr.add(self._update_camera, "camera_task")
        self.taskMgr.add(self.animate, "animate")

        # --- Inicializar detección de audio ---
        self._init_audio_detection()

    # ---------- Helpers ----------
    def _find(self, names):
        out = []
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

    def increase_speed(self): self.speed_factor = min(5.0, self.speed_factor + 0.1)
    def decrease_speed(self): self.speed_factor = max(0.1, self.speed_factor - 0.1)
    def _zoom_in(self): self.cam_dist = max(2,self.cam_dist-0.5)
    def _zoom_out(self): self.cam_dist = min(10,self.cam_dist+0.5)
    def _start_rotate(self):
        if self.mouseWatcherNode.hasMouse():
            self.is_rotating = True
            m = self.mouseWatcherNode.getMouse()
            self.last_mouse = (m.getX(), m.getY())
    def _stop_rotate(self): self.is_rotating = False
    def _update_camera(self, task):
        if self.is_rotating and self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            dx = m.getX()-self.last_mouse[0]
            dy = m.getY()-self.last_mouse[1]
            self.cam_angle_y += dx*100
            self.cam_angle_x = clamp(self.cam_angle_x - dy*100, -20, 60)
            self.last_mouse = (m.getX(), m.getY())
        x = self.cam_dist*math.sin(math.radians(self.cam_angle_y))
        y = -self.cam_dist*math.cos(math.radians(self.cam_angle_y))
        z = self.cam_dist*math.sin(math.radians(self.cam_angle_x))*0.2 + 1.8
        self.camera.setPos(x,y,z)
        self.camera.lookAt(0,0,1.5)
        return Task.cont

    # ---------- Animación ----------
    def _env_raw(self, t):
        v = 0.45 + 0.30*math.sin(11.0*t) + 0.20*math.sin(17.0*t+1.0) + 0.10*math.sin(23.0*t+0.3)
        v += 0.05*random.uniform(-1.0,1.0)
        return clamp(v,0.0,1.0)

    def _shape_env(self, x):
        if x <= 0.2: return 0.0
        y = (x-0.2)/(1-0.2)
        y = pow(y,0.85)
        return clamp(y*1.05,0.0,1.0)

    def animate(self, task):
        if self.is_talking:
            self.status_display.setText(f"Estado: HABLANDO | Velocidad: {self.speed_factor:.1f}")
            self.status_display.setTextColor(VBase4(0.2, 1.0, 0.2, 1))
        else:
            self.status_display.setText(f"Estado: REPOSO | Velocidad: {self.speed_factor:.1f}")
            self.status_display.setTextColor(VBase4(1.0, 0.5, 0.5, 1))
            return self._reset_bones(task)

        dt = globalClock.getDt() * self.speed_factor
        self.t += dt
        env = self._shape_env(self._env_raw(self.t))

        # --- Labios ---
        top_pitch = self.pitch_talk*self.top_ratio*math.sin(self.t*8.0)*env
        bot_pitch = self.pitch_talk*self.bot_ratio*math.sin(self.t*8.0+0.2)*env
        roll_amt  = self.roll_talk*env*math.sin(self.t*7.5)
        self._apply_lips(top_pitch,bot_pitch,roll_amt)

        # --- Cejas ---
        brow_pitch = 1.5*math.sin(self.t*1.8)+0.8*env
        for group in [self.eyebrow_L,self.eyebrow_R]:
            for name,j in group:
                h,p,r = self.base_hpr[name]
                j.setHpr(h,p+brow_pitch,r)

        # --- Mejillas ---
        cheek_pitch = 0.3*top_pitch + 0.2*bot_pitch
        for name,j in self.cheeks:
            h,p,r = self.base_hpr[name]
            j.setHpr(h,p+cheek_pitch,r)

        # --- Frente ---
        for name,j in self.forehead:
            h,p,r = self.base_hpr[name]
            j.setHpr(h,p+0.3*env,r)

        # --- Nariz ---
        for name,j in self.nose:
            h,p,r = self.base_hpr[name]
            j.setHpr(h,p+0.2*env,r)

        # --- Mandíbula y barbilla ---
        jaw_pitch = bot_pitch*0.5
        for name,j in self.jaw+self.chin:
            h,p,r = self.base_hpr[name]
            j.setHpr(h,p+jaw_pitch,r)

        # --- Lengua y dientes ---
        for name,j in self.tongue+self.teeth:
            h,p,r = self.base_hpr[name]
            j.setHpr(h,p+0.1*env,r)

        return Task.cont

    def _reset_bones(self, task):
        all_groups = self.upper_L + self.upper_R + self.lower_L + self.lower_R + \
                     self.eyebrow_L + self.eyebrow_R + self.lid_L + self.lid_R + \
                     self.cheeks + self.forehead + self.nose + self.jaw + self.chin + self.teeth + self.tongue
        for name,j in all_groups:
            if name in self.base_hpr:
                j.setHpr(self.base_hpr[name])
        return Task.cont

    def _apply_lips(self, top_pitch, bot_pitch, roll_amt):
        TOP_SIGN = -1.0
        BOT_SIGN = +1.0
        for name,j in self.upper_L+self.upper_R:
            h,p,r = self.base_hpr[name]
            phase = self.phase[name]
            jitter = 0.15*math.sin(self.t*10.0+phase)
            j.setHpr(h,p+TOP_SIGN*(top_pitch+jitter),r)
        for name,j in self.lower_L+self.lower_R:
            h,p,r = self.base_hpr[name]
            phase = self.phase[name]
            jitter = 0.15*math.sin(self.t*10.0+phase+0.5)
            j.setHpr(h,p+BOT_SIGN*(bot_pitch+jitter),r)
        for name,j in self.upper_L+self.lower_L:
            h,p,r=j.getHpr(); j.setHpr(h,p,r+roll_amt)
        for name,j in self.upper_R+self.lower_R:
            h,p,r=j.getHpr(); j.setHpr(h,p,r-roll_amt)

    # ---------------- Audio ----------------
    def _audio_loop(self):
        estado_anterior = False
        rms_history = deque(maxlen=AVERAGE_BLOCKS)

        dispositivos = sd.query_devices()
        loopback_idx = None
        for i, dev in enumerate(dispositivos):
            name = dev['name'].lower()
            if dev['max_input_channels'] > 0 and ('stereo mix' in name or 'mix' in name):
                loopback_idx = i
                print(f"Usando dispositivo de loopback: [{i}] {dev['name']}")
                break
        if loopback_idx is None:
            print("⚠ No se encontró dispositivo de loopback. Activa 'Stereo Mix' en Windows.")
            return

        try:
            with sd.InputStream(device=loopback_idx,
                                channels=1,
                                samplerate=SAMPLE_RATE,
                                blocksize=BLOCK_SIZE,
                                dtype='float32') as stream:

                while True:
                    block, _ = stream.read(BLOCK_SIZE)
                    rms_history.append(float(np.sqrt(np.mean(np.square(block[:,0])))))
                    rms_promedio = float(np.mean(rms_history))
                    sonido_actual = rms_promedio >= RMS_THRESHOLD
                    if sonido_actual != estado_anterior:
                        self.is_talking = sonido_actual
                        estado_anterior = sonido_actual

        except Exception as e:
            print("Error en stream de audio:", e)

    def _init_audio_detection(self):
        thread = threading.Thread(target=self._audio_loop, daemon=True)
        thread.start()


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True, help="ruta al modelo .glb")
    args = ap.parse_args()
    app = FacialSpeedDemo(args.glb)
    app.run()
