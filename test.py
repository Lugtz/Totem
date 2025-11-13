# -*- coding: utf-8 -*-
# facial_fullface_speed_control.py — Animación facial con control de velocidad
# Uso:
#   pip install panda3d panda3d-gltf
#   python facial_fullface_speed_control.py --glb tu_modelo.glb

from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import ClockObject, AmbientLight, DirectionalLight, VBase4
from direct.task import Task
import math, random, sys, argparse

globalClock = ClockObject.getGlobalClock()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class FacialSpeedDemo(ShowBase):
    def __init__(self, glb_path):
        ShowBase.__init__(self)
        self.setBackgroundColor(0.15, 0.15, 0.15)
        self.disableMouse()
        self.camera.setPos(0, -5, 2)
        self.camera.lookAt(0, 0, 1.5)

        # --- Luces ---
        amb = AmbientLight("amb"); amb.setColor(VBase4(0.6,0.6,0.6,1))
        dli = DirectionalLight("dir"); dli.setColor(VBase4(0.95,0.95,0.95,1))
        self.render.setLight(self.render.attachNewNode(amb))
        dnp = self.render.attachNewNode(dli); dnp.setHpr(25,-45,0)
        self.render.setLight(dnp)

        # --- Modelo ---
        try:
            self.actor = Actor(glb_path)
        except Exception as e:
            print(f"Error al cargar modelo: {e}\n(Instala: pip install panda3d-gltf)")
            sys.exit(1)
        self.actor.reparentTo(self.render)

        # --- Grupos de huesos ---
        self.upper_L = self._find(["lip.T.L","lip.T.L.001"])
        self.upper_R = self._find(["lip.T.R","lip.T.R.001"])
        self.lower_L = self._find(["lip.B.L","lip.B.L.001"])
        self.lower_R = self._find(["lip.B.R","lip.B.R.001"])
        self.eyebrow_L = self._find([
            "brow.B.L","brow.B.L.001","brow.B.L.002","brow.B.L.003",
            "brow.T.L","brow.T.L.001","brow.T.L.002","brow.T.L.003"
        ])
        self.eyebrow_R = self._find([
            "brow.B.R","brow.B.R.001","brow.B.R.002","brow.B.R.003",
            "brow.T.R","brow.T.R.001","brow.T.R.002","brow.T.R.003"
        ])
        self.lid_L = self._find(["lid.T.L","lid.T.L.001","lid.T.L.002","lid.T.L.003",
                                 "lid.B.L","lid.B.L.001","lid.B.L.002","lid.B.L.003"])
        self.lid_R = self._find(["lid.T.R","lid.T.R.001","lid.T.R.002","lid.T.R.003",
                                 "lid.B.R","lid.B.R.001","lid.B.R.002","lid.B.R.003"])
        self.eye_L = self._find(["eye.L"])
        self.eye_R = self._find(["eye.R"])
        self.cheeks = self._find([
            "cheek.T.L","cheek.T.L.001","cheek.T.R","cheek.T.R.001",
            "cheek.B.L","cheek.B.L.001","cheek.B.R","cheek.B.R.001"
        ])
        self.forehead = self._find(["forehead.L","forehead.L.001","forehead.L.002",
                                    "forehead.R","forehead.R.001","forehead.R.002"])
        self.nose = self._find(["nose","nose.001","nose.002","nose.003","nose.004",
                                "nose.L","nose.L.001","nose.R","nose.R.001"])
        self.jaw = self._find(["jaw","jaw.L","jaw.L.001","jaw.R","jaw.R.001"])
        self.chin = self._find(["chin","chin.001","chin.L"])
        self.teeth = self._find(["teeth.T","teeth.B"])
        self.tongue = self._find(["tongue","tongue.001","tongue.002"])

        # Guardar HPR base y fase aleatoria
        self.base_hpr = {}
        self.phase = {}
        for group in [self.upper_L,self.upper_R,self.lower_L,self.lower_R,
                      self.eyebrow_L,self.eyebrow_R,self.lid_L,self.lid_R,
                      self.eye_L,self.eye_R,self.cheeks,self.forehead,
                      self.nose,self.jaw,self.chin,self.teeth,self.tongue]:
            for name,j in group:
                self.base_hpr[name] = j.getHpr()
                self.phase[name] = random.random()*math.tau

        # --- Parámetros ---
        self.mode = "speech"
        self.t = 0.0
        self.top_ratio = 0.35
        self.bot_ratio = 0.90
        self.pitch_idle = 0.6
        self.pitch_talk = 6.0
        self.roll_talk = 6.0
        self.roll_idle = 0.6

        # --- Control de velocidad ---
        self.speed_factor = 1.0
        self.accept("+", self.increase_speed)
        self.accept("-", self.decrease_speed)

        # Cámara interactiva
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

        # Animación
        self.taskMgr.add(self.animate, "animate")

    # ---------- Helpers ----------
    def _find(self, names):
        out = []
        for n in names:
            try:
                j = self.actor.controlJoint(None, "modelRoot", n)
                if not j.isEmpty():
                    out.append((n,j))
            except: pass
        return out

    def increase_speed(self):
        self.speed_factor = min(5.0, self.speed_factor + 0.1)
        print(f"Velocidad: {self.speed_factor:.2f}")

    def decrease_speed(self):
        self.speed_factor = max(0.1, self.speed_factor - 0.1)
        print(f"Velocidad: {self.speed_factor:.2f}")

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
        if x <= 0.2: y=0.0
        else:
            y = (x-0.2)/(1-0.2)
            y = pow(y,0.85)
        return clamp(y*1.05,0.0,1.0)

    def animate(self, task):
        dt = globalClock.getDt() * self.speed_factor
        self.t += dt

        env = self._shape_env(self._env_raw(self.t))

        # --- Labios ---
        if self.mode=="idle":
            top_pitch = self.pitch_idle*math.sin(self.t*1.5)
            bot_pitch = self.pitch_idle*math.sin(self.t*1.5+0.3)
            roll_amt  = self.roll_idle*math.sin(self.t*1.2)
        else:
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

        # --- Parpadeo (ojos a velocidad constante, mitad de dt) ---
        # blink_val = max(0, math.sin(self.t*0.5*random.uniform(1.5,3.0))*1.0)
        # for group in [self.lid_L,self.lid_R]:
        #     for name,j in group:
        #         h,p,r = self.base_hpr[name]
        #         j.setHpr(h,p-blink_val*15,r)

        # --- Ojos (movimiento muy sutil, mitad de velocidad) ---
        eye_val = 0.05*math.sin(self.t*0.01)
        for name,j in self.eye_L+self.eye_R:
            h,p,r = self.base_hpr[name]
            j.setHpr(h,p+eye_val,r)

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
        for name,j in self.upper_L+self.lower_L: h,p,r=j.getHpr(); j.setHpr(h,p,r+roll_amt)
        for name,j in self.upper_R+self.lower_R: h,p,r=j.getHpr(); j.setHpr(h,p,r-roll_amt)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True, help="ruta al modelo .glb")
    args = ap.parse_args()
    app = FacialSpeedDemo(args.glb)
    app.run()
