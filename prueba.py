from panda3d.core import *
from direct.actor.Actor import Actor

actor = Actor("nacho.glb")
print("\n=== HUESOS DETECTADOS POR PANDA3D ===")
for bundle in actor.getAnimControlDict().keys():
    print("Bundle:", bundle)
print("\nNodos:")
actor.ls()
