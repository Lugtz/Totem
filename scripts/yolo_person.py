# yolo_person.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
from obsbot_patrol import DigitalPanner, PatrolConfig  # Para PTZ real, cambia por PTZStub si implementas PTZ

# ---- Utilidades UI ----
def put_text(img, text, org, scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

def draw_box(img, bbox, color=(0, 255, 0), thick=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

# ---- Detección ----
def detect_person(model, frame, conf_th=0.4):
    """
    Devuelve (found, best_bbox, best_conf)
    best_bbox = (x1,y1,x2,y2) en coords del frame
    """
    res = model.predict(frame, verbose=False, conf=conf_th, classes=[0], iou=0.5, imgsz=1280)
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return False, None, 0.0

    # Elegimos la persona con mayor área
    best = None
    best_area = 0
    best_conf = 0.0
    for b in r0.boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        conf = float(b.conf[0].cpu().numpy())
        x1, y1, x2, y2 = map(int, xyxy)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
            best_conf = conf

    return (best is not None), best, best_conf

def main():
    ap = argparse.ArgumentParser(description="Patrullaje con YOLO + paneo digital")
    ap.add_argument("--source", default="0", help="Índice de cámara o ruta de video")
    ap.add_argument("--width", type=int, default=3840, help="Ancho deseado de captura (p.ej. 3840)")
    ap.add_argument("--height", type=int, default=2160, help="Alto deseado de captura (p.ej. 2160)")
    ap.add_argument("--out-w", type=int, default=1920, help="Ancho de salida (recorte)")
    ap.add_argument("--out-h", type=int, default=1080, help="Alto de salida (recorte)")
    ap.add_argument("--conf", type=float, default=0.40, help="Confianza mínima YOLO")
    ap.add_argument("--no-box", action="store_true", help="No dibujar cajas")
    args = ap.parse_args()

    # Modelo (elige yolov8n por compatibilidad)
    model = YOLO("yolov8n.pt")

    # Fuente
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if isinstance(src, int) else 0)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # más FPS en varias webcams

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("No pude leer de la cámara/archivo. Revisa índice o permisos.")

    H, W = frame.shape[:2]
    # Ajuste si out_w/out_h mayores que frame
    out_w = min(args.out_w, W)
    out_h = min(args.out_h, H)

    cfg = PatrolConfig(out_w=out_w, out_h=out_h)
    panner = DigitalPanner(W, H, cfg)

    last_person_t = time.time()
    person_present = False
    show_boxes = (not args.no_box)
    frozen = False
    paused = False

    # FPS simple
    last_frame_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        if dt > 0:
            fps = 1.0 / dt

        if not frozen:
            # Detección en el frame completo para no perder a alguien fuera del recorte
            found, bbox, conf = detect_person(model, frame, conf_th=args.conf)
            if found:
                last_person_t = now
                person_present = True
                panner.ensure_person_inside(bbox)
            else:
                person_present = False

            # Si no hay persona por X seg, activar patrulla
            if not person_present and (now - last_person_t) >= cfg.idle_seconds_to_patrol and not paused:
                panner.step_patrol()

        # Calcular recorte
        x1, y1, x2, y2 = panner.crop_coords()
        crop = frame[y1:y2, x1:x2].copy()

        # Overlays
        if show_boxes and person_present:
            bx = list(bbox)
            bx[0] = max(0, bx[0] - x1); bx[2] = max(0, bx[2] - x1)
            bx[1] = max(0, bx[1] - y1); bx[3] = max(0, bx[3] - y1)
            draw_box(crop, tuple(bx), (0, 255, 0), 2)

        put_text(crop, f"Patrullaje: {'PAUSADO' if paused else 'ON'}   Persona: {'SI' if person_present else 'NO'}", (10, 28))
        put_text(crop, f"Crop: {x1}:{y1} -> {x2}:{y2}   FPS~{int(fps)}", (10, 56))
        put_text(crop, "Q: salir | P: pausar | R: reanudar | G: congelar | B: cajas on/off", (10, crop.shape[0]-12), 0.6, 2)

        cv2.imshow("PATRULLAJE - Ei3", crop)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = True
        elif key == ord('r'):
            paused = False
        elif key == ord('g'):
            frozen = not frozen
        elif key == ord('b'):
            show_boxes = not show_boxes

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
