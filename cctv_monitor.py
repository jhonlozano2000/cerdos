"""
cctv_monitor.py — Monitoreo CCTV en tiempo real
Reconocimiento automático de cerdas usando YOLOv8 + MobileNetV2.

Uso: python cctv_monitor.py [--source 0] [--confidence 0.5]
     python cctv_monitor.py --source "rtsp://camera:554/stream"

Controles:
  q  - Salir
  p  - Pausar/Reanudar
  s  - Guardar frame actual
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_identificacion_cerdos.h5"
CLASSES_PATH = BASE_DIR / "classes.json"
LOG_PATH = BASE_DIR / "cctv_log.jsonl"
YOLO_MODEL_PATH = BASE_DIR / "yolov8n.pt"

IMG_SIZE = (224, 224)
THRESHOLD = 0.50
YOLO_CONF = 0.25
TARGET_FPS = 5
FRAME_INTERVAL = 1.0 / TARGET_FPS


def load_classes():
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH) as f:
            data = json.load(f)
            return data.get("classes", [])
    return []


CLASS_NAMES = load_classes()
print(f"Clases cargadas: {len(CLASS_NAMES)} - {CLASS_NAMES}")


def load_models():
    print("Cargando YOLOv8...")
    yolo = YOLO(str(YOLO_MODEL_PATH))

    print("Cargando MobileNetV2 (identificación)...")
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print("Modelo de identificación cargado")
    else:
        print(f"Modelo no encontrado en {MODEL_PATH}, identificación deshabilitada")
        model = None

    return yolo, model


def preprocess_image(roi):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def identify_pig(model, roi):
    if model is None:
        return "Desconocido", 0.0

    img_array = preprocess_image(roi)
    predictions = model.predict(img_array, verbose=0)[0]
    top_idx = np.argmax(predictions)
    confidence = float(predictions[top_idx])

    if confidence < THRESHOLD:
        return "Desconocido", confidence

    class_name = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"ID_{top_idx}"
    return class_name, confidence


def log_detection(animal_id, confidence, bbox, frame_num):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "frame": frame_num,
        "animal_id": animal_id,
        "confidence": round(confidence, 4),
        "bbox": [round(v, 1) for v in bbox],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def draw_info(frame, detections, fps, paused):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    status = "PAUSADO" if paused else "EN VIVO"
    color_status = (0, 0, 255) if paused else (0, 255, 0)
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"Detectados: {detections}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Monitor CCTV para reconocimiento de cerdas")
    parser.add_argument("--source", default="0", help="Índice de cámara (0) o URL RTSP")
    parser.add_argument("--confidence", type=float, default=0.5, help="Umbral de confianza para identificación")
    args = parser.parse_args()

    global THRESHOLD
    THRESHOLD = args.confidence

    yolo, id_model = load_models()

    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la fuente: {args.source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "CCTV - Monitoreo de Cerdas"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    paused = False
    frame_num = 0
    prev_time = time.time()
    fps = 0.0

    print(f"\nIniciando monitoreo desde: {args.source}")
    print("Controles: [q] Salir  [p] Pausar  [s] Guardar frame\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame — reconectando...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    print("Error crítico: No se pudo reconectar")
                    break
                continue

            frame_num += 1
            current_time = time.time()
            elapsed = current_time - prev_time
            prev_time = current_time
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if elapsed > 0 else 0

            results = yolo(frame, conf=YOLO_CONF, verbose=False)
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    if roi.size == 0:
                        continue

                    animal_id, id_conf = identify_pig(id_model, roi)
                    detections.append((x1, y1, x2, y2, conf, animal_id, id_conf))

                    log_detection(animal_id, id_conf, (x1, y1, x2, y2), frame_num)

            for x1, y1, x2, y2, det_conf, animal_id, id_conf in detections:
                if animal_id == "Desconocido":
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{animal_id} ({id_conf:.0%})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            frame = draw_info(frame, len(detections), fps, paused)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Finalizando monitoreo...")
            break
        elif key == ord("p"):
            paused = not paused
            print(f"{'Pausado' if paused else 'Reanudado'}")
        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = BASE_DIR / f"captura_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"Frame guardado: {filename}")

        if not paused:
            sleep_time = FRAME_INTERVAL - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Log guardado en: {LOG_PATH}")
    print(f"Total frames procesados: {frame_num}")


if __name__ == "__main__":
    main()
