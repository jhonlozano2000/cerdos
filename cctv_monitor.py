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
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import utils


utils.load_env()
NEGATIVE_CLASS = utils.NEGATIVE_CLASS
BASE_DIR = utils.BASE_DIR
MODEL_PATH = BASE_DIR / "modelo_identificacion_cerdos.h5"
CLASSES_PATH = BASE_DIR / "classes.json"
LOG_PATH = BASE_DIR / "cctv_log.jsonl"
YOLO_MODEL_PATH = BASE_DIR / "runs" / "detect" / "output" / "yolo_detector" / "train" / "weights" / "best.pt"
YOLO_IS_CUSTOM = YOLO_MODEL_PATH.exists()
if not YOLO_IS_CUSTOM:
    YOLO_MODEL_PATH = BASE_DIR / "yolov8n.pt"
    logger.warning("Usando YOLO COCO (yolov8n.pt) — no tiene clase 'cerdo'. "
                   "Entrena YOLO custom para detección precisa.")

handler = RotatingFileHandler(BASE_DIR / "logs" / "cctv.log", maxBytes=100 * 1024 * 1024, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()])
logger = logging.getLogger(__name__)

IMG_SIZE = utils.IMG_SIZE
THRESHOLD = utils.load_threshold()
YOLO_CONF = 0.25
TARGET_FPS = 5
FRAME_INTERVAL = 1.0 / TARGET_FPS
FRAME_SKIP = 3  # Ejecutar YOLO cada N frames


CLASS_NAMES = utils.load_classes()
logger.info(f"Clases cargadas: {len(CLASS_NAMES)} - {CLASS_NAMES}")


def load_models():
    logger.info("Cargando YOLOv8...")
    yolo = YOLO(str(YOLO_MODEL_PATH))

    logger.info("Cargando MobileNetV2 (identificación)...")
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        logger.info("Modelo de identificación cargado")
    else:
        logger.warning(f"Modelo no encontrado en {MODEL_PATH}, identificación deshabilitada")
        model = None

    return yolo, model


def preprocess_image(roi):
    if len(roi.shape) != 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    img_array = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


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
    if class_name.lower() == NEGATIVE_CLASS.lower():
        return "Desconocido", confidence
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
        logger.error(f"No se pudo abrir la fuente: {args.source}")
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
    frame_counter = 0
    last_detections = []

    logger.info(f"Iniciando monitoreo desde: {args.source}")
    logger.info("Controles: [q] Salir  [p] Pausar  [s] Guardar frame")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Error al leer frame — reconectando...")
                cap.release()
                max_retries = 10
                for attempt in range(max_retries):
                    time.sleep(min(2 ** attempt, 30))
                    cap = cv2.VideoCapture(source)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    if cap.isOpened():
                        logger.info(f"Reconectado exitosamente (intento {attempt + 1})")
                        break
                    logger.warning(f"Reintento {attempt + 1}/{max_retries}...")
                else:
                    logger.error("No se pudo reconectar tras 10 intentos")
                    break
                continue

            frame_num += 1
            current_time = time.time()
            elapsed = current_time - prev_time
            prev_time = current_time
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if elapsed > 0 else 0

            frame_counter += 1

            # YOLO cada FRAME_SKIP frames; usar última detección entre medio
            if frame_counter % FRAME_SKIP == 0:
                results = yolo(frame, conf=YOLO_CONF, verbose=False)
                last_detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if YOLO_IS_CUSTOM and cls_id != 0:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])

                        roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        if roi.size == 0:
                            continue

                        animal_id, id_conf = identify_pig(id_model, roi)
                        last_detections.append((x1, y1, x2, y2, conf, animal_id, id_conf))

                        log_detection(animal_id, id_conf, (x1, y1, x2, y2), frame_num)

            for x1, y1, x2, y2, det_conf, animal_id, id_conf in last_detections:
                if animal_id == "Desconocido":
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{animal_id} ({id_conf:.0%})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            frame = draw_info(frame, len(last_detections), fps, paused)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            logger.info("Finalizando monitoreo...")
            break
        elif key == ord("p"):
            paused = not paused
            logger.info(f"{'Pausado' if paused else 'Reanudado'}")
            if not paused:
                prev_time = time.time()
                current_time = prev_time
        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = BASE_DIR / f"captura_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            logger.info(f"Frame guardado: {filename}")

        if not paused:
            sleep_time = FRAME_INTERVAL - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Log guardado en: {LOG_PATH}")
    logger.info(f"Total frames procesados: {frame_num}")


if __name__ == "__main__":
    main()
