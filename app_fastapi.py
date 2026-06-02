"""
API FastAPI para Reconocimiento de Cerdas
=========================================
Servicio principal del módulo de IA. Expone endpoints para:
- Reconocimiento biométrico (MobileNetV2)
- Detección de objetos (YOLOv8)
- Gestión del dataset
- Entrenamiento asíncrono
- Backup y restauración de modelos

Uso: uvicorn app_fastapi:app --reload
"""

import os
import io
import json
import base64
import shutil
import threading
import re
import logging
from datetime import datetime
from time import time
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
from pathlib import Path
from training_manager import manager, TASKS_DIR
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from ultralytics import YOLO
import utils


# ── Configuración ────────────────────────────────────────────────
utils.load_env()
NEGATIVE_CLASS = utils.NEGATIVE_CLASS

BASE_DIR = utils.BASE_DIR
MODEL_PATH = BASE_DIR / "modelo_identificacion_cerdos.h5"
MODELS_BACKUP_DIR = BASE_DIR / "model_backups"
MAX_BACKUPS = 3
IMG_SIZE = utils.IMG_SIZE
THRESHOLD = utils.load_threshold()

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    r"C:\laragon\www\Porci-Integral-backend\storage\app\public\datasets\animales"
)
RECONOCIMIENTOS_PATH = os.environ.get(
    "RECONOCIMIENTOS_PATH",
    r"C:\laragon\www\Porci-Integral-backend\storage\app\public\datasets\reconocimientos"
)

MODELS_BACKUP_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Cache de estadísticas ────────────────────────────────────────
_stats_cache = {"data": None, "timestamp": 0}
CACHE_TTL = 60


# ── Funciones de Backup del Modelo ─────────────────────────────

def backup_current_model():
    """
    Crea una copia del modelo actual antes de entrenar.
    Mantiene hasta MAX_BACKUPS versiones (3 por defecto).
    También respalda classes.json asociado.
    """
    if not MODEL_PATH.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"modelo_{timestamp}.h5"
    backup_path = MODELS_BACKUP_DIR / backup_name
    shutil.copy2(str(MODEL_PATH), str(backup_path))

    # Elimina backups excedentes (los más viejos)
    backups = sorted(MODELS_BACKUP_DIR.glob("modelo_*.h5"), reverse=True)
    for old in backups[MAX_BACKUPS:]:
        old.unlink()

    classes_src = BASE_DIR / "classes.json"
    if classes_src.exists():
        backup_classes = MODELS_BACKUP_DIR / f"classes_{timestamp}.json"
        shutil.copy2(str(classes_src), str(backup_classes))

    return backup_name


def list_model_backups():
    """Retorna metadatos de todos los backups disponibles."""
    backups = []
    for f in sorted(MODELS_BACKUP_DIR.glob("modelo_*.h5"), reverse=True):
        ts = f.stem.replace("modelo_", "")
        stats = f.stat()
        size_mb = stats.st_size / (1024 * 1024)
        classes_file = MODELS_BACKUP_DIR / f"classes_{ts}.json"
        backups.append({
            "version": ts,
            "filename": f.name,
            "size_mb": round(size_mb, 1),
            "created": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "has_classes": classes_file.exists(),
        })
    return backups


# ── App FastAPI ─────────────────────────────────────────────────

app = FastAPI(
    title="API Reconocimiento de Cerdas",
    description="Sistema de identificación biométrica de cerdas",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://porci-integral-backend.test", "http://127.0.0.1:5173", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiting ───────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# ── API Key Auth ────────────────────────────────────────────────
API_KEY = os.environ.get("API_KEY", "")

async def verify_admin_key(x_api_key: str = Header(None)):
    if not API_KEY:
        return True  # No key configured = no auth
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="API Key inválida")
    return True


# ── Carga del Modelo y Utilidades ──────────────────────────────

CLASS_NAMES = utils.load_classes()
THRESHOLD = utils.load_threshold()
logger.info(f"Clases cargadas: {len(CLASS_NAMES)} - {CLASS_NAMES}")

model_lock = threading.Lock()
model = None


def load_model():
    """
    Carga el modelo MobileNetV2 desde disco.
    Usa lazy loading: solo carga en el primer request.
    """
    global model
    if model is None:
        with model_lock:
            if model is None:  # Double-checked locking
                if os.path.exists(MODEL_PATH):
                    logger.info(f"Cargando modelo desde: {MODEL_PATH}")
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    logger.info("Modelo cargado exitosamente")
                else:
                    logger.warning(f"Modelo no encontrado en: {MODEL_PATH}")


def preprocess_image(image_bytes):
    return utils.preprocess_image(image_bytes)


def predict_top3(image_bytes, use_tta=True):
    """
    Ejecuta inferencia con Test-Time Augmentation y retorna las 3 clases
    con mayor probabilidad. Las predicciones por debajo de THRESHOLD
    se marcan como "Desconocido".
    """
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    if use_tta:
        tta_inputs = utils.preprocess_image_tta(image_bytes)
        all_preds = []
        for aug_array, _label in tta_inputs:
            with model_lock:
                preds = model.predict(aug_array, verbose=0)[0]
            all_preds.append(preds)
        avg_preds = np.mean(all_preds, axis=0)
        predictions = avg_preds
    else:
        img_array = preprocess_image(image_bytes)
        with model_lock:
            predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []
    for idx in top_indices:
        confidence = float(predictions[idx])
        is_unknown = confidence < THRESHOLD
        class_name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"clase_{idx}"

        is_no_cerdo = class_name.lower() == NEGATIVE_CLASS.lower()
        display_name = "Desconocido" if (is_unknown or is_no_cerdo) else class_name

        results.append({
            "class_id": int(idx),
            "class_name": display_name,
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.1f}%",
            "is_unknown": is_unknown or is_no_cerdo
        })

    return results


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Reconocimiento
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
def _precargar_modelo():
    """Precarga modelo y clases al iniciar el servicio."""
    if MODEL_PATH.exists():
        load_model()
        logger.info(f"Modelo precargado desde: {MODEL_PATH}")
    else:
        logger.warning(f"Modelo no encontrado en startup: {MODEL_PATH}")


@app.get("/")
def root():
    """Health-check simple."""
    return {
        "message": "API de Reconocimiento de Cerdas",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/ready")
def ready():
    """Healthcheck readiness — intenta cargar el modelo si no lo está."""
    load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {"status": "ready", "modelo_cargado": True}


@app.get("/salud")
def salud():
    """Estado completo del servicio: modelo, dataset, clases."""
    load_model()
    return {
        "status": "ok",
        "modelo_cargado": model is not None,
        "modelo_existe": MODEL_PATH.exists(),
        "modelo_ruta": str(MODEL_PATH),
        "dataset_path": DATASET_PATH,
        "dataset_existe": os.path.isdir(DATASET_PATH),
        "clases_disponibles": len(CLASS_NAMES),
        "clases": CLASS_NAMES,
        "threshold": THRESHOLD,
        "version": "1.0.0",
        "rate_limiting": "activo",
        "api_key_configured": bool(API_KEY),
    }


@app.get("/clases")
def get_clases():
    """Lista todas las clases disponibles para clasificación."""
    return {
        "clases": CLASS_NAMES,
        "total": len(CLASS_NAMES),
        "threshold": THRESHOLD
    }


@app.post("/reconocer")
@limiter.limit("60/minute")
async def reconocer(request: Request, file: UploadFile = File(...)):
    """
    Reconoce una cerda a partir de una imagen subida.
    Acepta: image/jpeg, image/png, image/webp (máx 10MB).
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(status_code=400, detail="Tipo de imagen no válido")

        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen muy grande (máx 10MB)")

        results = predict_top3(image_bytes)

        return {
            "success": True,
            "resultados": results,
            "top1": results[0]["class_name"] if results else None,
            "confianza_top1": results[0]["confidence_pct"] if results else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en reconocimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reconocer_base64")
@limiter.limit("60/minute")
async def reconocer_base64(request: Request, data: dict):
    """
    Reconoce una cerda a partir de una imagen en base64.
    Útil para integración con cámaras o dispositivos IoT.
    """
    try:
        image_data = data.get("imagen", "")
        if not isinstance(image_data, str) or not image_data:
            raise HTTPException(status_code=400, detail="Campo 'imagen' requerido")

        # Remueve el prefijo data:image/...;base64, si existe
        if "," in image_data:
            image_data = image_data.split(",")[1]

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Base64 inválido")

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen muy grande (máx 10MB)")

        results = predict_top3(image_bytes)

        return {
            "success": True,
            "resultados": results,
            "top1": results[0]["class_name"] if results else None,
            "confianza_top1": results[0]["confidence_pct"] if results else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en reconocimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ReconocerCompletoRequest(BaseModel):
    imagen: str
    incluir_deteccion: bool = True


@app.post("/reconocer-completo")
@limiter.limit("60/minute")
async def reconocer_completo(request: Request, data: ReconocerCompletoRequest):
    """Combina YOLO detection + MobileNetV2 classification in one call."""
    try:
        image_data = data.imagen
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # MobileNetV2 classification
        tf_results = predict_top3(image_bytes)
        top1 = tf_results[0]["class_name"] if tf_results else "Desconocido"

        result = {
            "success": True,
            "tensorflow_predictions": tf_results,
            "identified_as": top1,
            "confidence_top1": tf_results[0]["confidence_pct"] if tf_results else "0%",
            "yolo_detections": [],
            "total_detections": 0,
        }

        # YOLO detection if requested
        if data.incluir_deteccion:
            try:
                yolo = load_yolo()
                img = Image.open(io.BytesIO(image_bytes))
                yolo_results = yolo(img, conf=0.25, verbose=False)
                detections = []
                for r in yolo_results:
                    boxes = r.boxes
                    for box in boxes:
                        detections.append({
                            "clase": yolo.model.names[int(box.cls[0])],
                            "confianza": round(float(box.conf[0]) * 100),
                            "bbox": box.xyxy[0].tolist()
                        })
                result["yolo_detections"] = detections
                result["total_detections"] = len(detections)
            except Exception as yolo_err:
                logger.warning(f"YOLO detection error (non-fatal): {yolo_err}")

        return result
    except Exception as e:
        logger.error(f"Error en reconocer-completo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ConfirmacionRequest(BaseModel):
    class_name: str
    confidence: float
    animal_id: Optional[int] = None
    foto_base64: Optional[str] = None
    clase_confirmada: Optional[str] = None
    usuario_id: Optional[int] = None


@app.post("/confirmar")
async def confirmar(data: ConfirmacionRequest):
    """
    Registra la confirmación del usuario sobre una predicción.
    Se usa para retroalimentación y mejora del modelo.
    """
    try:
        clase_confirmada = data.clase_confirmada or data.class_name

        # Persist confirmation to log
        confirm_log_path = BASE_DIR / "confirmaciones_log.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "clase_predicha": data.class_name,
            "clase_confirmada": clase_confirmada,
            "confidence": data.confidence,
            "animal_id": data.animal_id,
            "usuario_id": data.usuario_id,
        }
        with open(confirm_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Save image to dataset if provided
        if data.foto_base64:
            try:
                image_data = data.foto_base64
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)

                class_dir = os.path.join(RECONOCIMIENTOS_PATH, clase_confirmada)
                os.makedirs(class_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                uuid_str = base64.urlsafe_b64encode(os.urandom(6)).decode().rstrip("=")
                filename = f"confirm_{timestamp}_{uuid_str}.jpg"
                filepath = os.path.join(class_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                entry["imagen_guardada"] = filename
            except Exception as img_err:
                logger.error(f"Error guardando imagen de confirmacion: {img_err}")

        # Count confirmations to trigger retrain if enough
        try:
            with open(confirm_log_path) as _cl:
                total_confirmaciones = sum(1 for _ in _cl)
                entry["total_confirmaciones"] = total_confirmaciones
        except Exception:
            pass

        return {
            "success": True,
            "message": "Confirmación registrada",
            "data": entry
        }
    except Exception as e:
        logger.error(f"Error en confirmar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

 
# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — YOLOv8 (Detección de objetos)
# ═══════════════════════════════════════════════════════════════

# Busca automáticamente el peso YOLO en distintas ubicaciones posibles
yolo_model_paths = [
    BASE_DIR / "runs" / "detect" / "output" / "yolo_detector" / "train" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "output" / "yolo_detector" / "runs" / "weights" / "best.pt",
    BASE_DIR / "yolov8n.pt",
]
YOLO_MODEL_PATH = next((str(p) for p in yolo_model_paths if p.exists()), str(yolo_model_paths[-1]))
yolo_model = None


def load_yolo():
    """Carga el modelo YOLO con lazy loading."""
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info(f"YOLO cargado: {yolo_model.model.names}")
    return yolo_model


@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    """
    Detecta cerdas en una imagen usando YOLOv8.
    Retorna bounding boxes, clase y confianza por detección.
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(status_code=400, detail="Tipo de imagen no válido")

        yolo = load_yolo()

        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        results = yolo(img, conf=0.25, verbose=False)

        deteccions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                deteccions.append({
                    "clase": yolo.model.names[int(box.cls[0])],
                    "confianza": round(float(box.conf[0]) * 100),
                    "bbox": box.xyxy[0].tolist()
                })

        return {
            "success": True,
            "detecciones": len(deteccions),
            "resultados": deteccions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en detección: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detector/status")
async def detector_status():
    """Estado y configuración del detector YOLO."""
    yolo = load_yolo()
    return {
        "success": True,
        "modelo": "YOLOv8",
        "clases": yolo.model.names,
        "peso": YOLO_MODEL_PATH
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Detección de Enfermedades
# ═══════════════════════════════════════════════════════════════

class EnfermedadRequest(BaseModel):
    animal_id: str
    peso_actual_kg: float
    peso_anterior_kg: float
    dias_entre_pesajes: int
    edad_dias: Optional[int] = None
    temperatura_opcional: Optional[float] = None


@app.post("/detectar-enfermedad")
async def detectar_enfermedad(data: EnfermedadRequest):
    """
    Evalúa riesgo de enfermedad basado en cambios de peso y temperatura.
    Reglas:
      - Pérdida > 5% en 7 días → ALTO riesgo
      - Pérdida > 3% en 7 días → MEDIO riesgo
      - Sin ganancia en joven (<90 días) → MEDIO riesgo
      - Normal → BAJO riesgo
    """
    try:
        peso_actual = data.peso_actual_kg
        peso_anterior = data.peso_anterior_kg
        dias = data.dias_entre_pesajes
        edad = data.edad_dias
        temperatura = data.temperatura_opcional

        factores = []
        riesgo = "BAJO"
        alerta = False
        cambio_pct = None

        if peso_anterior > 0 and dias > 0:
            cambio_pct = ((peso_actual - peso_anterior) / peso_anterior) * 100
            perdida_diaria = cambio_pct / dias if dias > 0 else 0

            perdida_pct_7d = perdida_diaria * 7

            if perdida_pct_7d < -5:
                riesgo = "ALTO"
                alerta = True
                factores.append(f"Pérdida de peso severa: {cambio_pct:+.1f}% en {dias} días")
            elif perdida_pct_7d < -3:
                riesgo = "MEDIO"
                factores.append(f"Pérdida de peso moderada: {cambio_pct:+.1f}% en {dias} días")
            elif edad is not None and edad < 90 and cambio_pct <= 0:
                riesgo = "MEDIO"
                factores.append(f"Animal joven ({edad} días) sin ganancia de peso")
            else:
                factores.append(f"Peso estable o ganancia: {cambio_pct:+.1f}% en {dias} días")
        else:
            factores.append("Datos insuficientes para evaluar cambio de peso")

        if temperatura is not None:
            if temperatura >= 40:
                riesgo = "ALTO"
                alerta = True
                factores.append(f"Fiebre detectada: {temperatura}°C")
            elif temperatura >= 39.5:
                if riesgo != "ALTO":
                    riesgo = "MEDIO"
                factores.append(f"Temperatura elevada: {temperatura}°C")
            else:
                factores.append(f"Temperatura normal: {temperatura}°C")

        recomendaciones = {
            "ALTO": "Aislar al animal, notificar al veterinario de inmediato, realizar revisión clínica completa, considerar tomar muestra para laboratorio.",
            "MEDIO": "Monitorear cada 6 horas, registrar temperatura y peso diariamente, evaluar apetito y comportamiento, preparar plan de intervención.",
            "BAJO": "Continuar monitoreo rutinario, mantener programa de alimentación y manejo sanitario habitual.",
        }

        return {
            "success": True,
            "animal_id": data.animal_id,
            "riesgo": riesgo,
            "factores": factores,
            "recomendacion": recomendaciones.get(riesgo, ""),
            "alerta": alerta,
            "cambio_peso_pct": round(cambio_pct, 2) if peso_anterior > 0 and dias > 0 else None,
        }

    except Exception as e:
        logger.error(f"Error en detección de enfermedad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Dataset
# ═══════════════════════════════════════════════════════════════

class ExportarFotoRequest(BaseModel):
    class_name: str
    image_base64: str


@app.post("/exportar-foto")
@limiter.limit("10/minute")
async def exportar_foto(request: Request, data: ExportarFotoRequest, auth: bool = Depends(verify_admin_key)):
    """
    Guarda una foto directamente en el dataset de entrenamiento.
    Ruta final: {DATASET_PATH}/{class_name}/foto.jpg
    """
    try:
        class_name = data.class_name.strip()
        if not class_name:
            raise HTTPException(status_code=400, detail="class_name es requerido")
        if not re.match(r'^[a-zA-Z0-9_\-]+$', class_name):
            raise HTTPException(status_code=400, detail="class_name contiene caracteres inválidos")

        class_dir = os.path.join(DATASET_PATH, class_name)
        os.makedirs(class_dir, exist_ok=True)

        image_data = data.image_base64
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{class_name}_{timestamp}.jpg"
        filepath = os.path.join(class_dir, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        return {"success": True, "filename": filename, "class_name": class_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset-estadisticas")
async def dataset_estadisticas():
    """
    Retorna estadísticas del dataset:
    - Lista de clases con conteo de imágenes
    - Total global de imágenes y clases
    """
    global _stats_cache
    now = time()
    if _stats_cache["data"] is not None and (now - _stats_cache["timestamp"]) < CACHE_TTL:
        return _stats_cache["data"]

    try:
        if not os.path.exists(DATASET_PATH):
            data = {"clases": [], "total_global": 0, "total_clases": 0}
            _stats_cache = {"data": data, "timestamp": now}
            return data

        clases = []
        total_global = 0
        for d in sorted(os.listdir(DATASET_PATH)):
            dir_path = os.path.join(DATASET_PATH, d)
            if os.path.isdir(dir_path):
                count = len([f for f in os.listdir(dir_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                clases.append({"nombre": d, "total": count})
                total_global += count

        data = {
            "clases": clases,
            "total_global": total_global,
            "total_clases": len(clases),
        }
        _stats_cache = {"data": data, "timestamp": now}
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training-stats")
async def training_stats():
    """
    Retorna estadísticas para el pipeline de reentrenamiento:
    - Total de fotos en la carpeta fuente (imagenes_originales)
    - Conteo por clase
    """
    try:
        source_path = r"D:\cerdos\imagenes_originales"
        if not os.path.exists(source_path):
            return {
                "source_path": source_path,
                "total_source_photos": 0,
                "classes": [],
                "error": None
            }

        classes_data = []
        total = 0
        for d in sorted(os.listdir(source_path)):
            dir_path = os.path.join(source_path, d)
            if os.path.isdir(dir_path):
                count = len([f for f in os.listdir(dir_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                classes_data.append({"class": d, "count": count})
                total += count

        return {
            "source_path": source_path,
            "total_source_photos": total,
            "classes": classes_data,
            "error": None
        }
    except Exception as e:
        return {
            "source_path": None,
            "total_source_photos": 0,
            "classes": [],
            "error": str(e)
        }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Entrenamiento
# ═══════════════════════════════════════════════════════════════

class EntrenarRequest(BaseModel):
    include_classes: Optional[List[str]] = None
    exclude_classes: Optional[List[str]] = None
    config: Optional[dict] = None


@app.get("/ml/training-config")
async def training_config():
    """
    Retorna la configuración por defecto del script de entrenamiento.
    Usada por el frontend para llenar el panel de configuración avanzada.
    """
    return {
        "batch_size": 16,
        "epochs_frozen": 60,
        "epochs_finetune": 80,
        "lr": 0.0001,
        "finetune_lr": 0.000001,
        "max_images_per_class": 200,
        "max_no_cerdo": 80,
        "unfreeze_layers": 8,
        "disable_mixup": False,
        "mixup_alpha": 0.2,
        "oversample_min": 80,
        "focal_gamma": 1.5,
        "disable_class_weights": False,
    }


@app.post("/entrenar")
@limiter.limit("10/minute")
async def iniciar_entrenamiento(request: Request, data: EntrenarRequest = None, auth: bool = Depends(verify_admin_key)):
    """
    Inicia entrenamiento asíncrono del modelo MobileNetV2.
    - Crea backup automático del modelo actual
    - Delega a training_manager que corre en un thread separado
    - Opcional: include_classes para entrenar solo clases específicas
    - Opcional: config con hiperparámetros avanzados
    """
    try:
        backup_current_model()

        include = data.include_classes if data else None
        exclude = data.exclude_classes if data else None
        config = data.config if data else None

        task_id = manager.start_training(include_classes=include, exclude_classes=exclude, config=config)
        return {"success": True, "task_id": task_id, "message": "Entrenamiento iniciado"}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entrenar/historial")
async def historial_entrenamiento():
    """Historial completo de entrenamientos realizados."""
    tasks = manager.get_all_tasks()
    return {"success": True, "tasks": tasks}


@app.get("/entrenar/{task_id}")
async def estado_entrenamiento(task_id: str):
    """Consulta el estado actual de una tarea de entrenamiento."""
    status = manager.get_status(task_id)
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

    # Calcular elapsed_seconds si started_at está presente
    started_at = status.get("started_at")
    if started_at:
        try:
            start_dt = datetime.fromisoformat(started_at)
            elapsed = (datetime.now() - start_dt).total_seconds()
            status["elapsed_seconds"] = int(elapsed)
        except Exception:
            pass

    return {"success": True, "task": status}


@app.get("/entrenar/{task_id}/log")
async def obtener_log_entrenamiento(task_id: str):
    """Retorna el log de error completo de una tarea de entrenamiento."""
    logs_dir = BASE_DIR / "logs"
    log_path = logs_dir / f"train_{task_id}_error.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log no disponible")
    return {"success": True, "log": log_path.read_text(encoding="utf-8")}


@app.get("/entrenar/{task_id}/history")
async def obtener_history_entrenamiento(task_id: str):
    """Retorna el historial de métricas por época (phase1 + phase2)."""
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    history = {}
    for phase_file in ["history_phase1.json", "history_phase2.json"]:
        path = task_dir / phase_file
        if path.exists():
            phase = phase_file.replace("history_", "").replace(".json", "")
            history[phase] = json.loads(path.read_text(encoding="utf-8"))
    return {"success": True, "history": history}


@app.post("/entrenar/{task_id}/cancel")
async def cancelar_entrenamiento(task_id: str, auth: bool = Depends(verify_admin_key)):
    """Cancela una tarea de entrenamiento en ejecución."""
    ok = manager.cancel_training(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="No se puede cancelar la tarea (no encontrada o ya finalizada)")
    return {"success": True, "message": "Entrenamiento cancelado"}


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Gestión del Modelo
# ═══════════════════════════════════════════════════════════════

@app.post("/modelo/recargar")
@limiter.limit("10/minute")
async def recargar_modelo(request: Request, auth: bool = Depends(verify_admin_key)):
    """
    Recarga el modelo desde disco.
    Útil después de un entrenamiento para usar el nuevo modelo sin reiniciar el servicio.
    """
    global model, CLASS_NAMES, THRESHOLD
    try:
        with model_lock:
            model = None
            CLASS_NAMES = load_classes()
            _reload_threshold()
            load_model()
        return {"success": True, "message": "Modelo recargado exitosamente", "clases": len(CLASS_NAMES)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modelo/versiones")
async def modelo_versiones():
    """Lista todas las versiones de backup disponibles para restauración."""
    return {
        "success": True,
        "backups": list_model_backups(),
        "actual": MODEL_PATH.name if MODEL_PATH.exists() else None
    }


class RestaurarRequest(BaseModel):
    version: str


@app.post("/modelo/restaurar")
@limiter.limit("10/minute")
async def modelo_restaurar(request: Request, data: RestaurarRequest, auth: bool = Depends(verify_admin_key)):
    """
    Restaura un modelo desde un backup existente.
    - Crea backup del modelo actual antes de restaurar
    - Recarga automáticamente el modelo restaurado
    """
    global model, CLASS_NAMES
    try:
        backup_file = MODELS_BACKUP_DIR / f"modelo_{data.version}.h5"
        if not backup_file.exists():
            raise HTTPException(status_code=404, detail=f"Version {data.version} no encontrada")

        backup_current_model()

        shutil.copy2(str(backup_file), str(MODEL_PATH))
        classes_file = MODELS_BACKUP_DIR / f"classes_{data.version}.json"
        if classes_file.exists():
            shutil.copy2(str(classes_file), str(BASE_DIR / "classes.json"))

        with model_lock:
            model = None
            CLASS_NAMES = load_classes()
            load_model()

        return {
            "success": True,
            "message": f"Modelo restaurado a version {data.version}",
            "clases": len(CLASS_NAMES),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test")
async def test_endpoint(data: dict):
    """Endpoint de prueba para verificar conectividad."""
    return {
        "success": True,
        "message": "Test exitoso",
        "data": data
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
