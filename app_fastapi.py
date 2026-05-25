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
from datetime import datetime
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
from pathlib import Path
from training_manager import manager


# ── Configuración ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_identificacion_cerdos.h5"
MODELS_BACKUP_DIR = BASE_DIR / "model_backups"
MAX_BACKUPS = 3

# Carga manual del .env (sin python-dotenv para evitar dependencias extras)
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip()

# Ruta al dataset — configurable via .env o variable de entorno
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    r"C:\laragon\www\Porci-Integral-backend\storage\app\public\fotos_animales"
)
IMG_SIZE = (224, 224)
THRESHOLD = 0.50  # Mínimo de confianza para considerar una predicción válida

MODELS_BACKUP_DIR.mkdir(exist_ok=True)


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Carga del Modelo y Utilidades ──────────────────────────────

def load_classes():
    """
    Escanea DATASET_PATH y retorna los nombres de directorio
    como lista de clases disponibles para clasificación.
    """
    if os.path.exists(DATASET_PATH):
        classes = sorted([d for d in os.listdir(DATASET_PATH)
                         if os.path.isdir(os.path.join(DATASET_PATH, d))])
        return classes
    return []


CLASS_NAMES = load_classes()
print(f"Clases cargadas: {len(CLASS_NAMES)} - {CLASS_NAMES}")

model = None


def load_model():
    """
    Carga el modelo MobileNetV2 desde disco.
    Usa lazy loading: solo carga en el primer request.
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            print(f"Cargando modelo desde: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Modelo cargado exitosamente")
        else:
            print(f"Modelo no encontrado en: {MODEL_PATH}")


def preprocess_image(image_bytes):
    """
    Convierte bytes de imagen a un array numpy normalizado
    listo para la inferencia de MobileNetV2.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_top3(image_bytes):
    """
    Ejecuta inferencia y retorna las 3 clases con mayor probabilidad.
    Las predicciones por debajo de THRESHOLD se marcan como "Desconocido".
    """
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []
    for idx in top_indices:
        confidence = float(predictions[idx])
        is_unknown = confidence < THRESHOLD

        results.append({
            "class_id": int(idx),
            "class_name": CLASS_NAMES[idx] if not is_unknown else "Desconocido",
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.1f}%",
            "is_unknown": is_unknown
        })

    return results


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Reconocimiento
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Health-check simple."""
    return {
        "message": "API de Reconocimiento de Cerdas",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/salud")
def salud():
    """Estado completo del servicio: modelo, dataset, clases."""
    return {
        "status": "ok",
        "modelo_cargado": model is not None,
        "modelo_existe": MODEL_PATH.exists(),
        "dataset_path": DATASET_PATH,
        "dataset_existe": os.path.isdir(DATASET_PATH),
        "clases_disponibles": len(CLASS_NAMES),
        "version": "1.0.0",
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
async def reconocer(file: UploadFile = File(...)):
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

    except Exception as e:
        print(f"Error en reconocimiento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reconocer_base64")
async def reconocer_base64(data: dict):
    """
    Reconoce una cerda a partir de una imagen en base64.
    Útil para integración con cámaras o dispositivos IoT.
    """
    try:
        image_data = data.get("imagen", "")

        # Remueve el prefijo data:image/...;base64, si existe
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        results = predict_top3(image_bytes)

        return {
            "success": True,
            "resultados": results,
            "top1": results[0]["class_name"] if results else None,
            "confianza_top1": results[0]["confidence_pct"] if results else None
        }

    except Exception as e:
        print(f"Error en reconocimiento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class ConfirmacionRequest(BaseModel):
    class_name: str
    confidence: float
    animal_id: Optional[int] = None
    foto_base64: Optional[str] = None


@app.post("/confirmar")
async def confirmar(data: ConfirmacionRequest):
    """
    Registra la confirmación del usuario sobre una predicción.
    Se usa para retroalimentación y mejora del modelo.
    """
    print(f"Confirmación recibida: {data.class_name} - {data.confidence}")

    return {
        "success": True,
        "message": "Confirmación registrada",
        "data": data.dict()
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — YOLOv8 (Detección de objetos)
# ═══════════════════════════════════════════════════════════════

from ultralytics import YOLO

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
        print(f"YOLO cargado: {yolo_model.model.names}")
    return yolo_model


@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    """
    Detecta cerdas en una imagen usando YOLOv8.
    Retorna bounding boxes, clase y confianza por detección.
    """
    try:
        model = load_yolo()

        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, conf=0.25, verbose=False)

        deteccions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                deteccions.append({
                    "clase": model.model.names[int(box.cls[0])],
                    "confianza": round(float(box.conf[0]) * 100),
                    "bbox": box.xyxy[0].tolist()
                })

        return {
            "success": True,
            "detecciones": len(deteccions),
            "resultados": deteccions
        }

    except Exception as e:
        print(f"Error detección: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detector/status")
async def detector_status():
    """Estado y configuración del detector YOLO."""
    model = load_yolo()
    return {
        "success": True,
        "modelo": "YOLOv8",
        "clases": model.model.names,
        "peso": YOLO_MODEL_PATH
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Dataset
# ═══════════════════════════════════════════════════════════════

class ExportarFotoRequest(BaseModel):
    class_name: str
    image_base64: str


@app.post("/exportar-foto")
async def exportar_foto(data: ExportarFotoRequest):
    """
    Guarda una foto directamente en el dataset de entrenamiento.
    Ruta final: {DATASET_PATH}/{class_name}/foto.jpg
    """
    try:
        class_name = data.class_name.strip()
        if not class_name:
            raise HTTPException(status_code=400, detail="class_name es requerido")

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset-estadisticas")
async def dataset_estadisticas():
    """
    Retorna estadísticas del dataset:
    - Lista de clases con conteo de imágenes
    - Total global de imágenes y clases
    """
    try:
        if not os.path.exists(DATASET_PATH):
            return {"clases": [], "total_global": 0, "total_clases": 0}

        clases = []
        total_global = 0
        for d in sorted(os.listdir(DATASET_PATH)):
            dir_path = os.path.join(DATASET_PATH, d)
            if os.path.isdir(dir_path):
                count = len([f for f in os.listdir(dir_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                clases.append({"nombre": d, "total": count})
                total_global += count

        return {
            "clases": clases,
            "total_global": total_global,
            "total_clases": len(clases),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Entrenamiento
# ═══════════════════════════════════════════════════════════════

class EntrenarRequest(BaseModel):
    include_classes: Optional[List[str]] = None
    exclude_classes: Optional[List[str]] = None


@app.post("/entrenar")
async def iniciar_entrenamiento(data: EntrenarRequest = None):
    """
    Inicia entrenamiento asíncrono del modelo MobileNetV2.
    - Crea backup automático del modelo actual
    - Delega a training_manager que corre en un thread separado
    - Opcional: include_classes para entrenar solo clases específicas
    """
    try:
        backup_current_model()

        include = data.include_classes if data else None
        exclude = data.exclude_classes if data else None

        task_id = manager.start_training(include_classes=include, exclude_classes=exclude)
        return {"success": True, "task_id": task_id, "message": "Entrenamiento iniciado"}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entrenar/{task_id}")
async def estado_entrenamiento(task_id: str):
    """Consulta el estado actual de una tarea de entrenamiento."""
    status = manager.get_status(task_id)
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    return {"success": True, "task": status}


@app.get("/entrenar/historial")
async def historial_entrenamiento():
    """Historial completo de entrenamientos realizados."""
    tasks = manager.get_all_tasks()
    return {"success": True, "tasks": tasks}


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Gestión del Modelo
# ═══════════════════════════════════════════════════════════════

@app.post("/modelo/recargar")
async def recargar_modelo():
    """
    Recarga el modelo desde disco.
    Útil después de un entrenamiento para usar el nuevo modelo sin reiniciar el servicio.
    """
    global model, CLASS_NAMES
    try:
        model = None
        CLASS_NAMES = load_classes()
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
async def modelo_restaurar(data: RestaurarRequest):
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
