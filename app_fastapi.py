"""
API FastAPI para Reconocimiento de Cerdas
Uso: uvicorn app_fastapi:app --reload
"""

import os
import io
import json
import base64
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import cv2
from pathlib import Path
from training_manager import manager

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_identificacion_cerdos.h5")
DATASET_PATH = r"C:\laragon\www\Porci-Integral-backend\storage\app\public\fotos_animales"
IMG_SIZE = (224, 224)
THRESHOLD = 0.50  # 50% confianza mínima

app = FastAPI(
    title="API Reconocimiento de Cerdas",
    description="Sistema de identificación biométrica de cerdas",
    version="1.0.0"
)

# CORS - permitir todas las conexiones
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar clases del modelo
def load_classes():
    if os.path.exists(DATASET_PATH):
        classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
        return classes
    return []

CLASS_NAMES = load_classes()
print(f"Clases cargadas: {len(CLASS_NAMES)} - {CLASS_NAMES}")

# Cargar modelo
model = None

def load_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            print(f"Cargando modelo desde: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Modelo cargado exitosamente")
        else:
            print(f"Modelo no encontrado en: {MODEL_PATH}")

# Procesar imagen
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predicción Top-3
def predict_top3(image_bytes):
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

# ============ ENDPOINTS ============

@app.get("/")
def root():
    return {
        "message": "API de Reconocimiento de Cerdas",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/salud")
def salud():
    return {"status": "ok", "modelo_cargado": model is not None}

@app.get("/clases")
def get_clases():
    return {
        "clases": CLASS_NAMES,
        "total": len(CLASS_NAMES),
        "threshold": THRESHOLD
    }

@app.post("/reconocer")
async def reconocer(file: UploadFile = File(...)):
    """Reconoce una cerda a partir de una imagen"""
    try:
        # Validar tipo de archivo
        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(status_code=400, detail="Tipo de imagen no válido")
        
        # Leer imagen
        image_bytes = await file.read()
        
        # Verificar tamaño máximo (10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen muy grande (máx 10MB)")
        
        # Realizar predicción
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
    """Reconoce una cerda a partir de una imagen en base64"""
    try:
        image_data = data.get("imagen", "")
        
        # Remover prefix data:image si existe
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        # Decodificar base64
        image_bytes = base64.b64decode(image_data)
        
        # Realizar predicción
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
    """Registra la confirmación del usuario"""
    # Por ahora solo logueamos - después guardaremos en DB
    print(f"Confirmación recibida: {data.class_name} - {data.confidence}")
    
    return {
        "success": True,
        "message": "Confirmación registrada",
        "data": data.dict()
    }

# YOLO Detection
from ultralytics import YOLO

YOLO_MODEL_PATH = r"C:\Users\Jhon\Desktop\cerdos\runs\detect\output\yolo_detector\runs\weights\best.pt"
yolo_model = None

def load_yolo():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO cargado: {yolo_model.model.names}")
    return yolo_model

@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    """Detecta cerdas en una imagen"""
    try:
        model = load_yolo()
        
        # Leer imagen
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # Detectar
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
    """Estado del detector YOLO"""
    model = load_yolo()
    return {
        "success": True,
        "modelo": "YOLOv8",
        "clases": model.model.names,
        "peso": YOLO_MODEL_PATH
    }

# ============ NUEVOS ENDPOINTS: EXPORTAR, ESTADISTICAS, ENTRENAR ============

class ExportarFotoRequest(BaseModel):
    class_name: str
    image_base64: str

@app.post("/exportar-foto")
async def exportar_foto(data: ExportarFotoRequest):
    """Guarda una foto en el dataset de entrenamiento"""
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
    """Retorna estadisticas del dataset de entrenamiento"""
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


class EntrenarRequest(BaseModel):
    include_classes: Optional[List[str]] = None
    exclude_classes: Optional[List[str]] = None


@app.post("/entrenar")
async def iniciar_entrenamiento(data: EntrenarRequest = None):
    """Inicia entrenamiento asincrono del modelo"""
    try:
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
    """Obtiene el estado de un entrenamiento"""
    status = manager.get_status(task_id)
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    return {"success": True, "task": status}


@app.get("/entrenar/historial")
async def historial_entrenamiento():
    """Obtiene historial de entrenamientos"""
    tasks = manager.get_all_tasks()
    return {"success": True, "tasks": tasks}


@app.post("/modelo/recargar")
async def recargar_modelo():
    """Recarga el modelo desde disco (despues de entrenamiento)"""
    global model, CLASS_NAMES
    try:
        model = None
        CLASS_NAMES = load_classes()
        load_model()
        return {"success": True, "message": "Modelo recargado exitosamente", "clases": len(CLASS_NAMES)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint de prueba
@app.post("/test")
async def test_endpoint(data: dict):
    return {
        "success": True,
        "message": "Test exitoso",
        "data": data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)