"""
API FastAPI para Reconocimiento de Cerdas
Uso: uvicorn app_fastapi:app --reload
"""

import os
import io
import base64
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import cv2

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_identificacion_cerdos.h5")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_procesado")
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