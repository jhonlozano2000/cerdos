"""
Script de inferencia para el modelo de identificación de cerdos
Uso: python inference.py --image ruta/a/imagen.jpg
"""

import os
import numpy as np
from PIL import Image
import argparse
import json

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_identificacion_cerdos.h5")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_procesado")

CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_PATH) 
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

CLASS_MAPPING = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

THRESHOLD = 0.80

def load_model():
    import tensorflow as tf
    print(f"Cargando modelo desde: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_top3(model, image_path):
    img_array = preprocess_image(image_path)
    
    predictions = model.predict(img_array, verbose=0)[0]
    
    top_indices = np.argsort(predictions)[::-1][:3]
    
    results = []
    for idx in top_indices:
        confidence = float(predictions[idx])
        is_unknown = confidence < THRESHOLD
        
        results.append({
            "class_id": int(idx),
            "class_name": CLASS_MAPPING[idx] if not is_unknown else "Desconocido",
            "confidence": confidence,
            "is_unknown": is_unknown
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Identificación de cerdos')
    parser.add_argument('--image', required=True, help='Ruta a la imagen')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: La imagen no existe: {args.image}")
        return
    
    model = load_model()
    print(f"Modelo cargado. Clases disponibles: {CLASS_NAMES}")
    print(f"Umbral de confianza: {THRESHOLD}")
    
    results = predict_top3(model, args.image)
    
    print("\n=== Resultados Top 3 ===")
    for i, r in enumerate(results, 1):
        status = "⚠️ DESCONOCIDO" if r["is_unknown"] else "✅"
        print(f"{i}. {r['class_name']} - Confianza: {r['confidence']:.2%} {status}")

if __name__ == "__main__":
    main()
