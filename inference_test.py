"""
Script de inferencia para el modelo de identificación de cerdas
Uso: python inference_test.py
"""

import os
import numpy as np
from pathlib import Path

MODEL_PATH = os.path.join(os.path.dirname(__file__), "output", "modelo_final.keras")
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
    from PIL import Image
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

def test_from_test_set(model, num_samples=10):
    """Prueba con imágenes del test set"""
    from pathlib import Path
    import random
    
    test_split = Path("output/test_split.csv")
    if not test_split.exists():
        print("❌ No se encontró test_split.csv")
        return
    
    import pandas as pd
    df = pd.read_csv(test_split)
    
    print(f"\n🧪 Probando con {num_samples} imágenes del test set...")
    print("=" * 60)
    
    correct = 0
    total = 0
    
    samples = df.sample(min(num_samples, len(df)), random_state=42)
    
    for _, row in samples[0].iterrows():
        image_path = row['path']
        true_class = row['class_name']
        
        if not os.path.exists(image_path):
            continue
        
        results = predict_top3(model, image_path)
        predicted = results[0]['class_name']
        confidence = results[0]['confidence']
        
        is_correct = predicted == true_class
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {os.path.basename(image_path)}")
        print(f"   Real: {true_class}")
        print(f"   Predicho: {predicted} ({confidence:.1%})")
        
        if len(results) > 1:
            print(f"   Alternativas: {results[1]['class_name']} ({results[1]['confidence']:.1%}), {results[2]['class_name']} ({results[2]['confidence']:.1%})")
        print()
    
    accuracy = correct / total if total > 0 else 0
    print("=" * 60)
    print(f"📊 Accuracy: {correct}/{total} ({accuracy:.1%})")
    
    return accuracy

def test_single_image(model, image_path):
    """Prueba con una sola imagen"""
    if not os.path.exists(image_path):
        print(f"❌ No encontré la imagen: {image_path}")
        return
    
    print(f"\n🖼️  Probando con: {image_path}")
    print("=" * 60)
    
    results = predict_top3(model, image_path)
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['class_name']}: {r['confidence']:.1%}")
    
    print(f"\n🎯 Predicción: {results[0]['class_name']} ({results[0]['confidence']:.1%})")

def main():
    print("=" * 60)
    print("🎯 Script de Inferencia - Identificación de Cerdas")
    print("=" * 60)
    print(f"\n📂 Modelo: {MODEL_PATH}")
    print(f"📂 Clases: {NUM_CLASSES}")
    print(f"   {CLASS_NAMES}")
    
    model = load_model()
    
    print("\n" + "=" * 60)
    print("ELEGIR MODO DE PRUEBA:")
    print("1. Probar imágenes del test set")
    print("2. Probar una imagen específica")
    print("0. Salir")
    print("=" * 60)
    
    opcion = input("\nOpción: ").strip()
    
    if opcion == "1":
        num = input("Número de muestras [10]: ").strip()
        num = int(num) if num else 10
        test_from_test_set(model, num)
    elif opcion == "2":
        image_path = input("Ruta de la imagen: ").strip().strip('"')
        test_single_image(model, image_path)
    else:
        print("Saliendo...")

if __name__ == "__main__":
    main()