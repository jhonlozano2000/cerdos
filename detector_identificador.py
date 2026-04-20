"""
Detector + Identificador de Cerdas
Usa YOLO para detectar cerdos, luego MobileNet para identificar
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "output", "modelo_final.keras")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_procesado")

CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_PATH) 
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])
CLASS_MAPPING = {i: name for i, name in enumerate(CLASS_NAMES)}

print("=" * 60)
print("🎯 Detector + Identificador de Cerdas")
print("=" * 60)

def load_models():
    print("\n📂 Cargando modelos...")
    
    print("   1. YOLO (detección de cerdos)...")
    yolo = YOLO("yolov8n.pt")
    
    print("   2. MobileNet (identificación)...")
    import tensorflow as tf
    identificador = tf.keras.models.load_model(MODEL_PATH)
    
    print("✅ Modelos cargados")
    return yolo, identificador

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def identify_pig(model, image_array):
    predictions = model.predict(image_array, verbose=0)[0]
    top_idx = np.argmax(predictions)
    confidence = float(predictions[top_idx])
    
    return {
        "class_id": int(top_idx),
        "class_name": CLASS_MAPPING[top_idx],
        "confidence": confidence
    }

def detect_and_identify(yolo, identificador, image_path, conf_threshold=0.3):
    """
    Detecta cerdos con YOLO y luego identifica con MobileNet
    """
    results = yolo.predict(
        image_path, 
        classes=[19],  # COCO pig class
        conf=conf_threshold,
        verbose=False
    )
    
    detections = results[0]
    
    if detections.boxes is None or len(detections.boxes) == 0:
        return {
            "detected": False,
            "message": "No se detectó ningún cerdo en la imagen",
            "cerda": None,
            "confidence": 0
        }
    
    detections_list = []
    for box in detections.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        
        x1, y1, x2, y2 = xyxy
        crop = Image.open(image_path).convert('RGB').crop((x1, y1, x2, y2))
        crop = crop.resize((224, 224))
        crop_array = np.array(crop, dtype=np.float32) / 255.0
        crop_array = np.expand_dims(crop_array, axis=0)
        
        result = identify_pig(identificador, crop_array)
        
        detections_list.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": conf,
            "cerda": result["class_name"],
            "cerda_confidence": result["confidence"]
        })
    
    best = max(detections_list, key=lambda x: x["cerda_confidence"])
    
    return {
        "detected": True,
        "message": f"Detectado: {best['cerda']}",
        "cerda": best["cerda"],
        "confidence": best["cerda_confidence"],
        "detections": detections_list
    }

def test_images():
    yolo, identificador = load_models()
    
    print("\n" + "=" * 60)
    print("PRUEBAS")
    print("=" * 60)
    
    test_images = [
        "C:\\Users\\Jhon\\Downloads\\perro.jpg",
        "C:\\Users\\Jhon\\Downloads\\cerdo.jpg",
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"\n❌ No existe: {img_path}")
            continue
        
        print(f"\n🖼️  {os.path.basename(img_path)}")
        print("-" * 40)
        
        result = detect_and_identify(yolo, identificador, img_path)
        
        if result["detected"]:
            print(f"   🐷 detectatdo")
            print(f"   🐷 Cerda: {result['cerda']}")
            print(f"   📊 Confianza: {result['confidence']:.1%}")
        else:
            print(f"   ❌ {result['message']}")
    
    print("\n" + "=" * 60)
    print("TEST CON IMÁGENES DEL DATASET")
    print("=" * 60)
    
    import pandas as pd
    df = pd.read_csv("output/test_split.csv")
    
    correct = 0
    total = 0
    no_detected = 0
    
    samples = df.sample(30, random_state=42)
    
    for _, row in samples.iterrows():
        img_path = row['path']
        true_class = row['class_name']
        
        if not os.path.exists(img_path):
            continue
        
        result = detect_and_identify(yolo, identificador, img_path)
        total += 1
        
        if not result["detected"]:
            no_detected += 1
            print(f"⚠️  {os.path.basename(img_path)}: NO detectado")
            continue
        
        predicted = result["cerda"]
        is_correct = predicted == true_class
        
        if is_correct:
            correct += 1
            print(f"✅ {os.path.basename(img_path)}: {true_class} → {predicted}")
        else:
            print(f"❌ {os.path.basename(img_path)}: {true_class} → {predicted}")
    
    print(f"\n📊 Resultados:")
    print(f"   Detectados: {total - no_detected}/{total}")
    print(f"   Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

def test_single():
    yolo, identificador = load_models()
    
    print("\n📷 Ingresa la ruta de la imagen:")
    image_path = input("Ruta: ").strip().strip('"')
    
    if not os.path.exists(image_path):
        print(f"❌ No encontré: {image_path}")
        return
    
    print(f"\n🖼️  Procesando...")
    
    result = detect_and_identify(yolo, identificador, image_path)
    
    if result["detected"]:
        print(f"\n✅ ¡Cerdo detectado!")
        print(f"   Cerda: {result['cerda']}")
        print(f"   Confianza: {result['confidence']:.1%}")
    else:
        print(f"\n❌ {result['message']}")

def main():
    print("\nMENU:")
    print("1. Probar con varias imágenes")
    print("2. Probar una imagen específica")
    print("3. Test con dataset (30 muestras)")
    print("0. Salir")
    
    opcion = input("\nOpción: ").strip()
    
    if opcion == "1":
        test_images()
    elif opcion == "2":
        test_single()
    elif opcion == "3":
        test_images()
    else:
        print("👋")

if __name__ == "__main__":
    main()