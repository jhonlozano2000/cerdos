"""
Script de inferencia rápido para probar el modelo
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
    print(f"📂 Cargando modelo: {MODEL_PATH}")
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

def predict(model, image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]
    
    top_idx = np.argmax(predictions)
    confidence = float(predictions[top_idx])
    
    return {
        "class_id": int(top_idx),
        "class_name": CLASS_MAPPING[top_idx],
        "confidence": confidence,
    }

def test_from_csv():
    import pandas as pd
    
    df = pd.read_csv("output/test_split.csv")
    
    model = load_model()
    
    print(f"\n🧪 Probando con {len(df)} imágenes del test set...")
    print("=" * 60)
    
    correct = 0
    errors = []
    
    for i, row in df.iterrows():
        img_path = row['path']
        true_class = row['class_name']
        
        if not os.path.exists(img_path):
            continue
        
        try:
            result = predict(model, img_path)
            predicted = result['class_name']
            confidence = result['confidence']
            
            is_correct = predicted == true_class
            if is_correct:
                correct += 1
            else:
                errors.append({
                    'image': os.path.basename(img_path),
                    'true': true_class,
                    'predicted': predicted,
                    'confidence': confidence
                })
                
            print(f"{'✅' if is_correct else '❌'} {os.path.basename(img_path)}: {true_class} → {predicted} ({confidence:.1%})")
            
        except Exception as e:
            print(f"❌ Error con {img_path}: {e}")
    
    accuracy = correct / len(df) * 100
    print("=" * 60)
    print(f"📊 Accuracy: {correct}/{len(df)} ({accuracy:.2f}%)")
    
    if errors:
        print(f"\n❌ Errores ({len(errors)}):")
        for e in errors[:5]:
            print(f"   {e['image']}: {e['true']} → {e['predicted']} ({e['confidence']:.1%})")

def test_single():
    import tensorflow as tf
    
    model = load_model()
    
    print("\n📷 Ingresa la ruta de la imagen:")
    print("   Ejemplo: C:\\Users\\Jhon\\Desktop\\cerdos\\dataset_procesado\\cerda_001\\foto.jpg")
    print("   (puedes arrastrar la imagen aquí)")
    
    image_path = input("\nRuta: ").strip().strip('"')
    
    if not os.path.exists(image_path):
        print(f"❌ No encontré: {image_path}")
        return
    
    print(f"\n🖼️  Procesando: {os.path.basename(image_path)}")
    
    result = predict(model, image_path)
    
    print(f"\n🎯 Resultado: {result['class_name']}")
    print(f"   Confianza: {result['confidence']:.1%}")

def main():
    print("=" * 60)
    print("🎯 PRUEBAS DEL MODELO - Identificación de Cerdas")
    print("=" * 60)
    print(f"\n📂 Clases ({NUM_CLASSES}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"   {i}: {name}")
    
    print("\n" + "=" * 60)
    print("MENU:")
    print("1. Probar con test set (CSV)")
    print("2. Probar una imagen específica")
    print("0. Salir")
    print("=" * 60)
    
    opcion = input("\nOpción: ").strip()
    
    if opcion == "1":
        test_from_csv()
    elif opcion == "2":
        test_single()
    else:
        print("👋")

if __name__ == "__main__":
    main()