"""
Test de inferencia con YOLO entrenado
"""
import os
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "runs", "detect", "output", "yolo_detector", "runs", "weights", "best.pt")

def load_model():
    print(f"📂 Cargando modelo: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

def detect(image_path, model, conf=0.25):
    img = Image.open(image_path)
    results = model(img, conf=conf, verbose=False)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detections.append({
                'class': model.model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
    
    return detections

if __name__ == "__main__":
    import sys
    
    model = load_model()
    print(f"✅ Modelo listo. Clases: {model.model.names}")
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "test_image.jpg"
    
    if os.path.exists(img_path):
        print(f"\n🔍 Detectando en: {img_path}")
        dets = detect(img_path, model)
        print(f"📊 Encontrados: {len(dets)}")
        for d in dets:
            print(f"   - {d['class']}: {d['confidence']:.1%}")
    else:
        print(f"❌ Imagen no encontrada: {img_path}")
        print("💡 Uso: python inference_yolo.py <imagen.jpg>")