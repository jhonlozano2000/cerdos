"""
Entrenamiento rápido YOLO para cerdas
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

DATASET = Path("dataset_procesado")
OUTPUT = Path("output/yolo_detector")
OUTPUT.mkdir(exist_ok=True)

def prepare_dataset():
    """Prepara estructura YOLO"""
    
    print("📦 Preparando dataset YOLO...")
    
    (OUTPUT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    df_train = pd.read_csv("output/train_split.csv")
    df_val = pd.read_csv("output/val_split.csv")
    
    count_train = 0
    count_val = 0
    
    for _, row in df_train.iterrows():
        src = row['path']
        if not os.path.exists(src):
            continue
        
        dst = OUTPUT / "images" / "train" / os.path.basename(src)
        shutil.copy2(src, dst)
        
        label_name = dst.stem + ".txt"
        label_file = OUTPUT / "labels" / "train" / label_name
        label_file.write_text("0 0.5 0.5 1 1\n")
        count_train += 1
    
    for _, row in df_val.iterrows():
        src = row['path']
        if not os.path.exists(src):
            continue
        
        dst = OUTPUT / "images" / "val" / os.path.basename(src)
        shutil.copy2(src, dst)
        
        label_name = dst.stem + ".txt"
        label_file = OUTPUT / "labels" / "val" / label_name
        label_file.write_text("0 0.5 0.5 1 1\n")
        count_val += 1
    
    print(f"✅ Train: {count_train}")
    print(f"✅ Val: {count_val}")
    
    yaml = OUTPUT / "data.yaml"
    yaml.write_text(f"""path: {OUTPUT}
train: images/train
val: images/val

names:
  0: cerda
""")
    print(f"✅ {yaml}")
    
    return OUTPUT / "data.yaml"

def train():
    print("=" * 50)
    print("🎯 YOLO Detector de Cerdas")
    print("=" * 50)
    
    data_yaml = prepare_dataset()
    
    model = YOLO("yolov8n.pt")
    
    print("\n🚀 Entrenando (10 epochs)...")
    
    results = model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=224,
        batch=16,
        project=str(OUTPUT),
        name="runs",
        exist_ok=True,
        verbose=False,
        device='cpu'
    )
    
    print("\n✅ Entrenamiento completado!")
    print(f"📁 Modelo: {OUTPUT}/runs/weights/best.pt")

if __name__ == "__main__":
    train()