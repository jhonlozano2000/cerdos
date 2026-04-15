# -*- coding: utf-8 -*-
"""
Utilidad de Limpieza y Estandarización del Dataset
Características:
- Renombrar carpetas con nomenclatura consistente
- Detectar y eliminar imágenes corruptas o muy pequeñas
- Generar reporte de calidad del dataset
"""

import os
import sys
import shutil
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

# Fix para Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

DATASET_DIR = Path("dataset_procesado")
REPORT_FILE = "dataset_reporte.json"
MIN_IMAGE_SIZE = (100, 100)
MIN_IMAGES_PER_CLASS = 5

class DatasetCleaner:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.report = {
            "fecha": datetime.now().isoformat(),
            "directorio": str(self.dataset_dir),
            " clases": {},
            "acciones": [],
            "problemas": []
        }
    
    def analyze(self):
        """Analiza el estado actual del dataset"""
        print("=" * 50)
        print("📊 ANALISIS DEL DATASET")
        print("=" * 50)
        
        classes = sorted([d for d in os.listdir(self.dataset_dir) 
                        if (self.dataset_dir / d).is_dir()])
        
        total_images = 0
        class_stats = {}
        
        for class_name in classes:
            class_dir = self.dataset_dir / class_name
            images = list(class_dir.glob("*"))
            
            valid_images = 0
            small_images = 0
            corrupt_images = 0
            
            for img_path in images:
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue
                    
                try:
                    with Image.open(img_path) as img:
                        if img.size[0] < MIN_IMAGE_SIZE[0] or img.size[1] < MIN_IMAGE_SIZE[1]:
                            small_images += 1
                        else:
                            valid_images += 1
                except Exception as e:
                    corrupt_images += 1
                    self.report["problemas"].append(f"Corrupta: {img_path}")
            
            count = valid_images
            total_images += count
            
            class_stats[class_name] = {
                "total": len(images),
                "validas": valid_images,
                "pequeñas": small_images,
                "corruptas": corrupt_images
            }
            
            status = "✅" if count >= MIN_IMAGES_PER_CLASS else "⚠️"
            print(f"{status} {class_name}: {count} imágenes válidas")
        
        self.report["clases"] = class_stats
        print(f"\n📈 Total: {total_images} imágenes en {len(classes)} clases")
        
        return class_stats
    
    def standardize_names(self, prefix="cerda", start=1):
        """Estandariza nombres de carpetas"""
        print("\n" + "=" * 50)
        print("🔄 Estandarizando nombres...")
        print("=" * 50)
        
        classes = sorted([d for d in os.listdir(self.dataset_dir) 
                        if (self.dataset_dir / d).is_dir()])
        
        new_names = {}
        idx = start
        
        for old_name in classes:
            new_name = f"{prefix}_{idx:03d}"
            old_path = self.dataset_dir / old_name
            new_path = self.dataset_dir / new_name
            
            if old_name != new_name:
                if new_path.exists():
                    self.report["problemas"].append(f"Conflicto: {new_name} ya existe")
                    continue
                    
                old_path.rename(new_path)
                new_names[old_name] = new_name
                self.report["acciones"].append(f"Renombrado: {old_name} -> {new_name}")
                print(f"   {old_name} -> {new_name}")
            
            idx += 1
        
        return new_names
    
    def remove_invalid_images(self, dry_run=True):
        """Elimina imágenes pequeñas o corruptas"""
        print("\n" + "=" * 50)
        print(f"{'🧹 LIMPIEZA (dry-run)' if dry_run else '🧹 LIMPIEZA'}")
        print("=" * 50)
        
        if dry_run:
            print("⚠️  Modo dry-run - no se eliminará nada")
        
        removed_count = 0
        
        for class_name in os.listdir(self.dataset_dir):
            class_dir = self.dataset_dir / class_name
            
            if not class_dir.is_dir():
                continue
            
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue
                
                try:
                    with Image.open(img_path) as img:
                        if img.size[0] < MIN_IMAGE_SIZE[0] or img.size[1] < MIN_IMAGE_SIZE[1]:
                            if not dry_run:
                                img_path.unlink()
                            removed_count += 1
                            self.report["acciones"].append(f"Eliminada (pequeña): {img_path}")
                            print(f"   ❌ {img_path.name} ({img.size})")
                except Exception as e:
                    if not dry_run:
                        img_path.unlink()
                    removed_count += 1
                    self.report["problemas"].append(f"Eliminada (corrupta): {img_path}")
                    print(f"   ❌ {img_path.name} (corrupta)")
        
        print(f"\n{'Encontradas' if dry_run else 'Eliminadas'}: {removed_count} imágenes problemáticas")
        return removed_count
    
    def create_test_split(self, test_ratio=0.2, copy=True):
        """Separa un conjunto de test real"""
        print("\n" + "=" * 50)
        print("📂 Creando test split...")
        print("=" * 50)
        
        test_dir = self.dataset_dir.parent / "dataset_test"
        test_dir.mkdir(exist_ok=True)
        
        classes = sorted([d for d in os.listdir(self.dataset_dir) 
                        if (self.dataset_dir / d).is_dir()])
        
        import math
        import random
        random.seed(42)
        
        for class_name in classes:
            class_dir = self.dataset_dir / class_name
            images = list(class_dir.glob("*"))
            images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
            
            if len(images) < 3:
                continue
            
            num_test = max(1, math.ceil(len(images) * test_ratio))
            test_images = random.sample(images, num_test)
            
            dest_dir = test_dir / class_name
            if copy:
                dest_dir.mkdir(exist_ok=True)
            
            for img in test_images:
                if copy:
                    shutil.copy2(img, dest_dir / img.name)
                else:
                    img.unlink()
            
            print(f"   {class_name}: {num_test} imágenes para test")
        
        print(f"\n📁 Test set guardado en: {test_dir}")
    
    def generate_report(self):
        """Genera reporte JSON"""
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Reporte guardado en: {REPORT_FILE}")
        return self.report

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Limpieza del dataset")
    parser.add_argument("--dry-run", action="store_true", help="Modo dry-run (no modifica nada)")
    parser.add_argument("--rename", action="store_true", help="Renombrar carpetas")
    parser.add_argument("--clean", action="store_true", help="Eliminar imágenes inválidas")
    parser.add_argument("--test-split", action="store_true", help="Crear test split")
    parser.add_argument("--prefix", default="cerda", help="Prefijo para renombrar")
    args = parser.parse_args()
    
    cleaner = DatasetCleaner(DATASET_DIR)
    
    cleaner.analyze()
    
    if args.rename:
        cleaner.standardize_names(prefix=args.prefix)
    
    if args.clean:
        cleaner.remove_invalid_images(dry_run=args.dry_run)
    
    if args.test_split:
        cleaner.create_test_split()
    
    cleaner.generate_report()
    
    print("\n" + "=" * 50)
    print("✅ LIMPIEZA COMPLETADA")
    print("=" * 50)

if __name__ == "__main__":
    main()
