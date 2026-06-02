"""
Analiza el dataset y genera reporte de clases que necesitan más fotos.

Uso:
    python analizar_dataset.py
"""

import os
import json
from pathlib import Path
from collections import Counter

DATASET_PATH = Path(r"C:\laragon\www\Porci-Integral-backend\storage\app\public\datasets\animales")
TARGET_MIN = 200  # imágenes ideales por clase
CRITICAL = 50     # mínimo crítico

print("=" * 60)
print("REPORTE DEL DATASET")
print("=" * 60)

classes = sorted([d for d in os.listdir(DATASET_PATH) if (DATASET_PATH / d).is_dir()])
total_global = 0
needs_data = []

for cls in classes:
    cls_dir = DATASET_PATH / cls
    images = [f for f in cls_dir.iterdir()
              if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
    count = len(images)
    total_global += count

    status = "✅" if count >= TARGET_MIN else ("⚠️" if count >= CRITICAL else "🔴")
    pct = (count / TARGET_MIN) * 100
    bar = "█" * min(int(pct / 5), 20)

    if count < TARGET_MIN:
        needed = TARGET_MIN - count
        needs_data.append((cls, count, needed))

    print(f"  {status} {cls:>15}: {count:>4} imágenes {bar}")

print(f"\n{'─' * 60}")
print(f"  Total: {total_global} imágenes en {len(classes)} clases")
print(f"{'─' * 60}")

if needs_data:
    print(f"\n📸 CLASES QUE NECESITAN MÁS FOTOS (target: {TARGET_MIN}):")
    for cls, count, needed in sorted(needs_data, key=lambda x: x[1]):
        urgency = "🔴 URGENTE" if count < CRITICAL else "⚠️"
        print(f"  {urgency}: {cls} ({count} actuales, faltan {needed})")

    print(f"\n📋 RESUMEN PARA EL VETERINARIO:")
    print(f"  Total de fotos necesarias: {sum(n for _, _, n in needs_data)}")
    print(f"  Clases críticas (<{CRITICAL}): {[c for c, cnt, _ in needs_data if cnt < CRITICAL]}")
    print(f"  Clases con pocas (<{TARGET_MIN}): {[c for c, cnt, _ in needs_data]}")
else:
    print("\n✅ Todas las clases tienen suficientes imágenes.")
