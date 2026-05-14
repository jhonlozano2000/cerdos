"""
Descarga imagenes negativas desde GitHub (ImageNet samples).
GitHub pasa el proxy/SSL sin problemas.

Uso: python descargar_negativos.py
"""

import os
import sys
import io
import numpy as np
from pathlib import Path

from PIL import Image, ImageFilter
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_DIR = Path(__file__).resolve().parent / "dataset_procesado" / "no_cerdo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = (224, 224)


def download_image(url, save_path, timeout=15):
    try:
        resp = requests.get(url, timeout=timeout, verify=False)
        if resp.status_code == 200 and len(resp.content) > 1000:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            img.save(save_path, quality=90)
            return True
    except Exception:
        pass
    return False


def download_from_github():
    """Descarga imagenes de ImageNet samples desde GitHub."""
    print("\n[1/2] Descargando imagenes de animales (GitHub)...")

    # Repo: https://github.com/EliSchwartz/imagenet-sample-images
    base = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master"

    images = [
        # Perros (15)
        ("dog_01", f"{base}/n02099601_golden_retriever.JPEG"),
        ("dog_02", f"{base}/n02106662_German_shepherd.JPEG"),
        ("dog_03", f"{base}/n02110185_Siberian_husky.JPEG"),
        ("dog_04", f"{base}/n02110958_pug.JPEG"),
        ("dog_05", f"{base}/n02111889_Samoyed.JPEG"),
        ("dog_06", f"{base}/n02113023_pembroke.JPEG"),
        ("dog_07", f"{base}/n02113624_toy_poodle.JPEG"),
        ("dog_08", f"{base}/n02113799_standard_poodle.JPEG"),
        ("dog_09", f"{base}/n02102040_English_springer.JPEG"),
        ("dog_10", f"{base}/n02100583_vizsla.JPEG"),
        ("dog_11", f"{base}/n02100735_English_setter.JPEG"),
        ("dog_12", f"{base}/n02101006_Gordon_setter.JPEG"),
        ("dog_13", f"{base}/n02102177_Welsh_springer_spaniel.JPEG"),
        ("dog_14", f"{base}/n02105641_Old_English_sheepdog.JPEG"),
        ("dog_15", f"{base}/n02109961_Eskimo_dog.JPEG"),
        # Gatos (10)
        ("cat_01", f"{base}/n02123045_tabby.JPEG"),
        ("cat_02", f"{base}/n02123159_tiger_cat.JPEG"),
        ("cat_03", f"{base}/n02123394_Persian_cat.JPEG"),
        ("cat_04", f"{base}/n02123597_Siamese_cat.JPEG"),
        ("cat_05", f"{base}/n02124075_Egyptian_cat.JPEG"),
        ("cat_06", f"{base}/n02127052_lynx.JPEG"),
        ("cat_07", f"{base}/n02128385_leopard.JPEG"),
        ("cat_08", f"{base}/n02128757_snow_leopard.JPEG"),
        ("cat_09", f"{base}/n02128925_jaguar.JPEG"),
        ("cat_10", f"{base}/n02129165_lion.JPEG"),
        # Caballos/burros (5)
        ("horse_01", f"{base}/n02389026_sorrel.JPEG"),
        ("horse_02", f"{base}/n02391049_zebra.JPEG"),
        ("horse_03", f"{base}/n02397096_warthog.JPEG"),
        ("horse_04", f"{base}/n02408429_water_buffalo.JPEG"),
        ("horse_05", f"{base}/n02412080_ram.JPEG"),
        # Vacas/ovejas/cabras (5)
        ("cow_01", f"{base}/n02403003_ox.JPEG"),
        ("cow_02", f"{base}/n02410509_bison.JPEG"),
        ("cow_03", f"{base}/n02415577_bighorn.JPEG"),
        ("cow_04", f"{base}/n02417914_ibex.JPEG"),
        ("cow_05", f"{base}/n02422106_hartebeest.JPEG"),
        # Otros animales (10)
        ("other_01", f"{base}/n02129604_tiger.JPEG"),
        ("other_02", f"{base}/n02130308_cheetah.JPEG"),
        ("other_03", f"{base}/n02132136_brown_bear.JPEG"),
        ("other_04", f"{base}/n02133161_American_black_bear.JPEG"),
        ("other_05", f"{base}/n02134084_ice_bear.JPEG"),
        ("other_06", f"{base}/n02134418_sloth_bear.JPEG"),
        ("other_07", f"{base}/n01882714_koala.JPEG"),
        ("other_08", f"{base}/n02504458_African_elephant.JPEG"),
        ("other_09", f"{base}/n02509815_lesser_panda.JPEG"),
        ("other_10", f"{base}/n02510455_giant_panda.JPEG"),
        # Objetos/vehiculos/paisajes (10)
        ("obj_01", f"{base}/n02814533_beach_wagon.JPEG"),
        ("obj_02", f"{base}/n02930766_cab.JPEG"),
        ("obj_03", f"{base}/n03100240_convertible.JPEG"),
        ("obj_04", f"{base}/n03417042_garbage_truck.JPEG"),
        ("obj_05", f"{base}/n03445777_golf_ball.JPEG"),
        ("obj_06", f"{base}/n03452741_grand_piano.JPEG"),
        ("obj_07", f"{base}/n03584829_iron.JPEG"),
        ("obj_08", f"{base}/n03594945_jeep.JPEG"),
        ("obj_09", f"{base}/n03670208_limousine.JPEG"),
        ("obj_10", f"{base}/n03770679_minivan.JPEG"),
        # Aves (5)
        ("bird_01", f"{base}/n01530575_brambling.JPEG"),
        ("bird_02", f"{base}/n01531178_goldfinch.JPEG"),
        ("bird_03", f"{base}/n01532829_house_finch.JPEG"),
        ("bird_04", f"{base}/n01534433_junco.JPEG"),
        ("bird_05", f"{base}/n01537544_indigo_bunting.JPEG"),
        # Insectos/otros (5)
        ("insect_01", f"{base}/n02206856_bee.JPEG"),
        ("insect_02", f"{base}/n02219486_ant.JPEG"),
        ("insect_03", f"{base}/n02226429_grasshopper.JPEG"),
        ("insect_04", f"{base}/n02229544_cricket.JPEG"),
        ("insect_05", f"{base}/n02231487_walking_stick.JPEG"),
    ]

    count = 0
    total = len(images)
    for name, url in images:
        save_path = OUTPUT_DIR / f"{name}.jpg"
        if save_path.exists():
            count += 1
            continue
        if download_image(url, save_path):
            count += 1
            print(f"\r   Descargando: {count}/{total}", end="", flush=True)
        else:
            print(f"\n   FAIL: {name}")

    print(f"\n   Descargadas: {count}/{total}")
    return count


def generate_synthetic():
    """Genera imagenes sinteticas de fondos tipo granja."""
    print("\n[2/2] Generando imagenes sinteticas...")
    count = 0
    np.random.seed(42)

    colors = [
        (139, 90, 43, "tierra"),
        (34, 139, 34, "pasto"),
        (128, 128, 128, "concreto"),
        (210, 180, 140, "arena"),
        (85, 107, 47, "verde_oscuro"),
        (160, 82, 45, "madera"),
        (105, 105, 105, "metal"),
        (222, 184, 135, "paja"),
        (169, 169, 169, "gris_claro"),
        (101, 67, 33, "barro"),
    ]

    for r, g, b, nombre in colors:
        for j in range(5):
            img = np.full((224, 224, 3), [r, g, b], dtype=np.uint8)
            noise = np.random.randint(-35, 35, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=2))
            img_pil.save(OUTPUT_DIR / f"bg_{nombre}_{j}.jpg", quality=90)
            count += 1

    for i in range(20):
        base_color = np.random.randint(40, 200, 3).tolist()
        img = np.full((224, 224, 3), base_color, dtype=np.uint8)
        noise = np.random.randint(-50, 50, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        blur_radius = int(np.random.choice([3, 5, 8, 12]))
        img_pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img_pil.save(OUTPUT_DIR / f"texture_{i:03d}.jpg", quality=90)
        count += 1

    print(f"   Generadas: {count} imagenes")
    return count


def main():
    print("=" * 60)
    print("DESCARGA DE IMAGENES NEGATIVAS (no_cerdo)")
    print("=" * 60)

    count_total = 0
    count_total += download_from_github()
    count_total += generate_synthetic()

    total_files = len(list(OUTPUT_DIR.glob("*.jpg")))
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_files} imagenes en no_cerdo/")
    print("=" * 60)
    print(f"\nCarpeta: {OUTPUT_DIR}")
    print("\nSiguiente paso: python entrenar_v2.py")


if __name__ == "__main__":
    main()
