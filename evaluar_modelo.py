"""
Evaluación de modelo de reconocimiento de cerdas.
Genera matriz de confusión, precisión por clase, y análisis de errores.

Uso:
    python evaluar_modelo.py [--model MODELO.keras] [--test-set test_set.csv]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

utils.load_env()
BASE_DIR = utils.BASE_DIR
OUTPUT_DIR = BASE_DIR / "output_v2"
IMG_SIZE = utils.IMG_SIZE


def load_test_set(test_csv: Path):
    df = pd.read_csv(test_csv)
    logger.info(f"Test set: {len(df)} imágenes, {df['class'].nunique()} clases")
    return df


def load_model(model_path: Path):
    import tensorflow as tf
    logger.info(f"Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    logger.info(f"  Input: {model.input_shape}, Output: {model.output_shape}")
    return model


def preprocess(img_path: str):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_tta(img_path: str):
    img = Image.open(img_path).convert("RGB")
    versions = [
        (img.copy(), "original"),
        (img.transpose(Image.FLIP_LEFT_RIGHT), "flip"),
        (img.rotate(10, resample=Image.Resampling.BILINEAR, expand=False, fillcolor="black"), "rot+10"),
        (img.rotate(-10, resample=Image.Resampling.BILINEAR, expand=False, fillcolor="black"), "rot-10"),
    ]
    return [(np.expand_dims(np.array(v.resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0, axis=0), lbl) for v, lbl in versions]


def evaluate(model, test_df, class_names, threshold=0.5, use_tta=False):
    y_true = []
    y_pred = []
    y_conf = []
    errors = []

    for idx, row in test_df.iterrows():
        true_class = row["class"]
        true_idx = class_names.index(true_class) if true_class in class_names else -1
        if true_idx == -1:
            logger.warning(f"  Clase desconocida: {true_class}, saltando")
            continue

        img_path = row["filename"]
        if not os.path.exists(img_path):
            logger.warning(f"  No existe: {img_path}")
            continue

        if use_tta:
            tta_versions = preprocess_tta(img_path)
            all_preds = []
            for aug_array, _lbl in tta_versions:
                all_preds.append(model.predict(aug_array, verbose=0)[0])
            avg_preds = np.mean(all_preds, axis=0)
            preds = avg_preds
        else:
            x = preprocess(img_path)
            preds = model.predict(x, verbose=0)[0]

        pred_idx = int(np.argmax(preds))
        conf = float(preds[pred_idx])

        if conf < threshold:
            pred_label = "no_cerdo"
            pred_idx = class_names.index("no_cerdo") if "no_cerdo" in class_names else -1
        else:
            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else "unknown"

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_conf.append(conf)

        if pred_label != true_class:
            errors.append({
                "file": os.path.basename(img_path),
                "true": true_class,
                "pred": pred_label,
                "confidence": conf,
            })

    return np.array(y_true), np.array(y_pred), np.array(y_conf), errors


def compute_metrics(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    logger.info("\n" + "=" * 70)
    logger.info("REPORTE DE CLASIFICACIÓN")
    logger.info("=" * 70)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=range(len(class_names)),
        digits=4,
        zero_division=0,
    )
    print(report)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=range(len(class_names)),
        output_dict=True,
        zero_division=0,
    )

    total_acc = report_dict.get("accuracy", 0)
    logger.info(f"\nAccuracy global: {total_acc:.4f} ({total_acc*100:.2f}%)")

    return cm, report_dict


def print_confusion_matrix(cm, class_names):
    logger.info("\n" + "=" * 70)
    logger.info("MATRIZ DE CONFUSIÓN")
    logger.info("=" * 70)

    max_name_len = max(len(n) for n in class_names)
    header = " " * (max_name_len + 2)
    for name in class_names:
        header += f"{name:>8}"
    logger.info(header)

    for i, name in enumerate(class_names):
        row = f"{name:>{max_name_len}}  "
        for j in range(len(class_names)):
            row += f"{cm[i][j]:>8}"
        logger.info(row)


def print_error_analysis(errors, class_names):
    logger.info("\n" + "=" * 70)
    logger.info("ANÁLISIS DE ERRORES")
    logger.info("=" * 70)

    if not errors:
        logger.info("  ¡Sin errores!")
        return

    # Top confused pairs
    confusion_pairs = Counter()
    for err in errors:
        pair = (err["true"], err["pred"])
        confusion_pairs[pair] += 1

    logger.info("\nPares más confundidos (true → pred):")
    for (true_cls, pred_cls), count in confusion_pairs.most_common(15):
        logger.info(f"  {true_cls} → {pred_cls}: {count} errores")

    # Per-class error rate
    class_errors = Counter()
    class_total = Counter()
    for err in errors:
        class_errors[err["true"]] += 1
        class_total[err["true"]] += 1

    for err in errors:
        class_total[err["true"]] += 0

    logger.info("\nError rate por clase:")
    for cls in sorted(class_total.keys()):
        total = class_total[cls]
        errs = class_errors.get(cls, 0)
        rate = errs / total if total else 0
        bar = "█" * int(rate * 40)
        logger.info(f"  {cls:>15}: {errs:>3}/{total:<3} ({rate:.1%}) {bar}")


def save_results(cm, report_dict, errors, output_dir):
    results = {
        "accuracy": report_dict.get("accuracy", 0),
        "per_class": {},
        "total_errors": len(errors),
        "confusion_matrix": cm.tolist(),
    }

    for cls_name, metrics in report_dict.items():
        if isinstance(metrics, dict):
            results["per_class"][cls_name] = {
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1-score": metrics.get("f1-score", 0),
                "support": metrics.get("support", 0),
            }

    results["top_errors"] = [
        {"true": e["true"], "pred": e["pred"], "confidence": e["confidence"]}
        for e in sorted(errors, key=lambda x: -x["confidence"])[:30]
    ]

    out_path = output_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResultados guardados: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo de reconocimiento")
    parser.add_argument("--model", default=str(OUTPUT_DIR / "best_model_phase1.keras"),
                        help="Ruta al modelo .keras o .h5")
    parser.add_argument("--test-set", default=str(OUTPUT_DIR / "test_set.csv"),
                        help="Ruta al CSV de test")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Umbral de confianza (default: desde classes.json o 0.20)")
    parser.add_argument("--tta", action="store_true", default=False,
                        help="Usar Test-Time Augmentation (4 predicciones promediadas)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Guardar resultados JSON")
    args = parser.parse_args()

    model_path = Path(args.model)
    test_csv = Path(args.test_set)

    if not model_path.exists():
        logger.error(f"Modelo no encontrado: {model_path}")
        sys.exit(1)
    if not test_csv.exists():
        logger.error(f"Test set no encontrado: {test_csv}")
        sys.exit(1)

    model = load_model(model_path)
    test_df = load_test_set(test_csv)

    classes_path = OUTPUT_DIR / "classes.json"
    if classes_path.exists():
        with open(classes_path) as f:
            class_data = json.load(f)
        class_names = class_data.get("classes", [])
    else:
        class_names = utils.load_classes()

    # Load threshold from classes.json if not explicitly provided via CLI
    if args.threshold is None:
        args.threshold = utils.load_threshold(classes_path)

    logger.info(f"Clases: {class_names} ({len(class_names)} total)")
    logger.info(f"Threshold: {args.threshold} (from classes.json)" if classes_path.exists() else f"Threshold: {args.threshold} (default)")
    logger.info(f"TTA: {'activado' if args.tta else 'desactivado'}")

    y_true, y_pred, y_conf, errors = evaluate(
        model, test_df, class_names, threshold=args.threshold, use_tta=args.tta
    )

    cm, report_dict = compute_metrics(y_true, y_pred, class_names)
    print_confusion_matrix(cm, class_names)
    print_error_analysis(errors, class_names)

    # Per-class accuracy
    logger.info("\n" + "=" * 70)
    logger.info("ACCURACY POR CLASE")
    logger.info("=" * 70)
    for i, cls in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            bar = "█" * int(acc * 30)
            logger.info(f"  {cls:>15}: {acc:.4f} ({acc*100:.2f}%) {bar}")

    if args.save:
        save_results(cm, report_dict, errors, OUTPUT_DIR)


if __name__ == "__main__":
    main()
