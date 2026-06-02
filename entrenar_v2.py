"""
Entrenamiento Mejorado — Reconocimiento de Cerdas v2.1
======================================================
Pipeline de entrenamiento en 2 fases para MobileNetV2
con Focal Loss, diseñado para dataset desbalanceado (14 clases,
desde 14 imágenes hasta 7452 por clase).

Arquitectura:
  MobileNetV2 (ImageNet) → GlobalAvgPool → BatchNorm → Dense(256, ReLU)
  → Dropout(0.5) → Dense(128, ReLU) → Dropout(0.4) → Dense(N, Softmax)

Fases de entrenamiento:
   1. Frozen (60 épocas máx): base MobileNetV2 congelada, solo entrena la cabeza.
      Learning rate 1e-4, ReduceLROnPlateau, EarlyStopping paciencia 5.
   2. Fine-Tune (80 épocas máx): últimas 8 capas descongeladas, LR 1e-6.
      EarlyStopping paciencia 10, ReduceLROnPlateau.

Mejoras contra desbalance:
   - Focal Loss (γ=1.5, α=0.25) — enfoca en muestras difíciles
   - Class weights balanceados automáticamente (sklearn)
   - MAX_IMAGES_PER_CLASS=200, MAX_NO_CERDO=80 — capping por clase
   - Oversampling sintético para clases con <80 imágenes
   - Data augmentation: rotación 30°, zoom 0.3, flip horizontal+vertical, brillo ±40%
   - Mixup augmentation (α=0.2) — mezcla pares de imágenes virtuales
   - Split estratificado 70/15/15 (train/val/test)

Salidas:
  - output_v2/modelo_identificacion_cerdos_v2.h5 — modelo entrenado
  - best_model_phase1.keras, best_model_v2.keras — checkpoints por fase
  - classes.json — nombres de clase, threshold, metadatos
  - test_set.csv — metadatos del test set para evaluación
  - model_registry.json — historial de versiones (últimas 20)
  - modelo_identificacion_cerdos.h5 (raíz) — copia a producción

Las clases se toman de los nombres de carpeta en DATASET_PATH.
Cada carpeta = un animal, y su nombre es el identificador
en la tabla inventario_animal (ej: "cerda_001", "cerda_012").

Ejecutado por training_manager.py como subproceso.
Uso directo: python entrenar_v2.py [--task-id ID] [--task-dir DIR] [--include-classes LISTA]
"""

import os
import sys
import json
import argparse
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import utils


# ── Configuración ──────────────────────────────────────────────
utils.load_env()

BASE_DIR = utils.BASE_DIR
DATASET_DIR = Path(os.environ.get(
    "DATASET_PATH",
    r"C:\laragon\www\Porci-Integral-backend\storage\app\public\datasets\animales"
))
OUTPUT_DIR = BASE_DIR / "output_v2"
MODEL_NAME = "modelo_identificacion_cerdos_v2.h5"

IMG_SIZE = utils.IMG_SIZE
BATCH_SIZE = 16
EPOCHS_FROZEN = 60
EPOCHS_FINETUNE = 80
LEARNING_RATE = 0.0001
FINETUNE_LR = 0.000001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.15
MAX_IMAGES_PER_CLASS = 200
MAX_NO_CERDO = 80
UNFREEZE_LAST_N = 8  # capas a descongelar en fine-tune (menos = menos overfitting)
OVERSAMPLE_MIN = 80    # clases con menos imágenes reciben oversampling
USE_MIXUP = True       # Mixup augmentation (mezcla pares de imágenes)
MIXUP_ALPHA = 0.2      # parámetro Beta para Mixup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR.mkdir(exist_ok=True)

PROGRESS_FILE = "progress.json"


def report_progress(task_dir, **kwargs):
    """
    Reporta progreso del entrenamiento.
    Escribe en el archivo progress.json del task_dir y también
    imprime una línea PROGRESS:{json} para que training_manager.py
    pueda capturarla desde stdout.
    """
    if task_dir:
        path = Path(task_dir) / PROGRESS_FILE
        try:
            data = {}
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
            data.update(kwargs)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    line = json.dumps(kwargs)
    print(f"PROGRESS:{line}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo de reconocimiento")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--task-dir", default="")
    parser.add_argument("--include-classes", default="",
                        help="Clases a incluir separadas por coma")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-frozen", type=int, default=60)
    parser.add_argument("--epochs-finetune", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--finetune-lr", type=float, default=0.000001)
    parser.add_argument("--max-images-per-class", type=int, default=200)
    parser.add_argument("--max-no-cerdo", type=int, default=80)
    parser.add_argument("--unfreeze-layers", type=int, default=8)
    parser.add_argument("--disable-mixup", action="store_true", help="Deshabilita Mixup augmentation")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--oversample-min", type=int, default=80)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--disable-class-weights", action="store_true", help="Deshabilita balanceo de class weights")
    args = parser.parse_args()

    task_dir = args.task_dir
    include_list = [c.strip() for c in args.include_classes.split(",") if c.strip()] \
                   if args.include_classes else None

    # Override constants from args (module-level → local)
    BATCH_SIZE = args.batch_size
    EPOCHS_FROZEN = args.epochs_frozen
    EPOCHS_FINETUNE = args.epochs_finetune
    LEARNING_RATE = args.lr
    FINETUNE_LR = args.finetune_lr
    MAX_IMAGES_PER_CLASS = args.max_images_per_class
    MAX_NO_CERDO = args.max_no_cerdo
    UNFREEZE_LAST_N = args.unfreeze_layers
    USE_MIXUP = not args.disable_mixup
    MIXUP_ALPHA = args.mixup_alpha
    OVERSAMPLE_MIN = args.oversample_min
    FOCAL_GAMMA = args.focal_gamma
    USE_CLASS_WEIGHTS = not args.disable_class_weights

    # ── Imports diferidos (solo se cargan al ejecutar, no al importar) ──
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import (
        GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
    )
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import train_test_split

    # ── Focal Loss (enfocado en clases difíciles) ─────────────
    # gamma=1.5 (menos agresivo que 2.0 — mejor para fine-tune)
    def focal_loss(gamma=1.5, alpha=0.25):
        def loss(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
            focal_weight = tf.pow(1.0 - p_t, gamma)
            focal_weight = tf.expand_dims(focal_weight, axis=-1)
            return alpha * tf.reduce_sum(focal_weight * cross_entropy, axis=-1)
        return loss

    # ── Progress Callback personalizado ───────────────────────
    # Envía métricas de cada época al training_manager
    class ProgressCallback(Callback):
        def __init__(self, task_dir, total_epochs, phase):
            super().__init__()
            self.task_dir = task_dir
            self.total_epochs = total_epochs
            self.phase = phase
            self.best_val_acc = 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_acc = logs.get("val_accuracy", 0)
            val_loss = logs.get("val_loss", 0)
            acc = logs.get("accuracy", 0)
            loss = logs.get("loss", 0)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

            # Progreso global: 0-50% frozen, 50-100% finetune
            progress_pct = int(((epoch + 1) / self.total_epochs) * 50)
            if self.phase == "finetune":
                progress_pct += 50

            try:
                lr_actual = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            except AttributeError:
                lr_actual = float(tf.keras.backend.get_value(self.model.optimizer.lr))

            report_progress(
                self.task_dir,
                current_epoch=epoch + 1,
                total_epochs=self.total_epochs,
                phase=self.phase,
                current_accuracy=float(acc),
                current_loss=float(loss),
                current_val_accuracy=float(val_acc),
                current_val_loss=float(val_loss),
                best_val_accuracy=float(self.best_val_acc),
                lr_actual=lr_actual,
                progress_pct=progress_pct,
                message=f"Fase {self.phase}: época {epoch + 1}/{self.total_epochs} "
                        f"- acc: {acc:.4f} - val_acc: {val_acc:.4f}",
            )

    # ── Mixup Generator ────────────────────────────────────────
    # Mezcla pares de imágenes con proporción aleatoria λ ~ Beta(α,α)
    # para mejorar generalización (Zhang et al., 2018)
    class MixupGenerator(tf.keras.utils.Sequence):
        def __init__(self, generator, alpha=0.2):
            self.generator = generator
            self.alpha = alpha
            self.batch_size = generator.batch_size

        def __len__(self):
            return len(self.generator)

        def __getitem__(self, index):
            x, y = self.generator[index]
            bs = x.shape[0]
            perm = np.random.permutation(bs)
            x_shuf = x[perm]
            y_shuf = y[perm]
            lam = np.random.beta(self.alpha, self.alpha, size=(bs, 1, 1, 1))
            lam_y = lam.reshape(bs, 1)
            x_mix = lam * x + (1 - lam) * x_shuf
            y_mix = lam_y * y + (1 - lam_y) * y_shuf
            return x_mix, y_mix

    # ── Data Augmentation ─────────────────────────────────────
    # Train: con augmentation para mejorar generalización
    # Validation: solo rescale, sin augmentation
    report_progress(task_dir, status="running", phase="preparing",
                    message="Organizando dataset...", progress_pct=2)

    import pandas as pd
    import random as _random
    _random.seed(42)

    # Collect all image paths and labels
    file_paths = []
    labels = []
    class_names_found = sorted([d for d in os.listdir(DATASET_DIR)
                                if (DATASET_DIR / d).is_dir()])
    if include_list:
        class_names_found = [c for c in class_names_found if c in include_list]

    if not class_names_found:
        logger.error("No se encontraron clases en el dataset")
        report_progress(task_dir, status="error", message="No hay clases en el dataset")
        sys.exit(1)

    class_to_idx = {name: i for i, name in enumerate(class_names_found)}

    for cls_name in class_names_found:
        cls_dir = DATASET_DIR / cls_name
        imgs = [str(f) for f in cls_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]

        # Capping diferenciado para no_cerdo
        cap = MAX_NO_CERDO if cls_name.lower() == "no_cerdo" else MAX_IMAGES_PER_CLASS
        if len(imgs) > cap:
            imgs = _random.sample(imgs, cap)
            logger.info(f"   [{cls_name}] Cap {cap}: {len(imgs)} de {len(os.listdir(cls_dir))}")

        # Oversampling sintético para clases con pocas imágenes
        if len(imgs) < OVERSAMPLE_MIN:
            n_reps = (OVERSAMPLE_MIN // len(imgs)) + 1
            imgs = (imgs * n_reps)[:OVERSAMPLE_MIN]
            logger.info(f"   [{cls_name}] Oversampling: {len(os.listdir(cls_dir))} → {len(imgs)}")

        file_paths.extend(imgs)
        labels.extend([class_to_idx[cls_name]] * len(imgs))

    logger.info(f"Total imágenes: {len(file_paths)}")
    logger.info(f"Clases: {class_names_found}")

    # Stratified split: 70% train, 15% val, 15% test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=(VALIDATION_SPLIT + TEST_SPLIT),
        stratify=labels, random_state=42
    )

    val_ratio = VALIDATION_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_ratio),
        stratify=temp_labels, random_state=42
    )

    logger.info(f"   Train: {len(train_paths)} imágenes")
    logger.info(f"   Val: {len(val_paths)} imágenes")
    logger.info(f"   Test: {len(test_paths)} imágenes")

    # Create DataFrames for flow_from_dataframe
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    train_df = pd.DataFrame({'filename': train_paths, 'class': [idx_to_class[l] for l in train_labels]})
    val_df = pd.DataFrame({'filename': val_paths, 'class': [idx_to_class[l] for l in val_labels]})

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.6, 1.4],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True, seed=42,
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False, seed=42,
    )

    num_classes = len(class_names_found)
    class_names = class_names_found
    logger.info(f"   Clases encontradas: {num_classes}")
    logger.info(f"   Nombres: {class_names}")

    # ── Class Weights ─────────────────────────────────────────
    # Balancea clases con pocas imágenes dándoles más peso en la pérdida
    report_progress(task_dir, status="running", phase="preparing",
                    message="Calculando class weights...", progress_pct=5,
                    classes_found=class_names)

    if USE_CLASS_WEIGHTS:
        labels = train_generator.classes
        unique_classes = np.unique(labels)
        weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
        class_weight_dict = dict(zip(unique_classes, weights))

        logger.info("   Pesos por clase:")
        for cls_idx, weight in sorted(class_weight_dict.items()):
            cls_name = class_names_found[cls_idx]
            count = int(np.sum(np.array(train_labels) == cls_idx))
            logger.info(f"   [{cls_idx}] {cls_name}: {count} imgs, weight={weight:.3f}")
    else:
        class_weight_dict = None
        logger.info("   Class weights deshabilitados por config")

    # ── Arquitectura del Modelo ───────────────────────────────
    report_progress(task_dir, status="running", phase="building_model",
                    message="Construyendo modelo MobileNetV2...", progress_pct=7)

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False  # Se congela para fase 1

    # Cabeza de clasificación personalizada
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(gamma=FOCAL_GAMMA),
        metrics=["accuracy"],
    )

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    logger.info(f"   Total params: {total_params:,}")
    logger.info(f"   Trainable params: {trainable_params:,}")

    # ═══════════════════════════════════════════════════════════
    # FASE 1: Frozen — solo entrena la cabeza
    # ═══════════════════════════════════════════════════════════
    logger.info(f"\n[Fase 1] Entrenando cabeza ({EPOCHS_FROZEN} epochs)...")

    callbacks_phase1 = [
        ModelCheckpoint(str(OUTPUT_DIR / "best_model_phase1.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
        ProgressCallback(task_dir, EPOCHS_FROZEN, phase="frozen"),
    ]

    train_gen_phase1 = MixupGenerator(train_generator, alpha=MIXUP_ALPHA) if USE_MIXUP else train_generator
    cw_phase1 = None if USE_MIXUP else class_weight_dict

    history1 = model.fit(
        train_gen_phase1,
        epochs=EPOCHS_FROZEN,
        validation_data=val_generator,
        class_weight=cw_phase1,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    if task_dir:
        try:
            with open(Path(task_dir) / "history_phase1.json", "w", encoding="utf-8") as _f:
                json.dump({
                    "loss": [float(v) for v in history1.history.get("loss", [])],
                    "accuracy": [float(v) for v in history1.history.get("accuracy", [])],
                    "val_loss": [float(v) for v in history1.history.get("val_loss", [])],
                    "val_accuracy": [float(v) for v in history1.history.get("val_accuracy", [])],
                    "lr": [float(v) for v in history1.history.get("lr", [])],
                }, _f)
        except Exception:
            pass

    best_val_acc_phase1 = max(history1.history["val_accuracy"])
    logger.info(f"   Mejor val_accuracy fase 1: {best_val_acc_phase1:.4f}")

    # ═══════════════════════════════════════════════════════════
    # FASE 2: Fine-tuning — descongela últimas N capas
    # ═══════════════════════════════════════════════════════════
    logger.info(f"\n[Fase 2] Fine-tuning (últimas {UNFREEZE_LAST_N} capas, {EPOCHS_FINETUNE} epochs, LR {FINETUNE_LR})...")

    for layer in base_model.layers[-UNFREEZE_LAST_N:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=FINETUNE_LR),
        loss=focal_loss(gamma=FOCAL_GAMMA),
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        EarlyStopping(monitor="val_accuracy", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(OUTPUT_DIR / "best_model_v2.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        ProgressCallback(task_dir, EPOCHS_FINETUNE, phase="finetune"),
    ]

    train_gen_phase2 = MixupGenerator(train_generator, alpha=MIXUP_ALPHA) if USE_MIXUP else train_generator
    cw_phase2 = None if USE_MIXUP else class_weight_dict

    history2 = model.fit(
        train_gen_phase2,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_generator,
        class_weight=cw_phase2,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    if task_dir:
        try:
            with open(Path(task_dir) / "history_phase2.json", "w", encoding="utf-8") as _f:
                json.dump({
                    "loss": [float(v) for v in history2.history.get("loss", [])],
                    "accuracy": [float(v) for v in history2.history.get("accuracy", [])],
                    "val_loss": [float(v) for v in history2.history.get("val_loss", [])],
                    "val_accuracy": [float(v) for v in history2.history.get("val_accuracy", [])],
                    "lr": [float(v) for v in history2.history.get("lr", [])],
                }, _f)
        except Exception:
            pass

    best_val_acc_phase2 = max(history2.history["val_accuracy"])
    logger.info(f"   Mejor val_accuracy fase 2: {best_val_acc_phase2:.4f}")

    # ── Guardado de modelo y classes ──────────────────────────
    final_model_path = OUTPUT_DIR / MODEL_NAME
    model.save(str(final_model_path))
    logger.info(f"   Modelo guardado: {final_model_path}")

    classes_data = {
        "classes": class_names,
        "num_classes": num_classes,
        "threshold": 0.20,
        "model_input_size": list(IMG_SIZE),
        "version": "2.0",
        "training_info": {
            "total_train_images": train_generator.samples,
            "total_val_images": val_generator.samples,
            "best_val_accuracy_phase1": float(best_val_acc_phase1),
            "best_val_accuracy_phase2": float(best_val_acc_phase2),
            "epochs_phase1": EPOCHS_FROZEN,
            "epochs_phase2": EPOCHS_FINETUNE,
            "augmentation": True,
            "class_weights": True,
            "fine_tuning": True,
            "oversampling_min": OVERSAMPLE_MIN,
            "max_no_cerdo": MAX_NO_CERDO,
            "unfreeze_layers": UNFREEZE_LAST_N,
            "mixup": USE_MIXUP,
            "mixup_alpha": MIXUP_ALPHA,
        },
    }

    classes_path = OUTPUT_DIR / "classes.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes_data, f, indent=2, ensure_ascii=False)
    logger.info(f"   Classes guardado: {classes_path}")

    # Save test set metadata for evaluation
    test_df = pd.DataFrame({'filename': test_paths, 'class': [idx_to_class[l] for l in test_labels]})
    test_df.to_csv(OUTPUT_DIR / "test_set.csv", index=False)
    logger.info(f"   Test set guardado: {OUTPUT_DIR / 'test_set.csv'} ({len(test_paths)} imágenes)")

    # Compare phase 2 with phase 1 before copying to production
    if best_val_acc_phase2 <= best_val_acc_phase1:
        logger.warning(f"Fase 2 ({best_val_acc_phase2:.4f}) no mejoró Fase 1 ({best_val_acc_phase1:.4f})")
        logger.warning("Usando modelo de Fase 1 para producción")
        phase1_model_path = OUTPUT_DIR / "best_model_phase1.keras"
        if phase1_model_path.exists():
            model = tf.keras.models.load_model(str(phase1_model_path), compile=False)
            model.save(str(final_model_path))
        classes_data["training_info"]["best_val_accuracy_final"] = float(best_val_acc_phase1)
    else:
        classes_data["training_info"]["best_val_accuracy_final"] = float(best_val_acc_phase2)

    # Model registry — historial de versiones
    registry_path = BASE_DIR / "model_registry.json"
    registry = []
    if registry_path.exists():
        with open(registry_path) as _f:
            registry = json.load(_f)
    registry.append({
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "date": datetime.now().isoformat(),
        "val_accuracy_phase1": float(best_val_acc_phase1),
        "val_accuracy_phase2": float(best_val_acc_phase2),
        "val_accuracy_final": float(classes_data["training_info"]["best_val_accuracy_final"]),
        "num_classes": num_classes,
        "total_train_images": train_generator.samples,
        "total_val_images": val_generator.samples,
        "model_path": str(final_model_path),
    })
    with open(registry_path, "w") as _f:
        json.dump(registry[-20:], _f, indent=2)  # keep last 20 entries
    logger.info(f"   Registry guardado: {registry_path} ({len(registry)} entradas)")

    # Copia a producción (modelo_identificacion_cerdos.h5 raíz)
    prod_model_path = BASE_DIR / "modelo_identificacion_cerdos.h5"
    shutil.copy2(str(final_model_path), str(prod_model_path))
    logger.info(f"   Modelo copiado a produccion: {prod_model_path}")

    prod_classes_path = BASE_DIR / "classes.json"
    shutil.copy2(str(classes_path), str(prod_classes_path))
    logger.info(f"   Classes copiado a produccion: {prod_classes_path}")

    # Copia adicional a Laragon ML si está configurado
    laragon_ml_env = os.environ.get("LARAGON_ML_PATH", "")
    if laragon_ml_env:
        laragon_ml = Path(laragon_ml_env)
        if laragon_ml.exists():
            shutil.copy2(str(final_model_path), str(laragon_ml / "modelo_identificacion_cerdos.h5"))
            shutil.copy2(str(classes_path), str(laragon_ml / "classes.json"))
            logger.info(f"   Modelo copiado a Laragon ML: {laragon_ml}")

    # Reporte final
    report_progress(
        task_dir,
        status="completed",
        phase="done",
        current_epoch=EPOCHS_FINETUNE,
        total_epochs=EPOCHS_FINETUNE,
        best_val_accuracy=float(best_val_acc_phase2),
        progress_pct=100,
        message=f"Entrenamiento completado. Accuracy: {best_val_acc_phase2:.2%}",
        finished_at=datetime.now().isoformat(),
    )

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info(f"   Val accuracy final: {best_val_acc_phase2:.1%}")
    logger.info(f"   Clases: {num_classes}")
    logger.info(f"   Modelo: {final_model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
