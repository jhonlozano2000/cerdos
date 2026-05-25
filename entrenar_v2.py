"""
Entrenamiento Mejorado - Reconocimiento de Cerdas v2.0
======================================================
- MobileNetV2 con fine-tuning
- Data augmentation agresiva
- Class weights automaticos para balancear
- Early stopping + ReduceLROnPlateau
- Guarda classes.json junto al modelo
- Soporte para progreso via archivo JSON (entrenamiento gestionado)

Uso: python entrenar_v2.py [--task-id ID] [--task-dir DIR] [--include-classes LISTA]
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuracion
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(r"C:\laragon\www\Porci-Integral-backend\storage\app\public\fotos_animales")
OUTPUT_DIR = BASE_DIR / "output_v2"
MODEL_NAME = "modelo_identificacion_cerdos_v2.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_FROZEN = 15
EPOCHS_FINETUNE = 20
LEARNING_RATE = 0.0001
FINETUNE_LR = 0.00002
VALIDATION_SPLIT = 0.2
MAX_IMAGES_PER_CLASS = 300

OUTPUT_DIR.mkdir(exist_ok=True)

PROGRESS_FILE = "progress.json"


def report_progress(task_dir, **kwargs):
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
    parser.add_argument("--include-classes", default="", help="Clases a incluir separadas por coma")
    args = parser.parse_args()

    task_dir = args.task_dir
    include_list = [c.strip() for c in args.include_classes.split(",") if c.strip()] if args.include_classes else None

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

            progress_pct = int(((epoch + 1) / self.total_epochs) * 50)
            if self.phase == "finetune":
                progress_pct += 50

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
                progress_pct=progress_pct,
                message=f"Fase {self.phase}: época {epoch + 1}/{self.total_epochs} - acc: {acc:.4f} - val_acc: {val_acc:.4f}",
            )

    report_progress(task_dir, status="running", phase="preparing", message="Configurando data augmentation...", progress_pct=2)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    if include_list:
        datasets_to_use = [d for d in include_list if (DATASET_DIR / d).is_dir()]
        if not datasets_to_use:
            print("ERROR: Ninguna de las clases especificadas existe en el dataset")
            report_progress(task_dir, status="error", message="Ninguna clase especificada existe en el dataset")
            sys.exit(1)
        print(f"Usando solo clases: {datasets_to_use}")
    else:
        datasets_to_use = None

    train_generator = train_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
        classes=datasets_to_use,
    )

    val_generator = val_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
        classes=datasets_to_use,
    )

    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())
    print(f"   Clases encontradas: {num_classes}")
    print(f"   Nombres: {class_names}")
    print(f"   Imagenes train: {train_generator.samples}")
    print(f"   Imagenes val: {val_generator.samples}")

    report_progress(task_dir, status="running", phase="preparing", message="Calculando class weights...", progress_pct=5, classes_found=class_names)

    labels = train_generator.classes
    unique_classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    class_weight_dict = dict(zip(unique_classes, weights))

    print("   Pesos por clase:")
    for idx, name in enumerate(class_names):
        count = np.sum(labels == idx)
        w = class_weight_dict.get(idx, 1.0)
        print(f"   [{idx}] {name}: {count} imgs, weight={w:.3f}")

    report_progress(task_dir, status="running", phase="building_model", message="Construyendo modelo MobileNetV2...", progress_pct=7)

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    print(f"\n[Fase 1] Entrenando cabeza ({EPOCHS_FROZEN} epochs)...")

    callbacks_phase1 = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ProgressCallback(task_dir, EPOCHS_FROZEN, phase="frozen"),
    ]

    history1 = model.fit(
        train_generator,
        epochs=EPOCHS_FROZEN,
        validation_data=val_generator,
        class_weight=class_weight_dict,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    best_val_acc_phase1 = max(history1.history["val_accuracy"])
    print(f"   Mejor val_accuracy fase 1: {best_val_acc_phase1:.4f}")

    print(f"\n[Fase 2] Fine-tuning capas superiores ({EPOCHS_FINETUNE} epochs)...")

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(OUTPUT_DIR / "best_model_v2.h5"), monitor="val_accuracy", save_best_only=True, verbose=1),
        ProgressCallback(task_dir, EPOCHS_FINETUNE, phase="finetune"),
    ]

    history2 = model.fit(
        train_generator,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_generator,
        class_weight=class_weight_dict,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    best_val_acc_phase2 = max(history2.history["val_accuracy"])
    print(f"   Mejor val_accuracy fase 2: {best_val_acc_phase2:.4f}")

    final_model_path = OUTPUT_DIR / MODEL_NAME
    model.save(str(final_model_path))
    print(f"\n   Modelo guardado: {final_model_path}")

    classes_data = {
        "classes": class_names,
        "num_classes": num_classes,
        "threshold": 0.60,
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
        },
    }

    classes_path = OUTPUT_DIR / "classes.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes_data, f, indent=2, ensure_ascii=False)
    print(f"   Classes guardado: {classes_path}")

    import shutil
    prod_model_path = BASE_DIR / "modelo_identificacion_cerdos.h5"
    shutil.copy2(str(final_model_path), str(prod_model_path))
    print(f"   Modelo copiado a produccion: {prod_model_path}")

    prod_classes_path = BASE_DIR / "classes.json"
    shutil.copy2(str(classes_path), str(prod_classes_path))
    print(f"   Classes copiado a produccion: {prod_classes_path}")

    laragon_ml = Path(r"C:\laragon\www\Porci-Integral-ML")
    if laragon_ml.exists():
        shutil.copy2(str(final_model_path), str(laragon_ml / "modelo_identificacion_cerdos.h5"))
        shutil.copy2(str(classes_path), str(laragon_ml / "classes.json"))
        print(f"   Modelo copiado a Laragon ML: {laragon_ml}")

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

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"   Val accuracy final: {best_val_acc_phase2:.1%}")
    print(f"   Clases: {num_classes}")
    print(f"   Modelo: {final_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
