"""
Entrenamiento Mejorado - Reconocimiento de Cerdas v2.0
======================================================
- MobileNetV2 con fine-tuning
- Data augmentation agresiva
- Class weights automaticos para balancear
- Early stopping + ReduceLROnPlateau
- Guarda classes.json junto al modelo

Uso: python entrenar_v2.py
Tiempo estimado: 30-60 min en CPU
"""

import os
import json
import numpy as np
from pathlib import Path

# Configuracion
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_procesado"
OUTPUT_DIR = BASE_DIR / "output_v2"
MODEL_NAME = "modelo_identificacion_cerdos_v2.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_FROZEN = 15      # Fase 1: base congelada
EPOCHS_FINETUNE = 20    # Fase 2: fine-tuning capas superiores
LEARNING_RATE = 0.0001
FINETUNE_LR = 0.00002
VALIDATION_SPLIT = 0.2
MAX_IMAGES_PER_CLASS = 300  # Limitar clases grandes para balancear

OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import (
        GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from sklearn.utils.class_weight import compute_class_weight

    print("=" * 60)
    print("ENTRENAMIENTO v2.0 - Reconocimiento de Cerdas")
    print("=" * 60)

    # =========================================================
    # 1. DATA AUGMENTATION
    # =========================================================
    print("\n[1/5] Configurando data augmentation...")

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

    train_generator = train_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_generator = val_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())
    print(f"   Clases encontradas: {num_classes}")
    print(f"   Nombres: {class_names}")
    print(f"   Imagenes train: {train_generator.samples}")
    print(f"   Imagenes val: {val_generator.samples}")

    # =========================================================
    # 2. CLASS WEIGHTS (balanceo automatico)
    # =========================================================
    print("\n[2/5] Calculando class weights...")

    labels = train_generator.classes
    unique_classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    class_weight_dict = dict(zip(unique_classes, weights))

    print("   Pesos por clase:")
    for idx, name in enumerate(class_names):
        count = np.sum(labels == idx)
        w = class_weight_dict.get(idx, 1.0)
        print(f"   [{idx}] {name}: {count} imgs, weight={w:.3f}")

    # =========================================================
    # 3. MODELO
    # =========================================================
    print("\n[3/5] Construyendo modelo MobileNetV2...")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False  # Congelar base inicialmente

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

    # =========================================================
    # 4. ENTRENAMIENTO FASE 1 (base congelada)
    # =========================================================
    print(f"\n[4/5] Fase 1: Entrenando cabeza ({EPOCHS_FROZEN} epochs)...")

    callbacks_phase1 = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
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

    # =========================================================
    # 5. ENTRENAMIENTO FASE 2 (fine-tuning)
    # =========================================================
    print(f"\n[5/5] Fase 2: Fine-tuning capas superiores ({EPOCHS_FINETUNE} epochs)...")

    # Descongelar las ultimas 30 capas de MobileNetV2
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            str(OUTPUT_DIR / "best_model_v2.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
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

    # =========================================================
    # GUARDAR MODELO Y CLASES
    # =========================================================
    final_model_path = OUTPUT_DIR / MODEL_NAME
    model.save(str(final_model_path))
    print(f"\n   Modelo guardado: {final_model_path}")

    # Guardar classes.json junto al modelo
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

    # Copiar modelo a ubicacion de produccion
    import shutil
    prod_model_path = BASE_DIR / "modelo_identificacion_cerdos.h5"
    shutil.copy2(str(final_model_path), str(prod_model_path))
    print(f"   Modelo copiado a produccion: {prod_model_path}")

    prod_classes_path = BASE_DIR / "classes.json"
    shutil.copy2(str(classes_path), str(prod_classes_path))
    print(f"   Classes copiado a produccion: {prod_classes_path}")

    # Tambien copiar al proyecto ML de laragon
    laragon_ml = Path(r"C:\laragon\www\Porci-Integral-ML")
    if laragon_ml.exists():
        shutil.copy2(str(final_model_path), str(laragon_ml / "modelo_identificacion_cerdos.h5"))
        shutil.copy2(str(classes_path), str(laragon_ml / "classes.json"))
        print(f"   Modelo copiado a Laragon ML: {laragon_ml}")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"   Val accuracy final: {best_val_acc_phase2:.1%}")
    print(f"   Clases: {num_classes}")
    print(f"   Modelo: {final_model_path}")
    print("=" * 60)
    print("\nPara usar el nuevo modelo, reinicia el servicio FastAPI:")
    print("   uvicorn app:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()
