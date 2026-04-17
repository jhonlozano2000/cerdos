"""
Entrenamiento Optimizado para Identificación de Cerdas
Versión rápida usando ImageDataGenerator de Keras
"""

import os
import numpy as np
import random
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 15
SEED = 42

BASE_DIR = Path("dataset_procesado")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 50)
print("🎯 Entrenamiento Rápido - Identificación de Cerdas")
print("=" * 50)

def get_class_directories():
    classes = sorted([d for d in os.listdir(BASE_DIR) if (BASE_DIR / d).is_dir()])
    print(f"\n📁 Clases encontradas: {len(classes)}")
    total = 0
    for c in classes:
        count = len(list((BASE_DIR / c).glob("*")))
        print(f"   - {c}: {count} imágenes")
        total += count
    print(f"\n📊 Total: {total} imágenes")
    return classes

def split_dataset_stratified(classes, test_size=0.2):
    """Divide el dataset en train/val/test de forma estratificada"""
    
    all_images = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for class_name in classes:
        class_dir = BASE_DIR / class_name
        images = list(class_dir.glob("*"))
        for img_path in images:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                all_images.append({
                    'path': str(img_path),
                    'class_name': class_name,
                    'class_id': class_to_idx[class_name]
                })
    
    df = pd.DataFrame(all_images)
    print(f"\n📊 Total imágenes: {len(df)}")
    
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['class_id'],
        random_state=SEED
    )
    
    train, val = train_test_split(
        train_val,
        test_size=0.2,
        stratify=train_val['class_id'],
        random_state=SEED
    )
    
    print(f"   Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test

def create_generators(train_df, val_df, classes):
    """Crea generadores optimizados"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='path',
        y_col='class_name',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True,
        seed=SEED
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='path',
        y_col='class_name',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False,
        seed=SEED
    )
    
    print(f"\n��� Clases: {train_generator.class_indices}")
    
    return train_generator, val_generator

def build_model(num_classes):
    """Construye el modelo con MobileNetV2"""
    
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_and_finetune(model, unfreeze_from=100):
    """Descongela capas para fine-tuning"""
    base_model = model.layers[1]
    base_model.trainable = True
    
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Fine-tuning: {sum([layer.trainable for layer in base_model.layers])} capas entrenables")

def create_test_generator(test_df):
    """Crea generador para test"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='path',
        y_col='class_name',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False,
        seed=SEED
    )
    
    return test_generator

def main():
    classes = get_class_directories()
    num_classes = len(classes)
    
    train_df, val_df, test_df = split_dataset_stratified(classes, test_size=0.2)
    
    train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_split.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)
    print(f"\n💾 Splits guardados en {OUTPUT_DIR}")
    
    train_ds, val_ds = create_generators(train_df, val_df, classes)
    test_ds = create_test_generator(test_df)
    
    steps_per_epoch = train_ds.samples // BATCH_SIZE
    validation_steps = val_ds.samples // BATCH_SIZE
    
    print(f"\n⚡ Steps por epoch: {steps_per_epoch}")
    print(f"⚡ Batch size: {BATCH_SIZE}")
    
    print("\n🔧 Construyendo modelo...")
    model = build_model(num_classes)
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        ModelCheckpoint(
            str(OUTPUT_DIR / "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n🚀 Fase 1: Entrenamiento inicial...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    print("\n🔧 Fase 2: Fine-tuning...")
    unfreeze_and_finetune(model, unfreeze_from=80)
    
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    print("\n🧪 Evaluando en test set...")
    test_results = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_results[1]:.2%}")
    
    model.save(OUTPUT_DIR / "modelo_final.keras")
    print(f"\n✅ Modelo guardado en: {OUTPUT_DIR / 'modelo_final.keras'}")

if __name__ == "__main__":
    main()