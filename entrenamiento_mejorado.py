"""
Entrenamiento Mejorado para Identificación de Cerdas
Características:
- Augmentation on-the-fly con Albumentations
- Fine-tuning progresivo
- Split estratificado
- Early stopping y learning rate scheduler
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

import albumentations as A
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
SEED = 42

BASE_DIR = Path("dataset_procesado")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 50)
print("🎯 Entrenamiento Mejorado - Identificación de Cerdas")
print("=" * 50)

def get_class_directories():
    classes = sorted([d for d in os.listdir(BASE_DIR) if (BASE_DIR / d).is_dir()])
    print(f"\n📁 Clases encontradas: {len(classes)}")
    for c in classes:
        count = len(list((BASE_DIR / c).glob("*")))
        print(f"   - {c}: {count} imágenes")
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

def create_augmentation_pipeline():
    """Pipeline de augmentación on-the-fly"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.Resize(*IMG_SIZE),
    ])

def load_and_augment_image(image_path, augment=None):
    """Carga y aplica augmentación a una imagen"""
    import cv2
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if augment:
        augmented = augment(image=img)
        img = augmented['image']
    
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

def create_dataset_from_dataframe(df, classes, augment=None, shuffle=False):
    """Crea un dataset de TensorFlow desde DataFrame"""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    
    def generator():
        while True:
            if shuffle:
                df_sample = df.sample(frac=1, random_state=SEED)
            else:
                df_sample = df
            
            for _, row in df_sample.iterrows():
                img = load_and_augment_image(row['path'], augment)
                label = class_to_idx[row['class_name']]
                
                yield img, label
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    return dataset

def build_model(num_classes):
    """Construye el modelo con fine-tuning"""
    
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

def main():
    classes = get_class_directories()
    num_classes = len(classes)
    
    train_df, val_df, test_df = split_dataset_stratified(classes, test_size=0.2)
    
    train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_split.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)
    print(f"\n💾 Splits guardados en {OUTPUT_DIR}")
    
    augment = create_augmentation_pipeline()
    
    train_ds = create_dataset_from_dataframe(train_df, classes, augment, shuffle=True)
    val_ds = create_dataset_from_dataframe(val_df, classes, augment=None, shuffle=False)
    test_ds = create_dataset_from_dataframe(test_df, classes, augment=None, shuffle=False)
    
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
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
            OUTPUT_DIR / "best_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n🚀 Fase 1: Entrenamiento inicial...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks
    )
    
    print("\n🔧 Fase 2: Fine-tuning...")
    unfreeze_and_finetune(model, unfreeze_from=80)
    
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks
    )
    
    print("\n🧪 Evaluando en test set...")
    test_results = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_results[1]:.2%}")
    
    model.save(OUTPUT_DIR / "modelo_final.h5")
    print(f"\n✅ Modelo guardado en: {OUTPUT_DIR / 'modelo_final.h5'}")

if __name__ == "__main__":
    main()
