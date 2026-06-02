# AGENTS.md

## Project Overview

Biometric identification system for individual sows (cerdas). Two-stage pipeline:
1. **YOLOv8** detects pig bounding boxes in images
2. **MobileNetV2** (transfer learning) classifies which specific sow it is

12 classes: `cerda_001` through `cerda_012`. Dataset is heavily imbalanced (cerda_012: 1458 imgs, cerda_009: 14 imgs).

## Environment

- **Python**: 3.10.6 via Laragon (`C:\laragon\bin\python\python-3.10`)
- **Venv**: project root IS the venv (Scripts/, pyvenv.cfg at root)
- **Activate**: `.\Scripts\activate`
- **OS**: Windows — all paths use backslashes; project path varies by user (`C:\Users\<username>\Desktop\cerdos`). Use relative paths or environment variables when possible.
- **Hardware**: CPU only — no GPU available. Training is slow (~20 min/epoch).

## Key Commands

```powershell
# Activate venv (run from project root)
.\Scripts\activate

# Train classification model (MobileNetV2, 2-phase: frozen then fine-tune)
python entrenamiento_mejorado.py   # Albumentations augmentation, 30 epochs
python entrenamiento_rapido.py     # Keras ImageDataGenerator, 15+5 epochs

# Train YOLO detector
python entrenar_yolo.py            # 10 epochs, CPU, outputs to output/yolo_detector/runs/

# Run Streamlit UI
streamlit run app.py

# Run FastAPI server
uvicorn app_fastapi:app --reload   # port 8000

# Clean/validate dataset
python limpiar_dataset.py

# Inference (CLI)
python inference.py
python detector_identificador.py   # Interactive menu: YOLO detect → MobileNet identify
```

## Architecture

| File | Role |
|------|------|
| `entrenamiento_mejorado.py` | Best training script — Albumentations, stratified split, 2-phase fine-tune |
| `entrenamiento_rapido.py` | Faster training — Keras ImageDataGenerator, same architecture |
| `entrenar_yolo.py` | YOLO training — uses full-image labels (`0 0.5 0.5 1 1`) as bounding box proxy |
| `detector_identificador.py` | Combined pipeline: YOLO detect → crop → MobileNet classify |
| `app_fastapi.py` | REST API: `/reconocer`, `/reconocer_base64`, `/detectar`, `/confirmar` |
| `app.py` | Streamlit UI for upload + identify + confirm |
| `limpiar_dataset.py` | Dataset validation: corrupt detection, rename, report |

## Model Details

- **Classification**: MobileNetV2 base (ImageNet) + GlobalAvgPool + Dense(128) + Dense(12, softmax)
- **Input size**: 224x224 RGB, normalized [0,1]
- **Loss**: sparse_categorical_crossentropy
- **Optimizer**: Adam (1e-3 frozen, 1e-5 fine-tune from layer 80)
- **Confidence threshold**: 50% — below this the prediction is "Desconocido"

## Dataset Structure

```
dataset_procesado/       # 12 folders, ~3790 images total (gitignored)
  cerda_001/ ... cerda_012/
dataset_test/            # Test split copies (gitignored)
imagenes_originales/     # Raw source photos (gitignored)
output/
  modelo_final.keras     # Best Keras model
  best_model.keras       # Checkpoint
  train_split.csv        # Stratified train/val/test split metadata
  val_split.csv
  test_split.csv
  yolo_detector/data.yaml
```

All image directories are **gitignored**. Only code, configs, and `.h5`/`.keras` models are tracked.

## Gotchas

- **YOLO labels are fake**: `entrenar_yolo.py` writes `0 0.5 0.5 1 1` for every image (full-image bounding box). The YOLO model learns "pig exists" but not precise localization.
- **Class imbalance**: cerda_012 has 100x more images than cerda_009. No class weighting is currently applied.
- **Venv at root**: The project root doubles as the Python venv — `Scripts/`, `share/`, `pyvenv.cfg` are venv artifacts, not project code.
- **modelo_identificacion_cerdos.h5** (root) vs **output/modelo_final.keras**: Two different model files. The FastAPI and Streamlit apps load the `.h5` at root; training scripts save to `output/`.
- **utils.py**: Shared module with `load_env()`, `load_classes()`, `load_threshold()`, `preprocess_image()`. Used by `app_fastapi.py`, `entrenar_v2.py`, `cctv_monitor.py`.
- **cctv_monitor.py**: `FRAME_SKIP=3` — YOLO runs every 3 frames to reduce CPU load; `last_detections` reused between frames.
- **entrenar_v2.py**: `EPOCHS_FINETUNE=30` (was 60) for faster training. Uses `LARAGON_ML_PATH` env var instead of hardcoded path.
