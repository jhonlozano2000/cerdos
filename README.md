# Porci-Integral: Servicio de IA para Reconocimiento de Animales

Servicio de inteligencia artificial para identificación biométrica de cerdas usando **MobileNetV2** + **YOLOv8**.

## Stack Tecnológico

| Componente         | Tecnología                        |
| ------------------ | --------------------------------- |
| API                | FastAPI + Uvicorn                 |
| Clasificación      | TensorFlow / Keras (MobileNetV2)  |
| Detección          | YOLOv8 (Ultralytics)              |
| Monitoreo CCTV     | OpenCV + RTSP + YOLO + MobileNet  |

## Estructura

```
cerdos/
├── app_fastapi.py                 # API principal (FastAPI)
├── entrenar_v2.py                 # Entrenamiento MobileNetV2 (2 fases)
├── training_manager.py            # Gestor de entrenamiento async
├── cctv_monitor.py                # Monitoreo CCTV en tiempo real
├── modelo_identificacion_cerdos.h5     # Modelo en producción
├── model_backups/                 # Backups automáticos del modelo (max 3)
├── output_v2/                     # Modelos entrenados + classes.json
├── training_tasks/                # Progreso de entrenamientos (auto-limpieza)
├── .env                           # Configuración (DATASET_PATH)
└── env.example                    # Plantilla para .env
```

## Iniciar el Servicio

```bash
.venv\Scripts\activate
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

## Endpoints

### Generales
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/`                       | GET    | Health check simple                  |
| `/salud`                  | GET    | Estado completo (modelo, dataset, clases) |
| `/clases`                 | GET    | Lista de clases disponibles          |
| `/reconocer`              | POST   | Reconocer cerda por imagen (file)    |
| `/reconocer_base64`       | POST   | Reconocer cerda por imagen (base64)  |
| `/detectar`               | POST   | Detectar cerdas en imagen (YOLO)     |
| `/detector/status`        | GET    | Estado del detector YOLO             |
| `/exportar-foto`          | POST   | Guardar foto en el dataset           |
| `/confirmar`              | POST   | Registrar confirmación de IA         |
| `/detectar-enfermedad`    | POST   | Evaluación de riesgo sanitario       |

### Dataset
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/dataset-estadisticas`   | GET    | Estadísticas del dataset             |

### Entrenamiento
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/entrenar`               | POST   | Iniciar entrenamiento async          |
| `/entrenar/historial`     | GET    | Historial de entrenamientos          |
| `/entrenar/{task_id}`     | GET    | Estado del entrenamiento             |
| `/entrenar/activo`        | GET    | Entrenamiento activo (si existe)     |

### Modelo
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/modelo/recargar`        | POST   | Recargar modelo desde disco          |
| `/modelo/versiones`       | GET    | Lista backups disponibles            |
| `/modelo/restaurar`       | POST   | Restaurar un backup                  |

## Detección Temprana de Enfermedades

```bash
curl -X POST http://127.0.0.1:8000/detectar-enfermedad \
  -H "Content-Type: application/json" \
  -d '{"animal_id": 5, "peso_actual_kg": 62, "peso_anterior_kg": 68, "dias_entre_pesajes": 10}'
```

Reglas de evaluación:
- Pérdida >5% en 7 días → ALTO
- Pérdida >3% en 7 días → MEDIO
- Sin ganancia en <90 días → MEDIO
- Temperatura ≥40°C → ALERTA

## Monitoreo CCTV en Tiempo Real

```bash
# Webcam local
python cctv_monitor.py --source 0

# Cámara IP (RTSP)
python cctv_monitor.py --source "rtsp://usuario:pass@camara:554/stream"

# Con confianza personalizada
python cctv_monitor.py --source 0 --confidence 0.6
```

Controles:
- `q` — Salir
- `p` — Pausar/Reanudar
- `s` — Guardar frame actual

Detecciones guardadas en `cctv_log.jsonl` para análisis posterior.

## Entrenamiento

```bash
curl -X POST http://127.0.0.1:8000/entrenar \
  -H "Content-Type: application/json" \
  -d '{"include_classes": ["cerda_001", "reproductor"]}'
```

### Fases
1. **Frozen (15 epochs)**: Base del modelo congelada
2. **Fine-tuning (20 epochs)**: Últimas 30 capas descongeladas

### Backup automático
Antes de cada entrenamiento se crea un backup. Máximo 3 versiones.

### Gestión de tareas
- Persisten en disco (`training_tasks/tasks_index.json`)
- Tareas "running" al reiniciar pasan a "error"
- `get_status()` recupera progreso desde disco si la tarea fue interrumpida por reinicio pero el subproceso sigue activo
- Limpieza automática (máximo 15 tareas completadas)

## Dataset

```
{DATASET_PATH}/
├── cerda_001/
├── cerda_002/
└── ...
```

Configurable via `DATASET_PATH` en `.env`.

## Notas

- **Tiempo estimado entrenamiento**: 30-60 min en CPU
- MobileNetV2 con pesos ImageNet
- Data augmentation: rotación, zoom, flip, brillo
- Class weighting automático

## Setup Portátil (sin Laragon)

### Requisitos
- Python 3.10+ (descargar de python.org)
- Pip instalado

### Instalación

```powershell
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar
.\.venv\Scripts\Activate.ps1   # PowerShell
# o
.\.venv\Scripts\activate       # CMD

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno (crear .env)
cp env.example .env
```

### Configuración del .env

```env
# Ruta al dataset de fotos de animales (backend Laravel)
DATASET_PATH=C:\laragon\www\Porci-Integral-backend\storage\app\public\fotos_animales

# API Key para autenticación con backend (opcional)
AI_API_KEY=tu_api_key_aqui
```

### Iniciar Servicio

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

Verificar en: `http://127.0.0.1:8000/docs` (Swagger UI interactivo)

### Solución de Problemas

| Problema | Solución |
|----------|----------|
| `Fatal error in launcher` | Recrear venv: `rmdir /s /q .venv && python -m venv .venv` |
| `ExecutionPolicy` | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Modelo no carga | Verificar que `modelo_identificacion_cerdos.h5` existe en la raíz |
| Dataset no encontrado | Configurar `DATASET_PATH` en `.env` con ruta válida |
