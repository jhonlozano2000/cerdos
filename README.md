# Porci-Integral: Servicio de IA para Reconocimiento de Animales

Servicio de inteligencia artificial para identificación biométrica de cerdas usando **MobileNetV2** + **YOLOv8**.

## Stack Tecnológico

| Componente         | Tecnología                     |
| ------------------ | ------------------------------ |
| API                | FastAPI + Uvicorn              |
| Clasificación      | TensorFlow / Keras (MobileNetV2) |
| Detección          | YOLOv8 (Ultralytics)           |

## Estructura

```
cerdos/
├── app_fastapi.py                 # API principal (FastAPI)
├── entrenar_v2.py                 # Entrenamiento MobileNetV2 (2 fases)
├── training_manager.py            # Gestor de entrenamiento async
├── modelo_identificacion_cerdos.h5     # Modelo en producción
├── model_backups/                 # Backups automáticos del modelo (max 3)
├── output_v2/                     # Modelos entrenados + classes.json
├── training_tasks/                # Progreso de entrenamientos (auto-limpieza)
├── .env                           # Configuración (DATASET_PATH)
└── env.example                    # Plantilla para .env
```

Las imágenes se leen directamente desde el storage de Laravel:
```
{DATASET_PATH}/{numero_identificacion}/
```

## Configuración

La ruta al dataset se define en `.env`:

```env
DATASET_PATH=C:\laragon\www\Porci-Integral-backend\storage\app\public\fotos_animales
```

Si cambias de servidor, solo editas esta variable.

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

### Dataset
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/dataset-estadisticas`   | GET    | Estadísticas del dataset             |

### Entrenamiento
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/entrenar`               | POST   | Iniciar entrenamiento async          |
| `/entrenar/{task_id}`     | GET    | Estado del entrenamiento             |
| `/entrenar/historial`     | GET    | Historial de entrenamientos          |

### Modelo
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/modelo/recargar`        | POST   | Recargar modelo desde disco          |
| `/modelo/versiones`       | GET    | Lista backups disponibles            |
| `/modelo/restaurar`       | POST   | Restaurar un backup (`{"version":"20250525_120000"}`) |

## Entrenamiento

Se inicia desde el frontend **IA → Entrenamiento** o vía API:

```bash
curl -X POST http://127.0.0.1:8000/entrenar \
  -H "Content-Type: application/json" \
  -d '{"include_classes": ["cerda_001", "reproductor"]}'
```

### Fases
1. **Frozen (15 epochs)**: Base del modelo congelada
2. **Fine-tuning (20 epochs)**: Últimas 30 capas descongeladas

### Backup automático
Antes de cada entrenamiento se crea un backup del modelo actual en `model_backups/`.
Se mantienen las últimas 3 versiones. Se pueden listar y restaurar desde la API o el frontend.

### Gestión de tareas
- Las tareas persisten en disco (`training_tasks/tasks_index.json`) - sobreviven a reinicios
- Al reiniciar, tareas "running" pasan a "error" automáticamente
- Tareas completadas se limpian automáticamente (máximo 15)

## Dataset

```
storage/app/public/fotos_animales/
├── cerda_001/
├── cerda_002/
└── ...
```

Al eliminar un animal desde el backend, su directorio de fotos se elimina automáticamente.

## Notas

- **Tiempo estimado**: 30-60 min en CPU
- MobileNetV2 con pesos ImageNet
- Data augmentation: rotación, zoom, flip, brillo
- Class weighting automático
