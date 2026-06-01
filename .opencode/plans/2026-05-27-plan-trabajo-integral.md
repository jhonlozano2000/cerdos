# Plan de Trabajo Integral — Porci-Integral

## Alcance

Tres proyectos sincronizados:
- **IA**: `C:\Users\jhonl\Desktop\cerdos` — FastAPI + MobileNetV2 + YOLOv8
- **Backend**: `C:\laragon\www\Porci-Integral-backend` — Laravel 12 + Sanctum
- **Frontend**: `C:\Users\jhonl\Desktop\Porci-Integral-Frontend` — React 19 + Tailwind v4

---

## FASE 0 — Crisis Mode (IA debe funcionar)

Orden obligatorio — cada paso depende del anterior.

### 0.1 Crear `.env` en cerdos/ y arreglar parser

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| IA | `cerdos/.env` (crear), `app_fastapi.py:39-46`, `entrenar_v2.py:39-47` | Crear `.env` con `DATASET_PATH=C:\laragon\www\Porci-Integral-backend\storage\app\public\fotos_animales`. Arreglar parser manual: agregar `.strip('"').strip("'")` a los valores parseados. |

### 0.2 Poblar `fotos_animales/` con datos reales

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| IA / Backend | `limpiar_dataset.py` (cerdos), storage | Sync `dataset_procesado/` (14 clases, ~4k imágenes) → `fotos_animales/`. Cada imagen debe estar en `fotos_animales/{class_name}/`. Ejecutar script `limpiar_dataset.py` para validar imágenes corruptas, luego copiar árbol de directorios. |

### 0.3 Thread-safety en modelo + CORS fix

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| IA | `app_fastapi.py:114-120,153-168,614-676` | **CORS**: Cambiar `allow_origins=["*"]` + `allow_credentials=True` → `allow_origins=["http://localhost:5173", "http://porci-integral-backend.test"]`. **Thread-safety**: Agregar `threading.Lock()` global. Envolver toda carga de modelo (`load_model`) y toda inferencia (`model.predict`) con `with model_lock:`. Idem para `recargar_modelo()` y `restaurar_modelo()`. |

### 0.4 Crear `/reconocer-completo` + arreglar `detector/status`

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| IA | `app_fastapi.py` | Agregar endpoint `POST /reconocer-completo` que acepte `{"imagen": "base64...", "incluir_deteccion": true}` y ejecute: 1) YOLO detect → bounding boxes, 2) Para cada box → crop → MobileNet predict top-3, 3) Retornar `{tensorflow_predictions, identified_as, yolo_detections}`. También crear `GET /detector/status`. |
| Frontend | `services/ia/CerdoDetectorService.jsx` | Agregar función `reconocerCompleto(imagenBase64, incluirDeteccion)` que llame al nuevo endpoint. |

### 0.5 Wire feedback loop (`/confirmar` persiste + gatilla reentrenamiento)

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| IA | `app_fastapi.py:317-329` | `/confirmar` debe: 1) Recibir `imagen`, `clase_predicha`, `clase_confirmada`, `usuario_id`. 2) Guardar la imagen en `DATASET_PATH/{clase_confirmada}/confirm_{timestamp}_{uuid}.jpg`. 3) Si hay ≥ 50 confirmaciones nuevas, disparar reentrenamiento ligero. |
| Frontend | `pages/ia/DetectorCerdas.jsx`, `services/ia/CerdoDetectorService.jsx` | Después de que usuario confirma la cerda correcta, llamar a `/confirmar` con imagen + clases. |

---

## FASE 1 — Arquitectura (IA + Backend)

### 1.1 Rate limiting + autenticación en IA

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| IA | `app_fastapi.py` | Agregar `slowapi` con `@limiter.limit("60/minute")` en `/reconocer*`. `@limiter.limit("10/minute")` en `/entrenar`, `/modelo/*`. |
| IA | `app_fastapi.py` | Agregar API Key via Header `X-API-Key`. Endpoints admin requieren key; públicos rate-limited. |

### 1.2 Centralizar tráfico IA a través de Laravel

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| Frontend | `services/ia/CerdoDetectorService.jsx` | Eliminar `API_URL = "http://127.0.0.1:8000"`. Cambiar todas las funciones para que usen `axiosClient` (Laravel proxy). |
| Frontend | `services/ia/` | Unificar en `iaService.js` (reemplazar `CerdoDetectorService.jsx` + `EntrenamientoService.jsx`). |

### 1.3 Split estratificado + capping de clases mayoritarias

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| IA | `entrenar_v2.py:63,166-216` | 1) `train_test_split` con `stratify=y`. 2) Implementar `MAX_IMAGES_PER_CLASS=300`. 3) Test set separado 70/15/15. 4) ModelCheckpoint también en Fase 1. 5) Validar que Fase 2 mejore a Fase 1 antes de copiar. |

### 1.4 CCTV fix

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| IA | `cctv_monitor.py:32,70` | 1) Usar modelo custom YOLO o filtrar class_id=19. 2) Usar `cv2.INTER_LINEAR` explícito. 3) Verificar `len(roi.shape)==3` antes de cvtColor. |

### 1.5 Stderr del training separado

| Proyecto | Archivos | Qué hacer |
|----------|----------|-----------|
| IA | `training_manager.py:177` | Cambiar `stderr=subprocess.STDOUT` → `stderr=subprocess.PIPE`. Loguear a `logs/train_{task_id}_error.log`. |

---

## FASE 2 — Módulo HC — Backend Migraciones + Modelos

### 2.1 Migraciones (4 nuevas)

| Proyecto | Migración | Archivo destino |
|----------|-----------|-----------------|
| Backend | `create_hc_recetas_table` | `database/migrations/2026_05_27_000001_create_hc_recetas_table.php` |
| Backend | `create_hc_receta_items_table` | `database/migrations/2026_05_27_000002_create_hc_receta_items_table.php` |
| Backend | `create_hc_dispensaciones_table` | `database/migrations/2026_05_27_000003_create_hc_dispensaciones_table.php` |
| Backend | `create_config_umbrales_iot_table` | `database/migrations/2026_05_27_000004_create_config_umbrales_iot_table.php` |

### 2.2 Modelos (4 nuevos)

| Proyecto | Modelo | Archivo destino |
|----------|--------|-----------------|
| Backend | `HcReceta` | `app/Models/HC/HcReceta.php` |
| Backend | `HcRecetaItem` | `app/Models/HC/HcRecetaItem.php` |
| Backend | `HcDispensacion` | `app/Models/HC/HcDispensacion.php` |
| Backend | `ConfigUmbralIot` | `app/Models/Configuracion/ConfigUmbralIot.php` |

### 2.3 FormRequests (4 nuevos)

| Proyecto | Request | Archivo destino |
|----------|---------|-----------------|
| Backend | `StoreRecetaRequest` | `app/Http/Requests/HC/StoreRecetaRequest.php` |
| Backend | `UpdateRecetaRequest` | `app/Http/Requests/HC/UpdateRecetaRequest.php` |
| Backend | `StoreDispensacionRequest` | `app/Http/Requests/HC/StoreDispensacionRequest.php` |
| Backend | `StoreUmbralIotRequest` | `app/Http/Requests/Configuracion/StoreUmbralIotRequest.php` |

### 2.4 Controladores + Rutas (3 nuevos)

| Proyecto | Controlador | Archivo destino |
|----------|-------------|-----------------|
| Backend | `HcRecetaController` | `app/Http/Controllers/Hc/HcRecetaController.php` — CRUD + `porAtencion`, `porAnimal`, `dispensar` |
| Backend | `HcDispensacionController` | `app/Http/Controllers/Hc/HcDispensacionController.php` — index, store, show |
| Backend | `ConfigUmbralIotController` | `app/Http/Controllers/Configuracion/ConfigUmbralIotController.php` — CRUD |

**Rutas a agregar:**

`routes/api-hc.php`:
```php
Route::apiResource('recetas', HcRecetaController::class);
Route::get('recetas/atencion/{atencionId}', [HcRecetaController::class, 'porAtencion']);
Route::get('recetas/animal/{animalId}', [HcRecetaController::class, 'porAnimal']);
Route::post('recetas/{recetaItem}/dispensar', [HcRecetaController::class, 'dispensar']);
Route::apiResource('dispensaciones', HcDispensacionController::class)->only(['index', 'store', 'show']);
```

`routes/api-configuracion.php` (o en `routes/api-iot.php`):
```php
Route::apiResource('umbrales-iot', ConfigUmbralIotController::class);
```

### 2.5 Validación de gestación en recetas

En `StoreRecetaRequest::withValidator()`: Si `hc_atenciones.gestacion_bloqueada = true`, revisar cada `config_producto_id` contra productos abortivos. Agregar migración para columna `es_abortivo` (boolean, default false) en `config_productos`.

---

## FASE 3 — Módulo HC — Frontend Tablet

### 3.1 Componentes UI base

| Archivo | Descripción |
|---------|-------------|
| `src/components/ui/FatFingerButton.jsx` | Botón ≥56px, contraste alto, variant/size/loading |
| `src/components/ui/BigInput.jsx` | Input/textarea táctil h-14 text-lg |
| `src/components/ui/TouchSelect.jsx` | Selector en grid de botones 2-3 columnas |
| `src/components/ui/OfflineBanner.jsx` | Banner offline/online fijo abajo |

### 3.2 Componentes HC de dominio

| Archivo | Descripción |
|---------|-------------|
| `src/components/hc/AtencionFormFatFinger.jsx` | Formulario atención tablet: botones grandes, gestación toggle, estado selector |
| `src/components/hc/SignosVitalesInput.jsx` | Grid 3 BigInput: temp, FC, FR |
| `src/components/hc/CondicionCorporalSelector.jsx` | 5 botones 1-5 con color |
| `src/components/hc/RecetaForm.jsx` | Productos + dosis + vía + items list |
| `src/components/hc/DispensacionForm.jsx` | Cantidad + unidad + notas |
| `src/components/hc/HistoriaClinicaTimeline.jsx` | Timeline vertical táctil con eventos |

### 3.3 Páginas

| Archivo | Descripción |
|---------|-------------|
| `src/pages/hc/recetas/RecetaPage.jsx` | Detalle receta + dispensaciones |
| `src/pages/hc/atenciones/NuevaAtencionPage.jsx` | Tablet mode: FatFinger; Desktop: normal |
| `src/pages/hc/consultas/ConsultaPage.jsx` | Pestañas: Datos, Diagnósticos, Procedimientos, Recetas |
| `src/pages/iot/umbrales/UmbralesPage.jsx` | Lista galpones + umbrales editables |

### 3.4 Hooks + Services

| Archivo | Descripción |
|---------|-------------|
| `src/hooks/hc/useRecetas.js` | CRUD recetas + dispensar |
| `src/hooks/hc/useOfflineSync.js` | Cola FIFO localStorage + replay on reconnect |
| `src/services/hc/recetaService.js` | Axios calls para recetas y dispensaciones |

---

## FASE 4 — Pulido General

| # | Proyecto | Tarea | Archivos |
|---|----------|-------|----------|
| 4.1 | IA | Versionado automático: `model_registry.json` con fecha, métricas, hash | `entrenar_v2.py` |
| 4.2 | IA | `print()` → `logging` en todos los archivos | `app_fastapi.py`, `entrenar_v2.py`, `training_manager.py`, `cctv_monitor.py` |
| 4.3 | IA | Test set separado 70/15/15 | `entrenar_v2.py` |
| 4.4 | IA | Fix resize: `Image.Resampling.BILINEAR` explícito | `app_fastapi.py:177` |
| 4.5 | IA | Endpoint `/ready` para healthcheck readiness | `app_fastapi.py` |
| 4.6 | IA | CCTV log rotation (100MB por archivo) | `cctv_monitor.py:92-101` |
| 4.7 | IA | CCTV reconexión con backoff exponencial | `cctv_monitor.py:163-167` |
| 4.8 | IA | CCTV threshold desde `classes.json` (no hardcode) | `cctv_monitor.py:35` |
| 4.9 | IA | Artisan command `ia:sync-dataset` | Nuevo comando Laravel |
| 4.10 | Backend | Cachear `dataset-estadisticas` con TTL 60s | `MLSyncService.php` |
| 4.11 | Backend | Columna `es_abortivo` en `config_productos` | Migración + StoreRecetaRequest |
| 4.12 | Frontend | Error handling en servicios IA unificados | `services/ia/iaService.js` |
| 4.13 | Frontend | Offline queue: persistir operaciones pendientes | `hooks/hc/useOfflineSync.js` |

---

## Dependencias entre fases

```
FASE 0 (IA Crisis) ──→ FASE 1 (Arquitectura IA) ──→ FASE 4 (Pulido IA)
                                                         │
FASE 2 (Backend HC) ──→ FASE 3 (Frontend HC) ───────────┤
                                                         │
FASE 0.2 (Dataset) ──────────────────────────────────────┘
```

**FASE 0** y **FASE 2** pueden ejecutarse en paralelo (no comparten archivos).
**FASE 3** depende de FASE 2 (necesita las rutas API).
**FASE 1** depende de FASE 0 (necesita IA funcionando).
**FASE 4** puede empezar después de FASE 1 y FASE 3.

---

## Criterios de verificación

| Fase | Verificación |
|------|-------------|
| 0 | `GET /salud` → 200. `GET /reconocer-completo` con imagen real → predicciones. `GET /dataset-estadisticas` → N>0 imágenes. |
| 1 | 61 requests a `/reconocer` en 1 min → 429. Sin API key → 403 en `/entrenar`. |
| 2 | `php artisan migrate` → 4 migraciones OK. `POST /api/hc/recetas` → 201. |
| 3 | Componentes se renderizan en tablet (≤768px). OfflineBanner cambia al desconectar. |
| 4 | `model_registry.json` existe. Logs rotan sin acumular. |
