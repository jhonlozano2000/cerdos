"""
Training Manager — Gestor Asíncrono de Entrenamiento
=====================================================
Maneja el ciclo de vida de entrenamientos como tareas asíncronas:
- Inicia entrenamientos en threads separados
- Monitorea progreso vía archivos JSON
- Persiste estado en disco para sobrevivir reinicios
- Limpia tareas completadas automáticamente

El entrenamiento se ejecuta como subproceso usando el mismo
intérprete de Python (sys.executable) para garantizar que se
use el entorno virtual correcto.
"""

import os
import sys
import json
import uuid
import subprocess
import shutil
import threading
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
TASKS_DIR = BASE_DIR / "training_tasks"
TASKS_INDEX = "tasks_index.json"
PROGRESS_FILE = "progress.json"

MAX_TASKS_KEEP = 15  # Número máximo de tareas completadas a conservar


class TrainingManager:
    """
    Gestiona entrenamientos asíncronos del modelo.
    Cada entrenamiento se ejecuta en un thread daemon que lanza
    entrenar_v2.py como subproceso.
    """

    def __init__(self):
        self.tasks = {}     # En memoria: {task_id: progress_dict}
        self.lock = threading.Lock()
        TASKS_DIR.mkdir(exist_ok=True)
        self._load_tasks()         # Recupera tareas persistidas
        self._cleanup_orphaned()   # Incorpora tareas huérfanas en disco

    # ── API Pública ──────────────────────────────────────────

    def start_training(self, include_classes=None, exclude_classes=None) -> str:
        """
        Inicia un nuevo entrenamiento.
        - Valida que no haya otro en curso
        - Crea directorio para la tarea
        - Lanza thread daemon con _run_training
        - Limpia tareas viejas automáticamente

        Returns: task_id (string de 12 caracteres hexadecimales)
        Raises: RuntimeError si ya hay un entrenamiento en curso
        """
        with self.lock:
            for tid, t in self.tasks.items():
                if t["status"] in ("running", "starting"):
                    raise RuntimeError(f"Ya hay un entrenamiento en curso: {tid}")

            task_id = uuid.uuid4().hex[:12]
            task_dir = TASKS_DIR / task_id
            task_dir.mkdir(parents=True)

            progress = {
                "task_id": task_id,
                "status": "starting",
                "phase": "",
                "current_epoch": 0,
                "total_epochs": 0,
                "current_accuracy": 0,
                "best_val_accuracy": 0,
                "current_loss": 0,
                "current_val_accuracy": 0,
                "current_val_loss": 0,
                "progress_pct": 0,
                "message": "Iniciando...",
                "classes_found": [],
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
            }
            self._save_progress(task_dir, progress)
            self.tasks[task_id] = progress
            self._persist_tasks()

            self._cleanup_old_tasks()

            thread = threading.Thread(
                target=self._run_training,
                args=(task_id, task_dir, include_classes, exclude_classes),
                daemon=True,
            )
            thread.start()

            return task_id

    def get_status(self, task_id: str) -> dict:
        """
        Retorna el estado actual de una tarea.
        Busca primero en memoria, luego en disco.
        """
        with self.lock:
            if task_id in self.tasks:
                return self.tasks[task_id]

            task_dir = TASKS_DIR / task_id
            if task_dir.exists():
                return self._load_progress(task_dir) or {"status": "not_found"}

            return {"status": "not_found"}

    def cancel_training(self, task_id: str) -> bool:
        """
        Marca una tarea para cancelación.
        El thread de entrenamiento detecta el flag y termina el subproceso.
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task or task["status"] not in ("running", "starting"):
                return False
            task["status"] = "cancelling"
            self._persist_tasks()
            return True

    def get_all_tasks(self) -> list:
        """
        Retorna lista completa de tareas (ordenadas por fecha DESC).
        Lee directamente del disco para incluir tareas de sesiones anteriores.
        """
        tasks = []
        if TASKS_DIR.exists():
            for d in sorted(TASKS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if d.is_dir():
                    p = self._load_progress(d)
                    if p:
                        tasks.append({
                            "task_id": p.get("task_id", d.name),
                            "status": p.get("status", "unknown"),
                            "best_val_accuracy": p.get("best_val_accuracy", 0),
                            "classes_found": p.get("classes_found", []),
                            "started_at": p.get("started_at"),
                            "finished_at": p.get("finished_at"),
                            "message": p.get("message", ""),
                        })
        return tasks

    # ── Ejecución del Entrenamiento ───────────────────────────

    def _run_training(self, task_id: str, task_dir: Path,
                      include_classes: list = None, exclude_classes: list = None):
        """
        Ejecuta entrenar_v2.py como subproceso.
        Corre en un thread daemon — el manager monitorea la salida
        y actualiza el progreso en tiempo real.
        """
        try:
            self._update_status(task_id, task_dir, status="running",
                                message="Preparando dataset...", progress_pct=1)

            # Usa sys.executable para garantizar el mismo entorno virtual
            cmd = [
                sys.executable, str(BASE_DIR / "entrenar_v2.py"),
                "--task-id", task_id,
                "--task-dir", str(task_dir),
            ]
            if include_classes:
                cmd += ["--include-classes", ",".join(include_classes)]

            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Monitoreo de línea por línea del output
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                if self._is_cancelled(task_id):
                    process.terminate()
                    self._update_status(task_id, task_dir, status="cancelled",
                                        message="Cancelado por el usuario", progress_pct=0)
                    self._persist_tasks()
                    return

                # Las líneas PROGRESS: contienen JSON con el estado actual
                if line.startswith("PROGRESS:"):
                    try:
                        data = json.loads(line[9:])
                        self._update_status(task_id, task_dir, **data)
                    except json.JSONDecodeError:
                        pass

            return_code = process.wait()
            if return_code == 0 and not self._is_cancelled(task_id):
                final = self._load_progress(task_dir)
                acc = final.get("best_val_accuracy", 0) if final else 0
                self._update_status(
                    task_id, task_dir,
                    status="completed",
                    message="Entrenamiento completado exitosamente",
                    progress_pct=100,
                    finished_at=datetime.now().isoformat(),
                    best_val_accuracy=acc,
                )
            elif return_code != 0 and not self._is_cancelled(task_id):
                self._update_status(
                    task_id, task_dir,
                    status="error",
                    message=f"Error: proceso terminó con código {return_code}",
                    finished_at=datetime.now().isoformat(),
                )

            self._persist_tasks()

        except Exception as e:
            self._update_status(
                task_id, task_dir,
                status="error",
                message=f"Error: {str(e)}",
                finished_at=datetime.now().isoformat(),
            )
            self._persist_tasks()

    # ── Persistencia ──────────────────────────────────────────

    def _save_progress(self, task_dir: Path, data: dict):
        """Guarda el progreso de una tarea en disco."""
        path = task_dir / PROGRESS_FILE
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_progress(self, task_dir: Path) -> dict:
        """Lee el progreso de una tarea desde disco."""
        path = task_dir / PROGRESS_FILE
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _update_status(self, task_id: str, task_dir: Path, **kwargs):
        """Actualiza el estado de una tarea en memoria y disco."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.update(kwargs)
            else:
                task = kwargs
                task["task_id"] = task_id
                self.tasks[task_id] = task
            self._save_progress(task_dir, task)

    def _is_cancelled(self, task_id: str) -> bool:
        """Verifica si una tarea fue marcada para cancelación."""
        with self.lock:
            return self.tasks.get(task_id, {}).get("status") == "cancelling"

    def _persist_tasks(self):
        """
        Guarda el índice completo de tareas en disco.
        tasks_index.json permite recuperar el estado al reiniciar el servicio.
        """
        path = TASKS_DIR / TASKS_INDEX
        try:
            with open(path, "w") as f:
                json.dump({k: v for k, v in self.tasks.items()}, f)
        except Exception:
            pass

    def _load_tasks(self):
        """
        Carga el índice de tareas desde disco.
        Las tareas que estaban 'running' al momento del reinicio
        se marcan como 'error' para mantener consistencia.
        """
        path = TASKS_DIR / TASKS_INDEX
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for task_id, task in data.items():
                if task.get("status") in ("running", "starting"):
                    task["status"] = "error"
                    task["message"] = "Interrumpido por reinicio del servicio"
                    task["finished_at"] = datetime.now().isoformat()
                self.tasks[task_id] = task
        except Exception:
            pass

    def _cleanup_orphaned(self):
        """
        Busca directorios de tareas en disco que no estén en el índice
        y los incorpora. Esto cubre el caso de migraciones o archivos sueltos.
        """
        if not TASKS_DIR.exists():
            return
        for d in TASKS_DIR.iterdir():
            if d.is_dir() and d.name != TASKS_INDEX.replace(".json", ""):
                p = self._load_progress(d)
                if not p:
                    continue
                tid = p.get("task_id", d.name)
                if tid not in self.tasks:
                    self.tasks[tid] = p

    def _cleanup_old_tasks(self):
        """
        Elimina directorios de tareas completadas que excedan
        MAX_TASKS_KEEP. Solo afecta tareas en estado terminal
        (completed, error, cancelled).
        """
        if not TASKS_DIR.exists():
            return
        dirs = sorted(
            [d for d in TASKS_DIR.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        terminal = {"completed", "error", "cancelled"}
        to_delete = []
        kept_finished = 0
        for d in dirs:
            p = self._load_progress(d)
            if p and p.get("status") in terminal:
                if kept_finished >= MAX_TASKS_KEEP:
                    to_delete.append(d)
                else:
                    kept_finished += 1
        for d in to_delete:
            try:
                shutil.rmtree(d)
                tid = d.name
                if tid in self.tasks:
                    del self.tasks[tid]
            except Exception:
                pass
        self._persist_tasks()


manager = TrainingManager()
