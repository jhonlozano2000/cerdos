import os
import json
import time
import uuid
import subprocess
import threading
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
TASKS_DIR = BASE_DIR / "training_tasks"
PROGRESS_FILE = "progress.json"

class TrainingManager:
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()
        TASKS_DIR.mkdir(exist_ok=True)

    def start_training(self, include_classes=None, exclude_classes=None) -> str:
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

            thread = threading.Thread(
                target=self._run_training,
                args=(task_id, task_dir, include_classes, exclude_classes),
                daemon=True,
            )
            thread.start()

            return task_id

    def get_status(self, task_id: str) -> dict:
        with self.lock:
            if task_id in self.tasks:
                return self.tasks[task_id]

            task_dir = TASKS_DIR / task_id
            if task_dir.exists():
                return self._load_progress(task_dir) or {"status": "not_found"}

            return {"status": "not_found"}

    def cancel_training(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task or task["status"] not in ("running", "starting"):
                return False
            task["status"] = "cancelling"
            return True

    def get_all_tasks(self) -> list:
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

    def _run_training(self, task_id: str, task_dir: Path, include_classes: list = None, exclude_classes: list = None):
        try:
            self._update_status(task_id, task_dir, status="running", message="Preparando dataset...", progress_pct=1)

            cmd = [
                "python", str(BASE_DIR / "entrenar_v2.py"),
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

            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                if self._is_cancelled(task_id):
                    process.terminate()
                    self._update_status(task_id, task_dir, status="cancelled", message="Cancelado por el usuario", progress_pct=0)
                    return

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
        except Exception as e:
            self._update_status(
                task_id, task_dir,
                status="error",
                message=f"Error: {str(e)}",
                finished_at=datetime.now().isoformat(),
            )

    def _save_progress(self, task_dir: Path, data: dict):
        path = task_dir / PROGRESS_FILE
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_progress(self, task_dir: Path) -> dict:
        path = task_dir / PROGRESS_FILE
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _update_status(self, task_id: str, task_dir: Path, **kwargs):
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
        with self.lock:
            return self.tasks.get(task_id, {}).get("status") == "cancelling"

manager = TrainingManager()
