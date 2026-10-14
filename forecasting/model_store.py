"""Serving-side model access: reads models exported from the MLflow registry to local files.

The API used to load each forecast model through the MLflow registry on every request
(mlflow.pyfunc.load_model re-downloads the artifacts each call, and `import mlflow` alone is
~90 MB resident). forecasting.export_model now copies the latest registered version of each
farm's model to data/models/<name>/ (model.joblib + meta.json) on every deploy and after
training; this module only reads those files, so the API image doesn't need mlflow at all.
"""
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


def model_name(farm_id: str) -> str:
    return f"windward-production-forecast-{farm_id}"


def model_dir(name: str) -> Path:
    return MODELS_DIR / name


@lru_cache(maxsize=8)
def _load(path: str, mtime: float):
    import joblib

    return joblib.load(path)


def load_latest_model(name: str):
    path = model_dir(name) / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"no exported model at {path} - run `python -m forecasting.export_model` (needs MLflow)"
        )
    # mtime in the cache key: a re-export replaces the file and the next call picks it up.
    return _load(str(path), path.stat().st_mtime)


def latest_model_info(name: str) -> dict:
    """Version + age of the exported (= latest registered) model — a real "is this model stale"
    signal for agents.graph.investigate_node."""
    meta_path = model_dir(name) / "meta.json"
    if not meta_path.exists():
        return {"version": None, "created_at": None, "age_days": None}
    meta = json.loads(meta_path.read_text())
    created = datetime.fromisoformat(meta["created_at"])
    age_days = (datetime.now(tz=timezone.utc) - created).days
    return {"version": meta["version"], "created_at": meta["created_at"], "age_days": age_days}
