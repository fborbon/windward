"""Copy the latest registered version of each farm's forecast model out of the MLflow registry
into data/models/<name>/ (model.joblib + meta.json), where the API reads it via
forecasting.model_store without needing mlflow. Runs in the full `windward-jobs` image on every
deploy (see .github/workflows/deploy.yml); run it by hand after retraining too.

    python -m forecasting.export_model
"""
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import config
from data_sources.farms import FARMS
from forecasting.model_store import model_dir, model_name


def export_one(client: MlflowClient, name: str) -> dict | None:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    out = model_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("version") == latest.version:
        return {"name": name, "version": latest.version, "skipped": True}

    model = mlflow.sklearn.load_model(f"models:/{name}/{latest.version}")
    tmp = out / "model.joblib.tmp"
    joblib.dump(model, tmp)
    tmp.replace(out / "model.joblib")
    created = datetime.fromtimestamp(latest.creation_timestamp / 1000, tz=timezone.utc)
    meta_path.write_text(json.dumps({"version": latest.version, "created_at": created.isoformat()}))
    return {"name": name, "version": latest.version, "skipped": False}


def main() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    for farm_id in FARMS:
        print(export_one(client, model_name(farm_id)) or f"{model_name(farm_id)}: not registered")


if __name__ == "__main__":
    main()
