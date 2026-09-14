"""Model registry helpers — load the latest registered forecasting model from Azure ML."""
from datetime import datetime, timezone

import mlflow

import config


def load_latest_model(name: str):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{name}/latest")


def latest_model_info(name: str) -> dict:
    """Version + age of the latest registered model — a real "is this model stale" signal
    for agents.graph.investigate_node, not derived from load_latest_model() itself since
    the pyfunc wrapper doesn't carry registry timestamps."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return {"version": None, "created_at": None, "age_days": None}

    latest = max(versions, key=lambda v: int(v.version))
    created = datetime.fromtimestamp(latest.creation_timestamp / 1000, tz=timezone.utc)
    age_days = (datetime.now(tz=timezone.utc) - created).days
    return {"version": latest.version, "created_at": created.isoformat(), "age_days": age_days}
