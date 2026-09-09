"""Model registry helpers — load the latest registered forecasting model from Azure ML."""
import mlflow

import config


def load_latest_model(name: str):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{name}/latest")
