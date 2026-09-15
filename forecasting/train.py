"""Baseline forecasting model, tracked via Azure ML's MLflow-compatible server. One model
per farm, registered as f"windward-production-forecast-{farm_id}".

Auth is via `az login` + DefaultAzureCredential (see config.MLFLOW_TRACKING_URI) — no key required.
Training runs locally; only metrics/artifacts/model are pushed to Azure ML, so there is no
Azure compute cost from running this.
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split

import config
from forecasting.features import FEATURE_COLUMNS, TARGET_COLUMN

KFOLD_SPLITS = 5


def model_name(farm_id: str) -> str:
    return f"windward-production-forecast-{farm_id}"


def set_tracking():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("windward-production-forecast")


def train(farm_id: str, df: pd.DataFrame, register: bool = True):
    """df: output of forecasting.pipeline.build_training_frame."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    set_tracking()
    with mlflow.start_run(run_name=farm_id) as run:
        params = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        mlflow.log_param("farm_id", farm_id)
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_params(params)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))

        model = GradientBoostingRegressor(**params)

        # K-fold CV on the training split, in addition to the single held-out test split
        # below — a lower-variance read on generalization (a single 80/20 split can land
        # lucky or unlucky) than either replaces, since the final registered model is still
        # the one fit on X_train/y_train exactly as before, not a fold-averaged model.
        kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
        fold_train_r2, fold_test_r2 = [], []
        for fold_train_idx, fold_test_idx in kf.split(X_train):
            fold_model = clone(model)
            fold_model.fit(X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx])
            fold_train_r2.append(r2_score(y_train.iloc[fold_train_idx], fold_model.predict(X_train.iloc[fold_train_idx])))
            fold_test_r2.append(r2_score(y_train.iloc[fold_test_idx], fold_model.predict(X_train.iloc[fold_test_idx])))
        mlflow.log_metric("kfold_mean_train_r2", float(np.mean(fold_train_r2)))
        mlflow.log_metric("kfold_mean_test_r2", float(np.mean(fold_test_r2)))
        mlflow.log_metric("kfold_std_test_r2", float(np.std(fold_test_r2)))

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mlflow.log_metric("mae_mw", mae)
        mlflow.log_metric("r2", r2)

        signature = mlflow.models.infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name(farm_id) if register else None,
        )

        print(
            f"[{farm_id}] run_id={run.info.run_id} mae_mw={mae:.3f} r2={r2:.3f} "
            f"kfold_test_r2={np.mean(fold_test_r2):.3f}+/-{np.std(fold_test_r2):.3f}"
        )
        return model, {
            "mae_mw": mae, "r2": r2, "run_id": run.info.run_id,
            "kfold_mean_test_r2": float(np.mean(fold_test_r2)), "kfold_std_test_r2": float(np.std(fold_test_r2)),
        }


if __name__ == "__main__":
    from data_sources.farms import FARMS
    from forecasting.pipeline import build_training_frame

    for farm in FARMS.values():
        frame = build_training_frame(farm)
        train(farm.farm_id, frame)
