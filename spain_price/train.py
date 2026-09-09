"""Spain day-ahead price forecasting model — same self-hosted MLflow server as the wind
farm models, registered as `spain-price-forecast`.
"""
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import config
from spain_price.features import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_NAME = "spain-price-forecast"


def set_tracking():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("spain-price-forecast")


def train(df, register: bool = True):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    set_tracking()
    with mlflow.start_run() as run:
        params = dict(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_params(params)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mlflow.log_metric("mae_eur_mwh", mae)
        mlflow.log_metric("r2", r2)

        signature = mlflow.models.infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME if register else None,
        )

        print(f"run_id={run.info.run_id} mae_eur_mwh={mae:.2f} r2={r2:.3f}")
        return model, {"mae_eur_mwh": mae, "r2": r2, "run_id": run.info.run_id}


if __name__ == "__main__":
    from spain_price.pipeline import build_training_frame

    frame = build_training_frame()
    train(frame)
