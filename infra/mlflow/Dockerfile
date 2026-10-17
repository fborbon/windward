# Self-hosted MLflow tracking server + model registry for windward and energy-trader.
# Runs on demand only (infra/mlflow/with-mlflow.sh): nothing on the request path of either live
# service needs it, since both APIs serve models exported to files (forecasting/export_model.py).
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==2.22.5 boto3
ENV HOME=/tmp
EXPOSE 5000
