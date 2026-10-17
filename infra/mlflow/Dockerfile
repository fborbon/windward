# Self-hosted MLflow tracking server + model registry for windward and energy-trader.
# Runs on demand only (infra/mlflow/with-mlflow.sh): nothing on the request path of either live
# service needs it, since both APIs serve models exported to files (forecasting/export_model.py).
FROM python:3.12-slim
# Exact versions from the native install it replaced (pip freeze): unpinned, SQLAlchemy 2.1 gets
# pulled in and mlflow 2.22 fails at startup (FallbackAsyncAdaptedQueuePool was removed).
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
ENV HOME=/tmp
EXPOSE 5000
