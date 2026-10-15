# Slim API image: only what the request handlers import (requirements-api.txt). No torch,
# chronos, statsmodels, mlflow or llama-index - those live in Dockerfile.jobs, built on top of
# this one, for the daily live forecast, model export and offline work.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 8020
# One gunicorn-managed uvicorn worker, recycled every ~100 requests: the LangGraph/LiteLLM/FAISS
# agent stack (~150-250 MB) loads on the first agent request and Python never hands that memory
# back, while real traffic is a few requests a day. A fresh worker starts back at the ~190 MB
# baseline; requests arriving during the ~10s restart wait in the listen backlog, not dropped.
CMD ["gunicorn", "api.main:app", "-k", "uvicorn_worker.UvicornWorker", "-w", "1", \
     "-b", "127.0.0.1:8020", "--timeout", "300", \
     "--max-requests", "100", "--max-requests-jitter", "20"]
