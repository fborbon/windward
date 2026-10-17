#!/usr/bin/env bash
# Runs ON the EC2 host (via SSM from .github/workflows/deploy.yml), as root, from the repo dir.
# Kept in the repo rather than inline in the workflow so the host-side steps (cron included) are
# versioned with the code they deploy.
set -euo pipefail
cd /home/ubuntu/Developments/windward

docker build -t windward-api:latest -f Dockerfile .
docker run --rm windward-api:latest python -m infra.slim_image_smoke
docker build -t windward-jobs:latest -f Dockerfile.jobs .

# MLflow runs on demand as a container (infra/mlflow/), started by with-mlflow only while a job needs
# it. One-time migration off the old native systemd service; its unit file is kept in
# ~ubuntu/backups/systemd-units/ for rollback. The SQLite registry stays where it was.
docker compose -f infra/mlflow/docker-compose.yml build
install -m 755 infra/mlflow/with-mlflow.sh /usr/local/bin/with-mlflow
if [ -f /etc/systemd/system/mlflow.service ]; then
  systemctl disable --now mlflow.service
  install -d -o ubuntu -g ubuntu -m 700 /home/ubuntu/backups/systemd-units
  install -m 600 -o ubuntu -g ubuntu /etc/systemd/system/mlflow.service /home/ubuntu/backups/systemd-units/
  rm /etc/systemd/system/mlflow.service
  systemctl daemon-reload
fi

# The API reads the forecast model from data/models/ instead of MLflow (forecasting/model_store.py).
# Export before swapping the API container, so a failed export leaves the old one running.
with-mlflow docker run --rm --network host -v "$PWD/data:/app/data" --env-file .env \
  windward-jobs:latest python -m forecasting.export_model

install -m 644 infra/cron/windward /etc/cron.d/windward
# Retire the old host-only ubuntu crontab entry for the same job (replaced by /etc/cron.d above).
(sudo -u ubuntu crontab -l 2>/dev/null | grep -v 'wind_prediction.live_forecast' || true) | sudo -u ubuntu crontab -

docker compose -f docker-compose.prod.yml up -d --force-recreate --remove-orphans
docker image prune -f
