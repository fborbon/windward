#!/usr/bin/env bash
# Runs ON the EC2 host (via SSM from .github/workflows/deploy.yml), as root, from the repo dir.
# Kept in the repo rather than inline in the workflow so the host-side steps (cron included) are
# versioned with the code they deploy.
set -euo pipefail
cd /home/ubuntu/Developments/windward

docker build -t windward-api:latest -f Dockerfile .
docker run --rm windward-api:latest python -m infra.slim_image_smoke
docker build -t windward-jobs:latest -f Dockerfile.jobs .

# The API reads the forecast model from data/models/ instead of MLflow (forecasting/model_store.py).
# Export before swapping the API container, so a failed export leaves the old one running.
docker run --rm --network host -v "$PWD/data:/app/data" --env-file .env \
  windward-jobs:latest python -m forecasting.export_model

install -m 644 infra/cron/windward /etc/cron.d/windward
# Retire the old host-only ubuntu crontab entry for the same job (replaced by /etc/cron.d above).
(sudo -u ubuntu crontab -l 2>/dev/null | grep -v 'wind_prediction.live_forecast' || true) | sudo -u ubuntu crontab -

docker compose -f docker-compose.prod.yml up -d --force-recreate --remove-orphans
docker image prune -f
