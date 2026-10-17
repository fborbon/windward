#!/usr/bin/env bash
# with-mlflow <command...>: run a command with the MLflow server up, then put it back the way it was.
#
# MLflow is only needed while training, backtesting or exporting a model, so it is not kept
# running (it used to be a native systemd service holding ~180 MB around the clock). This starts
# the container, waits until it answers, runs the command and stops the container again, unless it
# was already running before (someone browsing the UI through an SSH tunnel), in which case it is
# left alone. A lock serialises concurrent callers so one cannot stop the server under another.
# Installed as /usr/local/bin/with-mlflow by infra/deploy_host.sh.
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: with-mlflow <command...>" >&2; exit 2; }

COMPOSE_FILE=/home/ubuntu/Developments/windward/infra/mlflow/docker-compose.yml
LOCK=/tmp/with-mlflow.lock

exec 9>>"$LOCK"
chmod 666 "$LOCK" 2>/dev/null || true
flock 9

was_running=$(docker inspect -f '{{.State.Running}}' mlflow 2>/dev/null || echo false)
if [ "$was_running" != true ]; then
  docker compose -f "$COMPOSE_FILE" up -d --no-build mlflow >/dev/null 2>&1 \
    || docker compose -f "$COMPOSE_FILE" up -d --build mlflow >/dev/null
fi
stop() { [ "$was_running" = true ] || docker compose -f "$COMPOSE_FILE" stop mlflow >/dev/null 2>&1; }
trap stop EXIT

for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:5000/health >/dev/null && break
  sleep 2
done
curl -sf http://127.0.0.1:5000/health >/dev/null || { echo "with-mlflow: server did not come up" >&2; docker logs --tail 20 mlflow >&2; exit 1; }

"$@"
