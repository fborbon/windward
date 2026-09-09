# AWS resources — live deployment

Everything below runs on `forwardforecasting-dev` (EC2 `i-06b771ce9bfd7ec87`, `t3.small`, `eu-west-1`), an existing instance also hosting `job-hunter-suite` and reachable over both a public IP and Tailscale. Its idle-shutdown automation (`idle-shutdown.timer`) was disabled specifically so this project stays up persistently — it was previously a personal dev sandbox that auto-stopped after 30 minutes of inactivity.

| Resource | Detail |
|---|---|
| MLflow tracking server | `/home/ubuntu/mlflow-server`, native systemd service (`mlflow.service`), `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://windward-mlflow-artifacts-ff --host 127.0.0.1 --port 5000`. ~180MB RSS. |
| S3 bucket | `windward-mlflow-artifacts-ff` (`eu-west-1`), versioning enabled — MLflow model/artifact storage |
| FastAPI + agent service | `/home/ubuntu/Developments/windward`, Docker (`docker-compose.prod.yml`), `network_mode: host` (so the container can reach the host's MLflow server on `127.0.0.1:5000` — a bridge-networked container would see its own loopback instead and fail with connection refused, a real bug hit during setup), 700MB memory limit |
| nginx + SSL | `windward.forwardforecasting.eu` → `127.0.0.1:8000`, real Let's Encrypt cert via `certbot --nginx` |
| DNS | Route53 `A` record, `windward.forwardforecasting.eu` → the instance's public IP |
| IAM | Inline policy `windward-app-access` on the existing role `forwardforecasting-dev-ssm-role` (attached to the instance profile) — scoped to exactly the S3 bucket, the `windward-agent-sessions` DynamoDB table, and the two Bedrock models used. No static keys anywhere; the app and the MLflow server both pick up credentials from EC2 instance metadata automatically via `boto3`. |
| DynamoDB | `windward-agent-sessions` (`eu-west-1`, on-demand billing) — unchanged from the original setup, was already AWS |

## Reproducing this

```bash
# On the target EC2:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# .env: MLFLOW_TRACKING_URI=http://127.0.0.1:5000, AWS_REGION=eu-west-1, etc. (see .env.example)
.venv/bin/python3 -m forecasting.train      # populates the MLflow registry

sudo docker build -t windward-api:latest -f Dockerfile .
sudo docker compose -f docker-compose.prod.yml up -d

sudo certbot --nginx -d windward.forwardforecasting.eu --non-interactive --agree-tos
```

## Useful commands

```bash
sudo systemctl status mlflow.service                 # tracking server health
sudo docker logs windward_api_prod --tail 50          # app logs
sudo docker compose -f docker-compose.prod.yml restart
curl https://windward.forwardforecasting.eu/health
```
