"""Central config, loaded from environment / .env."""
import os
from dotenv import load_dotenv

load_dotenv()

# MLflow — self-hosted on AWS (see infra/aws/README.md). No Azure ML SDK, no static key:
# the tracking server sits on the same private network as anything that calls it.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_S3_BUCKET = os.getenv("MLFLOW_S3_BUCKET", "windward-mlflow-artifacts-ff")

# LLM routing (LiteLLM)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "eu-west-1")

# Data sources
OPEN_METEO_BASE_URL = os.getenv("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1")
ENTSOE_API_TOKEN = os.getenv("ENTSOE_API_TOKEN")

# Observability
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# AWS / DynamoDB
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
DYNAMODB_TABLE_SESSIONS = os.getenv("DYNAMODB_TABLE_SESSIONS", "windward-agent-sessions")
