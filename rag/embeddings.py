"""Minimal LangChain Embeddings wrapper around Bedrock Titan Text Embeddings v2 — invocable
on-demand (unlike the Nova chat models, no inference profile needed), reusing the same AWS
credentials as the rest of the project."""
import json

import boto3
from langchain_core.embeddings import Embeddings

import config

MODEL_ID = "amazon.titan-embed-text-v2:0"


class BedrockTitanEmbeddings(Embeddings):
    def __init__(self, region_name: str = None):
        self._client = boto3.client("bedrock-runtime", region_name=region_name or config.BEDROCK_REGION)

    def _embed_one(self, text: str) -> list[float]:
        resp = self._client.invoke_model(modelId=MODEL_ID, body=json.dumps({"inputText": text[:8000]}))
        return json.loads(resp["body"].read())["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)
