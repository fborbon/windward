"""LiteLLM-based provider router — Bedrock Nova.

Auth: existing AWS credentials/profile (same pattern as other projects, e.g. global-news-agent).
LiteLLM keeps this swappable to another provider later without touching agent code.
"""
import litellm

import config

MODEL_MAP = {
    # eu. cross-region inference profile — Nova doesn't support on-demand invocation by
    # bare model ID in eu-west-1, only via a region-prefixed inference profile.
    "bedrock": "bedrock/eu.amazon.nova-lite-v1:0",
}


def complete(messages: list[dict], **kwargs):
    model = MODEL_MAP.get(config.LLM_PROVIDER)
    if not model:
        raise RuntimeError(f"No model configured for LLM_PROVIDER={config.LLM_PROVIDER}")
    return litellm.completion(model=model, messages=messages, aws_region_name=config.BEDROCK_REGION, **kwargs)
