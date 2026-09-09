"""Vision-LLM pass over turbine blade inspection photos -> structured damage assessment.

Uses Amazon Nova Lite's multimodal input directly via the Bedrock Converse API (boto3) —
LiteLLM's Bedrock image support is inconsistent across Nova versions, and Converse's image
block is the officially documented path for Nova vision input.
"""
import json
from pathlib import Path

import boto3

import config
from schemas.models import BladeInspectionResult

MODEL_ID = "eu.amazon.nova-lite-v1:0"  # cross-region inference profile — same requirement as agents/llm_router.py

PROMPT = """You are a wind turbine blade inspection assistant. Look at this photo and assess it for visible \
blade damage (leading-edge erosion, cracks, delamination, lightning strike marks, icing).

Reply with ONLY a JSON object, no other text:
{"damage_detected": bool, "damage_types": [string], "confidence": float (0-1), "description": string (1-2 sentences)}

If the photo doesn't show damage, or shows something other than a blade close-up (e.g. a wide inspection shot), \
say so plainly in "description" and set damage_detected to false."""


def inspect_image(turbine_id: str, image_path: str) -> BladeInspectionResult:
    client = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)
    image_bytes = Path(image_path).read_bytes()
    fmt = Path(image_path).suffix.lstrip(".").lower().replace("jpg", "jpeg")

    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
                    {"text": PROMPT},
                ],
            }
        ],
    )
    text = response["output"]["message"]["content"][0]["text"]
    # Nova sometimes wraps JSON in a markdown fence despite the instruction — strip it.
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(text)

    return BladeInspectionResult(
        turbine_id=turbine_id,
        image_ref=image_path,
        damage_detected=parsed["damage_detected"],
        damage_types=parsed.get("damage_types", []),
        confidence=parsed["confidence"],
        description=parsed.get("description", ""),
    )
