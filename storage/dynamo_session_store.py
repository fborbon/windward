"""Agent conversation/session state in DynamoDB (existing AWS account/credentials).

Deliberate cross-cloud choice: exercises the DynamoDB skill in an otherwise Azure-first
project. Table stays within the AWS always-free tier at this scale (on-demand billing,
single-digit items/session).
"""
from decimal import Decimal

import boto3

import config

_table = None


def get_table():
    global _table
    if _table is None:
        dynamodb = boto3.resource("dynamodb", region_name=config.AWS_REGION)
        _table = dynamodb.Table(config.DYNAMODB_TABLE_SESSIONS)
    return _table


def _floats_to_decimal(obj):
    """DynamoDB's boto3 resource API rejects native float — every float in the item
    needs to be a Decimal instead. Recurses through dicts/lists."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _floats_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_floats_to_decimal(v) for v in obj]
    return obj


def save_session(session_id: str, state: dict):
    get_table().put_item(Item=_floats_to_decimal({"session_id": session_id, **state}))


def load_session(session_id: str) -> dict | None:
    resp = get_table().get_item(Key={"session_id": session_id})
    return resp.get("Item")
