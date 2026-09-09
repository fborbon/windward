# Architecture Notes

## History: originally built on Azure ML, migrated to self-hosted MLflow on AWS

This project started as a deliberate **Azure MLflow** skill demonstration (see `otros/skills/improve_skills.txt`) — Azure ML workspace for MLflow-compatible experiment tracking + model registry, AKS for a real (demo-then-teardown) Kubernetes deployment. That phase worked and is documented in the [blog write-up](https://education.forwardforecasting.eu/windward-agent/): real training runs, real registered models, a real AKS deployment verified end-to-end then torn down.

As of 2026-09-09, the project pivoted toward becoming an actual product (to be combined with an energy-price-prediction service and commercialized), which meant migrating off Azure entirely onto AWS — the account and infrastructure this actually gets operated on. A separate, simpler project will serve as the Azure MLflow demonstration case going forward, so that skill signal isn't lost — it's just no longer this project's job to carry it.

**What changed:** Azure ML workspace → self-hosted MLflow (Docker, on the same EC2 that now runs the service persistently), artifacts in S3 instead of an Azure storage account, AKS → a persistent Docker Compose deployment on that EC2 (nginx + Let's Encrypt, matching every other project on this host) instead of a torn-down demo cluster. Auth changed from `az login`/`DefaultAzureCredential` to the existing AWS credentials already used for Bedrock/DynamoDB. Everything else — LangGraph, the two RAG stacks, the forecasting model, the real SCADA data — is unchanged.

## Why LangGraph over a custom orchestrator

`global-news-agent` (sibling project) hand-rolls its own agent orchestration. Windward deliberately uses LangGraph instead so the state machine (ingest → forecast → diagnose → RAG → recommend → explain) is declarative, checkpointable, and traceable via LangSmith/Langfuse out of the box.

## Why two RAG stacks (LangChain + LlamaIndex)

Two separate corpora, two separate retrieval paths, both exercised deliberately:
- **LangChain + FAISS** over the real turbine fault-event log (chain-based, tool-wrapped for the agent).
- **LlamaIndex** over turbine spec metadata (index-based, structured facts rather than incident narratives).

## Why LiteLLM

The agent layer isn't hard-wired to one LLM provider — LiteLLM gives a single interface so the same agent code could target a different provider later without touching agent code. Bedrock Nova is the only one actually wired up today.

## Why DynamoDB

Session/conversation state for the agent is kept in DynamoDB — a natural fit now that everything runs on AWS, not a cross-cloud demonstration anymore.

## Multimodal scope

`multimodal/blade_inspection.py` takes turbine blade inspection photos and runs a vision-capable LLM pass (Amazon Nova, Bedrock Converse API) to flag visible damage as a structured (Pydantic) result, which feeds into the diagnosis node alongside the numerical anomaly signal.

## Open questions / decisions deferred

- Real drone/inspection photo feed — currently one real, openly-licensed sample photo, not a live feed.
- Energy-price-prediction integration — planned, not yet built.
- Commercial product surface (auth, billing, a dedicated frontend beyond the current dashboard) — not yet designed.
