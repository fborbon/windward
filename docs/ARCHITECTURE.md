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

## Why EDP Wind Farm A isn't in `data_sources.farms.FARMS`

`Farm` (see `data_sources/farms.py`) requires real coordinates (for the Open-Meteo weather join `forecasting/pipeline.py` needs) and real rated power/rotor diameter (for the Cp/Betz-limit and capacity-factor math in `analysis/efficiency.py`). EDP Wind Farm A — real turbine SCADA from the CARE-to-Compare benchmark — discloses none of those; it's anonymized (no coordinates, no rated power, no real calendar timestamps, power channels rescaled to a fraction rather than kW) specifically to protect the source farm's identity. Forcing it into `FARMS` would mean either fabricating a location/capacity (which the whole "real data, no invented numbers" premise of this project rules out) or silently breaking `agents/graph.py`'s forecast-coupled nodes for it.

Instead it's a parallel, standalone feature — `data_sources/edp_scada.py`, `rag/edp_incident_corpus.py` + `rag/edp_retriever.py`, and `/edp/*` API routes — that plays to what the dataset actually has: 22 real, independently labeled fault/normal case studies, genuinely better ground-truth anomaly labels than the other three farms carry. It reuses `analysis.efficiency.binned_power_curve` (which only needs wind speed + power columns, no farm metadata) and the same LangChain/FAISS/Bedrock-Titan RAG pipeline shape as the other farms' corpus, just with its own corpus builder and its own FAISS index, decoupled from `FARMS` and `agents/graph.py` entirely.

## DSWE Inland-Offshore, and why it gets a real forecast-adjacent feature that EDP doesn't

Same `FARMS` exclusion reasoning as EDP applies to the DSWE Inland-Offshore dataset (`data_sources/dswe_scada.py`) — no disclosed coordinates, so no real Open-Meteo weather join is possible. But this dataset has something EDP doesn't: a real on-site meteorological mast, paired with each turbine, plus a real documented calendar date range for that mast's measurement period (from the dataset's own Zenodo description — not derived from the data itself, which has no per-row timestamps at all, only a sequence number, and real gaps: WT1 has 47,542 rows against ~52,704 expected for continuous 10-minute coverage over its documented year, so timestamps can't be reconstructed by assuming even spacing either).

That's enough to run a **Measure-Correlate-Predict (MCP)** ratio (`analysis.efficiency.measure_correlate_predict`) — real on-site mast mean vs. real Open-Meteo ERA5 mean over the same real calendar period, at a location the caller supplies. It's a ratio-of-means, not full regression-based MCP (which needs concurrent timestamp-paired samples this dataset can't provide). Deliberately never assumes a location silently: the dashboard's reference point is editable, defaults to a real but explicitly illustrative coordinate, and the computed ratio — including how far it lands from 1.0 — is presented as evidence of how representative that reference is, not hidden behind a single "predicted" number. This is diagnosis/resource-assessment, not the same claim as the three real farms' Open-Meteo-based forecasting.

## Open questions / decisions deferred

- Real drone/inspection photo feed — currently one real, openly-licensed sample photo, not a live feed.
- Energy-price-prediction integration — planned, not yet built.
- Commercial product surface (auth, billing, a dedicated frontend beyond the current dashboard) — not yet designed.
