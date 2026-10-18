"""`python -m infra.slim_image_smoke`, run inside the freshly built slim API image by
infra/deploy_host.sh before the live container is swapped: imports every module the request
handlers reach, including the ones they import lazily inside a function (which a plain
`import api.main` never touches). A dependency missing
from requirements-api.txt then fails the deploy instead of 500-ing on a user's first request."""
import importlib

MODULES = [
    "api.main",
    "agents.graph",
    "agents.qa_agent",
    "agents.llm_router",
    "analysis.efficiency",
    "analysis.vendor.dswe_amk",
    "operational_assessment.payload",
    "rag.edp_retriever",
    "rag.dswe_retriever",
    "rag.pdf_retriever",
    "rag.datasheet_specs",
    "mcp_server.tools",
    "data_sources.farm_facts",
    "data_sources.edp_scada",
    "data_sources.dswe_scada",
    "data_sources.meteo_client",
    "storage.dynamo_session_store",
    "observability.tracing",
    # third-party modules only imported lazily inside functions
    "langfuse.langchain",
    "langchain_community.vectorstores",
    "faiss",
    "litellm",
    "joblib",
    "sklearn.ensemble",
]

for m in MODULES:
    importlib.import_module(m)

# An import check can't see what liteLLM loads lazily inside completion(): on 2026-10-18 a newer
# liteLLM started importing `mcp` on any call with tools=, the slim image lacked it, and every
# /ask tool call fell back to the stuffed prompt. A mocked tool call walks that path offline.
import litellm  # noqa: E402

litellm.completion(
    model="bedrock/eu.amazon.nova-lite-v1:0",
    messages=[{"role": "user", "content": "smoke"}],
    tools=[{"type": "function", "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}}}],
    mock_response="ok",
    aws_region_name="eu-west-1",
)
# .dockerignore excludes dashboard/data/ except listed payloads: on 2026-10-18 a new payload
# wasn't listed, and get_farm_facts answered "no farm facts exported" in production only.
from pathlib import Path  # noqa: E402

DASHBOARD_DATA = Path(__file__).resolve().parent.parent / "dashboard" / "data"
PAYLOADS = ("wind_prediction_payload.json", "wind_speed_payload.json", "operational_payload.json",
            "farm_facts_payload.json", "rag_eval_payload.json")
missing = [p for p in PAYLOADS if not (DASHBOARD_DATA / p).exists()]
assert not missing, f"payloads missing from the image (add them to .dockerignore's ! list): {missing}"
print(f"slim image OK: {len(MODULES)} modules imported, mocked tool call completed, {len(PAYLOADS)} payloads present")
