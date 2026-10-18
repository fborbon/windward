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
print(f"slim image OK: {len(MODULES)} modules imported")
