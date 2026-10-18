"""The API runs from a slim image (requirements-api.txt) without torch, chronos, statsmodels,
mlflow or llama-index. Guard against a request-path import quietly pulling one back in: it
would crash the slim image at request time (ImportError), not at build time."""
import subprocess
import sys

HEAVY = ("torch", "chronos", "statsmodels", "mlflow", "llama_index", "dswe", "pymupdf", "fitz")
API_MODULES = (
    "api.main",
    "agents.graph",
    "agents.qa_agent",
    "agents.llm_router",
    "analysis.efficiency",
    "operational_assessment.payload",
    "rag.edp_retriever",
    "rag.dswe_retriever",
    "rag.pdf_retriever",
    "rag.datasheet_specs",
    "data_sources.edp_scada",
    "data_sources.dswe_scada",
    "storage.dynamo_session_store",
)


def test_request_path_imports_stay_slim():
    code = (
        "import importlib, sys\n"
        f"for m in {API_MODULES!r}: importlib.import_module(m)\n"
        "from analysis.efficiency import smooth_power_curve\n"
        "from analysis.vendor.dswe_amk import AMK\n"
        f"print(','.join(h for h in {HEAVY!r} if h in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
