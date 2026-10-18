"""Builds the RAG Evaluation tab's payload (README §17) from the question battery, the graded
runs in rag_eval/runs/ and the written review:

    python -m rag_eval.run --label baseline     # before the changes (live deployment)
    python -m rag_eval.run --label final        # after they deployed
    python -m rag_eval.export                   # writes dashboard/data/rag_eval_payload.json
"""
import json
from pathlib import Path

from rag_eval.questions import QUESTIONS
from rag_eval.review import REVIEW
from rag_eval.run import RUNS_DIR

EXPORT_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "rag_eval_payload.json"

# Same battery, same code, answered by two Bedrock models (rag_eval.run --local --model ...),
# at the stage before the citation and parallel-call fixes.
MODEL_COMPARISON = [
    ("compare-nova-lite", "Amazon Nova Lite", "bedrock/eu.amazon.nova-lite-v1:0"),
    ("compare-nova-2-lite", "Amazon Nova 2 Lite", "bedrock/eu.amazon.nova-2-lite-v1:0"),
]


def _load(label: str) -> dict | None:
    path = RUNS_DIR / f"{label}.json"
    return json.loads(path.read_text()) if path.exists() else None


def build() -> dict:
    baseline, final = _load("baseline"), _load("final")
    by_id = lambda run: {r["id"]: r for r in (run or {}).get("results", [])}  # noqa: E731
    base, fin = by_id(baseline), by_id(final)
    questions = []
    for q in QUESTIONS:
        review = REVIEW.get(q["id"], {})
        questions.append({
            **{k: q[k] for k in ("id", "farm", "capability", "question", "truth", "sources")},
            "baseline": {**{k: base.get(q["id"], {}).get(k) for k in ("answer", "sources", "pass")}, "verdict": review.get("baseline")},
            "final": {**{k: fin.get(q["id"], {}).get(k) for k in ("answer", "sources", "pass", "latency_s")}, "verdict": review.get("final")},
            "diagnosis": review.get("diagnosis"),
            "fix": review.get("fix"),
        })

    def tally(key: str) -> dict:
        out: dict[str, int] = {}
        for q in questions:
            v = q[key]["verdict"]
            if v:
                out[v] = out.get(v, 0) + 1
        return out

    comparison = []
    for label, name, model_id in MODEL_COMPARISON:
        run = _load(label)
        if run:
            comparison.append({"model": name, "model_id": model_id,
                               "passed": sum(r["pass"] for r in run["results"]), "total": len(run["results"])})
    return {
        "answer_model": {"name": "Amazon Nova Lite", "id": "eu.amazon.nova-lite-v1:0", "via": "LiteLLM on Amazon Bedrock (eu-west-1)",
                         "embeddings": "Amazon Titan Text Embeddings v2"},
        "baseline_run_at": (baseline or {}).get("run_at"),
        "final_run_at": (final or {}).get("run_at"),
        "verdicts": {"baseline": tally("baseline"), "final": tally("final")},
        "auto_pass": {"baseline": sum(1 for q in questions if q["baseline"]["pass"]),
                      "final": sum(1 for q in questions if q["final"]["pass"])},
        "model_comparison": comparison,
        "questions": questions,
    }


if __name__ == "__main__":
    EXPORT_PATH.write_text(json.dumps(build(), indent=1, ensure_ascii=False))
    print(f"wrote {EXPORT_PATH}")
