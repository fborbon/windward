"""Builds the RAG Evaluation tab's payload (README §17) from the question battery, the graded
runs in rag_eval/runs/ and the written review:

    python -m rag_eval.run --label baseline       # set 1 before its fixes (live deployment)
    python -m rag_eval.run --label set2-baseline --ids q16,...,q35   # set 2, before its own fixes
    python -m rag_eval.run --label final          # all 35 after every fix deployed
    python -m rag_eval.export                     # writes dashboard/data/rag_eval_payload.json

Set 2's "before" already includes set 1's fixes: it measures how they held up on new questions.
"""
import json
from pathlib import Path

from rag_eval.questions import QUESTIONS
from rag_eval.review import REVIEW
from rag_eval.run import RUNS_DIR

EXPORT_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "rag_eval_payload.json"

SETS = {
    1: {"name": "Set 1", "ids": range(1, 16), "baseline_run": "baseline",
        "about": "The first 15 questions, one per capability. 'Before' is the agent as it was."},
    2: {"name": "Set 2", "ids": range(16, 36), "baseline_run": "set2-baseline",
        "about": "20 new questions added afterwards, including traps (a turbine that doesn't exist, a result "
                 "never computed). 'Before' already has set 1's fixes, so it shows how they held up."},
}


def _set_of(qid: str) -> int:
    n = int(qid[1:])
    return next(k for k, v in SETS.items() if n in v["ids"])


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
    final = _load("final")
    by_id = lambda run: {r["id"]: r for r in (run or {}).get("results", [])}  # noqa: E731
    base = {}
    for meta in SETS.values():
        base.update(by_id(_load(meta["baseline_run"])))
    fin = by_id(final)
    questions = []
    for q in QUESTIONS:
        review = REVIEW.get(q["id"], {})
        questions.append({
            **{k: q[k] for k in ("id", "farm", "capability", "question", "truth", "sources")},
            "set": _set_of(q["id"]),
            "baseline": {**{k: base.get(q["id"], {}).get(k) for k in ("answer", "sources", "pass")}, "verdict": review.get("baseline")},
            "final": {**{k: fin.get(q["id"], {}).get(k) for k in ("answer", "sources", "pass", "latency_s")}, "verdict": review.get("final")},
            "diagnosis": review.get("diagnosis"),
            "fix": review.get("fix"),
        })

    def tally(key: str, set_id: int | None = None) -> dict:
        out: dict[str, int] = {}
        for q in questions:
            if set_id is not None and q["set"] != set_id:
                continue
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
        "final_run_at": (final or {}).get("run_at"),
        "sets": [{
            "set": k, "name": v["name"], "about": v["about"], "size": sum(1 for q in questions if q["set"] == k),
            "baseline_run_at": (_load(v["baseline_run"]) or {}).get("run_at"),
            "verdicts": {"baseline": tally("baseline", k), "final": tally("final", k)},
            "auto_pass": {"baseline": sum(1 for q in questions if q["set"] == k and q["baseline"]["pass"]),
                          "final": sum(1 for q in questions if q["set"] == k and q["final"]["pass"])},
        } for k, v in SETS.items()],
        "verdicts": {"baseline": tally("baseline"), "final": tally("final")},
        "model_comparison": comparison,
        "questions": questions,
    }


if __name__ == "__main__":
    EXPORT_PATH.write_text(json.dumps(build(), indent=1, ensure_ascii=False))
    print(f"wrote {EXPORT_PATH}")
