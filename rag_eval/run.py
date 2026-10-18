"""Runs the question battery (rag_eval/questions.py) against a Windward deployment's /ask
endpoint and grades each answer automatically: did the agent use a tool that can answer it,
and does the answer contain the reference facts. The grades are a first pass; the written
diagnosis of each answer (what went wrong and which fix addressed it) is kept in
rag_eval/review.py and merged into the payload by rag_eval.export.

    python -m rag_eval.run --label baseline   # writes rag_eval/runs/baseline.json
    python -m rag_eval.run --label dev --local

--local answers with this checkout's agent code (agents/qa_agent.py, local PDF index and
payloads), fed the deployment's own /analysis payload, so a change is evaluated before it ships.
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from rag_eval.questions import QUESTIONS

RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_URL = "https://windward.forwardforecasting.eu"


PAUSE_S = 8  # the live /ask is rate-limited per client (nginx); 15 back-to-back questions hit it


def ask(base_url: str, farm: str, question: str) -> dict:
    req = urllib.request.Request(
        f"{base_url}/analysis/{farm}/ask",
        data=json.dumps({"question": question}).encode(),
        headers={"content-type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 3:
                raise
            time.sleep(30 * (attempt + 1))


_analysis: dict[str, dict] = {}


def ask_local(base_url: str, farm: str, question: str) -> dict:
    from agents.qa_agent import answer_question

    if farm not in _analysis:
        with urllib.request.urlopen(f"{base_url}/analysis/{farm}", timeout=300) as resp:
            _analysis[farm] = json.loads(resp.read())
    return answer_question(farm, question, _analysis[farm])


def grade(q: dict, answer: str, sources: list[str]) -> dict:
    facts = [bool(re.search(p, answer or "", re.IGNORECASE)) for p in q["expect"]]
    tool_ok = not q["sources"] or any(s in sources for s in q["sources"])
    forbidden = [p for p in q.get("forbid", []) if re.search(p, answer or "", re.IGNORECASE)]
    return {
        "tool_ok": tool_ok,
        "facts_found": sum(facts),
        "facts_total": len(facts),
        "missing_facts": [p for p, ok in zip(q["expect"], facts) if not ok],
        "forbidden_found": forbidden,
        "pass": tool_ok and all(facts) and not forbidden,
    }


def run(base_url: str, local: bool = False, ids: set[str] | None = None) -> list[dict]:
    results = []
    for q in QUESTIONS:
        if ids and q["id"] not in ids:
            continue
        if not local and results:
            time.sleep(PAUSE_S)
        start = time.time()
        try:
            out = (ask_local if local else ask)(base_url, q["farm"], q["question"])
        except Exception as e:  # one failed call shouldn't lose the rest of the run
            out = {"answer": f"(request failed: {e})", "sources": []}
        answer, sources = out.get("answer") or "", out.get("sources") or []
        results.append({
            "id": q["id"], "answer": answer.strip(), "sources": sources,
            "latency_s": round(time.time() - start, 1), **grade(q, answer, sources),
        })
        print(f"{q['id']} {'PASS' if results[-1]['pass'] else 'FAIL'} {sources} {results[-1]['latency_s']}s", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--local", action="store_true", help="answer with this checkout's code")
    parser.add_argument("--merge", action="store_true", help="with --ids: replace only those results in an existing run")
    parser.add_argument("--ids", help="comma-separated question ids to run (default: all)")
    parser.add_argument("--model", help="--local only: LiteLLM model id to answer with instead of the configured one")
    args = parser.parse_args()
    if args.model:
        from agents import llm_router

        llm_router.MODEL_MAP[llm_router.config.LLM_PROVIDER] = args.model
    results = run(args.url.rstrip("/"), args.local, set(args.ids.split(",")) if args.ids else None)
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{args.label}.json"
    if args.merge and path.exists():  # re-run some questions, keep the rest of the run
        rerun = {r["id"] for r in results}
        results = [r for r in json.loads(path.read_text())["results"] if r["id"] not in rerun] + results
        order = [q["id"] for q in QUESTIONS]
        results.sort(key=lambda r: order.index(r["id"]))
    out = {"label": args.label, "url": args.url, "local": args.local, "model": args.model, "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": results}
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{sum(r['pass'] for r in results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
