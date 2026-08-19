"""
Guardrails evaluation — runs the 6 golden_dataset.json guardrail_tests
against the live backend and classifies each as TP/TN/FP/FN, matching
DOCS/11_EVALS_PIPELINE.md's design. Deliberately separate from Phase 2's
RAGAS metrics: this measures the guardrail *gate*, not the RAG pipeline
behind it, and needs none of RAGAS's judge-LLM machinery.
"""

import os
import time
import copy

from evals.pipeline import query_backend, BACKEND_URL, INTER_CALL_DELAY_SECONDS


def _was_blocked(result: dict) -> bool:
    """A rail fired iff the thought_process trail says so — mirrors the
    exact string app/guardrails/rails.py's guard() detection relies on
    downstream in main.py's response shape."""
    thought_process = result.get("thought_process", []) or []
    return any("Intent: Guardrails Fired" in step for step in thought_process)


def _classify(expected_blocked: bool, actually_blocked: bool) -> str:
    if expected_blocked and actually_blocked:
        return "TP"
    if not expected_blocked and not actually_blocked:
        return "TN"
    if not expected_blocked and actually_blocked:
        return "FP"
    return "FN"  # expected_blocked and not actually_blocked


def run_guardrails_eval(
    guardrail_tests: list[dict],
    backend_url: str = BACKEND_URL,
    progress_callback=None,
) -> list[dict]:
    """
    Runs every test case through the live backend and returns each one
    annotated with `actually_blocked` and `classification` (TP/TN/FP/FN).
    Does not mutate the input list.
    """
    results = []
    total = len(guardrail_tests)

    for i, test in enumerate(guardrail_tests, start=1):
        annotated = copy.deepcopy(test)
        thread_id = f"eval-guardrail-{test['id']}"
        try:
            result = query_backend(test["input"], thread_id, backend_url)
            actually_blocked = _was_blocked(result)
            annotated["actually_blocked"] = actually_blocked
            annotated["classification"] = _classify(test["expected_blocked"], actually_blocked)
        except Exception as e:
            annotated["actually_blocked"] = None
            annotated["classification"] = "ERROR"
            annotated["_error"] = str(e)

        results.append(annotated)

        if progress_callback:
            progress_callback(i, total, annotated)

        if i < total:
            time.sleep(INTER_CALL_DELAY_SECONDS)

    return results


def compute_guardrail_metrics(results: list[dict]) -> dict:
    """
    Precision/recall/accuracy for the guardrail layer, treating "blocked"
    as the positive class — precision answers "of what we blocked, how
    much actually deserved it," recall answers "of what deserved blocking,
    how much did we actually catch."
    """
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "ERROR": 0}
    for r in results:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1

    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    scored_total = tp + tn + fp + fn  # excludes ERROR — a transport failure isn't a guardrail miss

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    accuracy = (tp + tn) / scored_total if scored_total > 0 else None

    return {
        "counts": counts,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    import json

    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(dataset_path) as f:
        golden = json.load(f)

    def _print_progress(i, total, annotated):
        print(f"[{i}/{total}] id={annotated['id']} type={annotated['type']} "
              f"expected_blocked={annotated['expected_blocked']} "
              f"actually_blocked={annotated['actually_blocked']} -> {annotated['classification']}")

    print(f"Running guardrails eval against {BACKEND_URL} — "
          f"{len(golden['guardrail_tests'])} test cases...")
    eval_results = run_guardrails_eval(golden["guardrail_tests"], progress_callback=_print_progress)
    metrics = compute_guardrail_metrics(eval_results)

    print()
    print("Counts:", metrics["counts"])
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"Accuracy:  {metrics['accuracy']}")
