"""
Phase 1 — live response collection.

Sends each golden question through the real running FastAPI backend
(POST /query) and captures what the system actually said, actually
retrieved, and which tool the planner actually routed to. This is the
enrichment step that turns golden_dataset.json's static questions into
real (question, response, context) triples that Phase 2 (metrics.py) can
score with RAGAS.
"""

import os
import time
import copy

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# RAGAS judges the response against the retrieved context, not against
# response length — 300 chars is enough signal for faithfulness/relevancy
# scoring. Passing the full response would roughly double token cost in
# Phase 2 for no accuracy gain.
RESPONSE_TRUNCATE_CHARS = 300

# Groq RPM buffer on the main GROQ_API_KEY — this pipeline shares that key
# with live production traffic, so 15 questions run back-to-back with no
# spacing risks a 429 mid-run.
INTER_CALL_DELAY_SECONDS = 10


def _detect_tool(thought_process: list[str]) -> str:
    """
    Map the planner's thought_process trail to a tool name for Tool
    Correctness scoring (Jaccard set overlap against expected_tools).
    """
    joined = " ".join(thought_process or [])
    if "Intent: Guardrails Fired" in joined:
        return "guardrails"
    if "Intent: Technical" in joined:
        return "retrieve_documents"
    if "Intent: Conversational" in joined:
        return "direct_answer"
    return "unknown"


def query_backend(question: str, thread_id: str, backend_url: str = BACKEND_URL, timeout: int = 60) -> dict:
    """Single POST /query call. Raises on transport failure — the caller
    decides how to handle a failed sample rather than this silently
    returning empty data."""
    response = requests.post(
        f"{backend_url}/query",
        json={"q": question, "thread_id": thread_id},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def run_live_pipeline(
    golden_data: dict,
    backend_url: str = BACKEND_URL,
    progress_callback=None,
) -> dict:
    """
    Phase 1 — enrich every rag_sample in golden_data with a real response
    from the live backend. Returns a deep copy; the input dict (and the
    file it may have been loaded from) is never mutated in place.

    progress_callback(index, total, sample) is called after each sample,
    if provided — lets a Streamlit UI show live progress without this
    module knowing anything about Streamlit.
    """
    enriched = copy.deepcopy(golden_data)
    samples = enriched["rag_samples"]
    total = len(samples)

    for i, sample in enumerate(samples, start=1):
        thread_id = f"eval-{sample['id']}"
        try:
            result = query_backend(sample["question"], thread_id, backend_url)
            full_response = result.get("answer", "") or ""
            sample["actual_response"] = full_response[:RESPONSE_TRUNCATE_CHARS]
            sample["actual_contexts"] = result.get("sources", [])
            sample["actual_tools_called"] = [_detect_tool(result.get("thought_process", []))]
        except Exception as e:
            # A single failed sample shouldn't abort the whole run — leave
            # it visibly empty (Phase 2 will naturally score it poorly)
            # and keep going, matching the best-effort philosophy used
            # throughout the ingestion pipeline this evaluates.
            sample["actual_response"] = ""
            sample["actual_contexts"] = []
            sample["actual_tools_called"] = ["error"]
            sample["_pipeline_error"] = str(e)

        if progress_callback:
            progress_callback(i, total, sample)

        if i < total:
            time.sleep(INTER_CALL_DELAY_SECONDS)

    return enriched


if __name__ == "__main__":
    import json

    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(dataset_path) as f:
        golden = json.load(f)

    def _print_progress(i, total, sample):
        tool = sample["actual_tools_called"][0] if sample["actual_tools_called"] else "?"
        print(f"[{i}/{total}] id={sample['id']} tool={tool} response_len={len(sample['actual_response'])}")

    print(f"Running Phase 1 against {BACKEND_URL} — {len(golden['rag_samples'])} samples, "
          f"~{INTER_CALL_DELAY_SECONDS}s between calls...")
    enriched_result = run_live_pipeline(golden, progress_callback=_print_progress)

    out_path = os.path.join(os.path.dirname(__file__), "enriched_dataset.json")
    with open(out_path, "w") as f:
        json.dump(enriched_result, f, indent=2)
    print(f"Saved enriched dataset -> {out_path}")
