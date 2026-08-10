"""
Phase 2 — RAGAS metric scoring, plus Tool Correctness (Jaccard, zero LLM).

Follows the RAGAS 0.4.3 API rules documented in DOCS/10_EVALS.md exactly —
those rules exist because the obvious/older-tutorial API silently fails on
this version:

  1. llm_factory(), NOT LangchainLLMWrapper — collections metrics require
     InstructorLLM internally; LangchainLLMWrapper is rejected.
  2. AsyncOpenAI, NOT Groq or sync OpenAI — abatch_score() is async;
     Groq's client has no .messages attribute in the shape RAGAS expects.
  3. abatch_score(), NOT ragas.evaluate() — collections metrics aren't
     Metric subclasses, evaluate() rejects them with a TypeError.
  4. One sample at a time (GENERAL_BATCH_SIZE = 1) — the JUDGE_GROQ key's
     confirmed live TPM ceiling (6,000, not the nominal 14,400) means
     abatch_score() firing all sample coroutines concurrently saturates
     the window past 2 samples.
"""

import os
import asyncio

# --- Judge model config ---------------------------------------------------
# Separate key from the production GROQ_API_KEY (DOCS/05) so a heavy eval
# run can never rate-limit the live app. llama-3.1-8b-instant has the
# highest free-tier TPM, making it the right judge choice for eval workloads.
JUDGE_GROQ_API_KEY = os.getenv("JUDGE_GROQ", os.getenv("GROQ_API_KEY"))
JUDGE_MODEL = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# --- Rate-limit pacing (DOCS/10/11 — confirmed from live 429s) ------------
GENERAL_BATCH_SIZE = 1
SAMPLE_COOLDOWN_SECONDS = 40
EXPERIMENT_COOLDOWN_SECONDS = 62

# --- Context truncation — a single Faithfulness call exceeds 7,000 tokens
# without this (DOCS/10) ----------------------------------------------------
CONTEXT_TRUNCATE_CHARS = 300
CONTEXT_LIMIT = 2

_judge_llm = None
_local_embeddings = None


def get_judge_llm():
    """Lazy-init the RAGAS judge LLM — llm_factory + AsyncOpenAI per the
    API rules above. Matches the lazy-loading convention used throughout
    app/services/ (never initialize a client at import time)."""
    global _judge_llm
    if _judge_llm is None:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory

        client = AsyncOpenAI(api_key=JUDGE_GROQ_API_KEY, base_url=GROQ_BASE_URL)
        _judge_llm = llm_factory(JUDGE_MODEL, provider="openai", client=client)
    return _judge_llm


def get_local_embeddings():
    """Local sentence-transformers embeddings for Answer Relevancy /
    Answer Correctness — zero token cost, reuses the same library FlashRank
    and the text-embedding fallback already depend on. Class name and
    kwargs match DOCS/10_EVALS.md's quick reference exactly — this API
    (HuggingFaceEmbeddings, `model=`, `use_api=False`) is easy to get
    subtly wrong against RAGAS 0.4.3."""
    global _local_embeddings
    if _local_embeddings is None:
        from ragas.embeddings import HuggingFaceEmbeddings

        _local_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", use_api=False)
    return _local_embeddings


def _truncate_contexts(contexts: list[str]) -> list[str]:
    return [c[:CONTEXT_TRUNCATE_CHARS] for c in (contexts or [])[:CONTEXT_LIMIT]]


async def async_cooldown(seconds: int, on_tick=None):
    """asyncio.sleep, not time.sleep — yields control back to the event
    loop so a Streamlit status callback (on_tick) can update live instead
    of the whole run freezing for the cooldown duration."""
    step = 10
    remaining = seconds
    while remaining > 0:
        wait = min(step, remaining)
        await asyncio.sleep(wait)
        remaining -= wait
        if on_tick:
            on_tick(remaining)


# --- The 5 RAGAS metrics ---------------------------------------------------
# Each function scores ONE sample at a time (GENERAL_BATCH_SIZE) with a
# cooldown between, per the rate-limit pacing above. Every input dict shape
# matches DOCS/10's per-metric abatch_score() signature table exactly.

async def score_faithfulness(samples: list[dict], on_progress=None) -> list[float | None]:
    from ragas.metrics.collections import Faithfulness

    metric = Faithfulness(llm=get_judge_llm())
    scores = []
    for i, s in enumerate(samples):
        try:
            result = await metric.abatch_score([{
                "user_input": s["question"],
                "response": s["actual_response"],
                "retrieved_contexts": _truncate_contexts(s["actual_contexts"]),
            }])
            scores.append(float(result[0].value))
        except Exception:
            scores.append(None)
        if on_progress:
            on_progress(i + 1, len(samples))
        if i < len(samples) - 1:
            await async_cooldown(SAMPLE_COOLDOWN_SECONDS)
    return scores


async def score_answer_relevancy(samples: list[dict], on_progress=None) -> list[float | None]:
    from ragas.metrics.collections import AnswerRelevancy

    metric = AnswerRelevancy(llm=get_judge_llm(), embeddings=get_local_embeddings())
    scores = []
    for i, s in enumerate(samples):
        try:
            result = await metric.abatch_score([{
                "user_input": s["question"],
                "response": s["actual_response"],
            }])
            scores.append(float(result[0].value))
        except Exception:
            scores.append(None)
        if on_progress:
            on_progress(i + 1, len(samples))
        if i < len(samples) - 1:
            await async_cooldown(SAMPLE_COOLDOWN_SECONDS)
    return scores


async def score_context_precision(samples: list[dict], on_progress=None) -> list[float | None]:
    from ragas.metrics.collections import ContextPrecision

    metric = ContextPrecision(llm=get_judge_llm())
    scores = []
    for i, s in enumerate(samples):
        try:
            result = await metric.abatch_score([{
                "user_input": s["question"],
                "reference": s["reference"],
                "retrieved_contexts": _truncate_contexts(s["actual_contexts"]),
            }])
            scores.append(float(result[0].value))
        except Exception:
            scores.append(None)
        if on_progress:
            on_progress(i + 1, len(samples))
        if i < len(samples) - 1:
            await async_cooldown(SAMPLE_COOLDOWN_SECONDS)
    return scores


async def score_context_recall(samples: list[dict], on_progress=None) -> list[float | None]:
    from ragas.metrics.collections import ContextRecall

    metric = ContextRecall(llm=get_judge_llm())
    scores = []
    for i, s in enumerate(samples):
        try:
            result = await metric.abatch_score([{
                "user_input": s["question"],
                "retrieved_contexts": _truncate_contexts(s["actual_contexts"]),
                "reference": s["reference"],
            }])
            scores.append(float(result[0].value))
        except Exception:
            scores.append(None)
        if on_progress:
            on_progress(i + 1, len(samples))
        if i < len(samples) - 1:
            await async_cooldown(SAMPLE_COOLDOWN_SECONDS)
    return scores


async def score_answer_correctness(samples: list[dict], on_progress=None) -> list[float | None]:
    from ragas.metrics.collections import AnswerCorrectness

    metric = AnswerCorrectness(llm=get_judge_llm(), embeddings=get_local_embeddings())
    scores = []
    for i, s in enumerate(samples):
        try:
            result = await metric.abatch_score([{
                "user_input": s["question"],
                "response": s["actual_response"],
                "reference": s["reference"],
            }])
            scores.append(float(result[0].value))
        except Exception:
            scores.append(None)
        if on_progress:
            on_progress(i + 1, len(samples))
        if i < len(samples) - 1:
            await async_cooldown(SAMPLE_COOLDOWN_SECONDS)
    return scores


def score_tool_correctness(samples: list[dict]) -> list[float]:
    """
    Pure Jaccard set overlap — |called ∩ expected| / |called ∪ expected|.
    Zero LLM calls, zero embeddings, no rate limit, instant. Penalizes
    both missing tools (recall failure) and extra tools (precision
    failure) in one score.
    """
    scores = []
    for s in samples:
        called = set(s.get("actual_tools_called", []))
        expected = set(s.get("expected_tools", []))
        union = called | expected
        scores.append(len(called & expected) / len(union) if union else 0.0)
    return scores


# --- Orchestration -----------------------------------------------------

METRIC_FUNCS = {
    "faithfulness": score_faithfulness,
    "answer_relevancy": score_answer_relevancy,
    "context_precision": score_context_precision,
    "context_recall": score_context_recall,
    "answer_correctness": score_answer_correctness,
}


async def run_all_metrics(enriched_data: dict, on_progress=None) -> dict:
    """
    Runs all 5 LLM-judged metrics (with inter-experiment cooldowns) plus
    Tool Correctness, against samples that already have actual_response /
    actual_contexts / actual_tools_called populated by pipeline.py's Phase 1.

    Returns {metric_name: [scores...]} — one score list per metric, aligned
    to enriched_data["rag_samples"] order.
    """
    samples = enriched_data["rag_samples"]
    results = {}

    metric_names = list(METRIC_FUNCS.keys())
    for idx, name in enumerate(metric_names):
        if on_progress:
            on_progress(f"Starting {name}...", None)
        results[name] = await METRIC_FUNCS[name](
            samples,
            on_progress=(lambda i, t, n=name: on_progress(f"{n}: sample {i}/{t}", i / t)) if on_progress else None,
        )
        if idx < len(metric_names) - 1:
            if on_progress:
                on_progress(f"Cooldown before next experiment ({EXPERIMENT_COOLDOWN_SECONDS}s)...", None)
            await async_cooldown(EXPERIMENT_COOLDOWN_SECONDS)

    # Tool Correctness — no cooldown needed, zero LLM calls
    results["tool_correctness"] = score_tool_correctness(samples)

    return results


def summarize_scores(results: dict) -> dict:
    """Mean per metric, ignoring None (failed judge calls) — and a verdict
    per the DOCS/10/11 threshold convention: >=0.75 good, 0.50-0.75 fair, <0.50 poor."""
    summary = {}
    for name, scores in results.items():
        valid = [s for s in scores if s is not None]
        mean = sum(valid) / len(valid) if valid else None
        if mean is None:
            verdict = "N/A"
        elif mean >= 0.75:
            verdict = "Good"
        elif mean >= 0.50:
            verdict = "Fair"
        else:
            verdict = "Poor"
        summary[name] = {"mean": mean, "verdict": verdict, "n": len(scores), "n_valid": len(valid)}
    return summary


if __name__ == "__main__":
    import json

    enriched_path = os.path.join(os.path.dirname(__file__), "enriched_dataset.json")
    if not os.path.exists(enriched_path):
        print(f"No {enriched_path} found — run pipeline.py first to populate actual_response/actual_contexts.")
        raise SystemExit(1)

    with open(enriched_path) as f:
        enriched = json.load(f)

    def _print_progress(msg, frac):
        print(msg)

    all_results = asyncio.run(run_all_metrics(enriched, on_progress=_print_progress))
    summary = summarize_scores(all_results)

    print()
    print("=" * 60)
    for name, s in summary.items():
        mean_str = f"{s['mean']:.3f}" if s["mean"] is not None else "N/A"
        print(f"{name:22} mean={mean_str:8} verdict={s['verdict']:5} (n={s['n_valid']}/{s['n']})")
