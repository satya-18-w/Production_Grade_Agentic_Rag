"""
Eval dashboard — 3 tabs: Ground Truth, Live Pipeline, Eval Metrics.
Matches the design in DOCS/11_EVALS_PIPELINE.md exactly.

Run: streamlit run evals/app.py   (requires the FastAPI backend on :8000)
"""

import asyncio
import json
import os
import sys

import streamlit as st
import nest_asyncio

# nest_asyncio lets RAGAS's async abatch_score() calls run from inside
# Streamlit's own synchronous script-rerun loop, which already has an
# event loop of its own — without this, asyncio.run() inside a Streamlit
# callback raises "event loop already running".
nest_asyncio.apply()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.pipeline import run_live_pipeline, BACKEND_URL
from evals.guardrails_eval import run_guardrails_eval, compute_guardrail_metrics
from evals.metrics import run_all_metrics, summarize_scores
from evals.multimodal_metrics import run_multimodal_validation

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

st.set_page_config(page_title="RAG Eval Dashboard", page_icon="🧪", layout="wide")
st.title("🧪 Agentic RAG — Evaluation Dashboard")
st.caption(f"Backend: {BACKEND_URL}")


@st.cache_data
def load_golden_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


golden = load_golden_dataset()

if "enriched_dataset" not in st.session_state:
    st.session_state.enriched_dataset = None
if "guardrail_results" not in st.session_state:
    st.session_state.guardrail_results = None
if "metric_results" not in st.session_state:
    st.session_state.metric_results = None
if "multimodal_results" not in st.session_state:
    st.session_state.multimodal_results = None

tab_ground_truth, tab_live_pipeline, tab_eval_metrics, tab_multimodal = st.tabs(
    ["📋 Ground Truth", "🚀 Live Pipeline", "📊 Eval Metrics", "🖼️ Multimodal"]
)

# ── Tab 1: Ground Truth ─────────────────────────────────────────────────

with tab_ground_truth:
    st.subheader(f"{len(golden['rag_samples'])} RAG Samples")
    for sample in golden["rag_samples"]:
        with st.expander(f"#{sample['id']} [{sample['domain']}] {sample['question']}"):
            st.markdown(f"**Reference answer:** {sample['reference']}")
            st.markdown("**Relevant context(s):**")
            for ctx in sample["relevant_contexts"]:
                st.code(ctx, language=None)
            st.caption(f"Expected tools: {sample['expected_tools']}")

    st.subheader(f"{len(golden['guardrail_tests'])} Guardrail Tests")
    st.dataframe(
        [
            {
                "id": t["id"],
                "type": t["type"],
                "input": t["input"],
                "expected_blocked": t["expected_blocked"],
            }
            for t in golden["guardrail_tests"]
        ],
        width="stretch",
        hide_index=True,
    )

# ── Tab 2: Live Pipeline ─────────────────────────────────────────────────

with tab_live_pipeline:
    st.markdown(
        "Sends each golden question + guardrail test through the **live** "
        f"`{BACKEND_URL}/query` endpoint. ~10s between calls (Groq RPM buffer) — "
        f"expect roughly {(len(golden['rag_samples']) + len(golden['guardrail_tests'])) * 10 // 60} "
        "minutes total."
    )

    if st.button("▶️ Run Live Pipeline", type="primary"):
        progress_bar = st.progress(0.0, text="Starting Phase 1 — RAG samples...")
        total_steps = len(golden["rag_samples"]) + len(golden["guardrail_tests"])

        def _rag_progress(i, total, sample):
            progress_bar.progress(i / total_steps, text=f"RAG sample {i}/{len(golden['rag_samples'])}")

        enriched = run_live_pipeline(golden, progress_callback=_rag_progress)
        st.session_state.enriched_dataset = enriched

        def _guard_progress(i, total, annotated):
            frac = (len(golden["rag_samples"]) + i) / total_steps
            progress_bar.progress(frac, text=f"Guardrail test {i}/{total}")

        guard_results = run_guardrails_eval(golden["guardrail_tests"], progress_callback=_guard_progress)
        st.session_state.guardrail_results = guard_results

        progress_bar.progress(1.0, text="Done.")
        st.success("Live pipeline complete.")

    if st.session_state.enriched_dataset:
        st.subheader("RAG Sample Responses")
        st.dataframe(
            [
                {
                    "id": s["id"],
                    "question": s["question"],
                    "tool_called": s["actual_tools_called"][0] if s["actual_tools_called"] else "",
                    "response_preview": (s["actual_response"] or "")[:150],
                    "n_contexts": len(s["actual_contexts"]),
                }
                for s in st.session_state.enriched_dataset["rag_samples"]
            ],
            width="stretch",
            hide_index=True,
        )

    if st.session_state.guardrail_results:
        st.subheader("Guardrail Results")
        metrics = compute_guardrail_metrics(st.session_state.guardrail_results)
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{metrics['precision']:.2f}" if metrics["precision"] is not None else "N/A")
        col2.metric("Recall", f"{metrics['recall']:.2f}" if metrics["recall"] is not None else "N/A")
        col3.metric("Accuracy", f"{metrics['accuracy']:.2f}" if metrics["accuracy"] is not None else "N/A")
        st.dataframe(
            [
                {
                    "id": r["id"],
                    "type": r["type"],
                    "expected_blocked": r["expected_blocked"],
                    "actually_blocked": r["actually_blocked"],
                    "classification": r["classification"],
                }
                for r in st.session_state.guardrail_results
            ],
            width="stretch",
            hide_index=True,
        )

# ── Tab 3: Eval Metrics ─────────────────────────────────────────────────

with tab_eval_metrics:
    if not st.session_state.enriched_dataset:
        st.info("Run the Live Pipeline tab first — RAGAS scores the *actual* responses it collects.")
    else:
        st.markdown(
            "Runs the 5 RAGAS metrics (one sample at a time, with cooldowns for the "
            "`JUDGE_GROQ` rate limit) plus Tool Correctness. **Expect this to take "
            "roughly 50 minutes on the free tier** — see DOCS/11_EVALS_PIPELINE.md "
            "for the full token-budget breakdown."
        )

        if st.button("▶️ Run Eval Metrics", type="primary"):
            status_placeholder = st.empty()

            def _on_progress(msg, frac):
                if frac is not None:
                    status_placeholder.progress(frac, text=msg)
                else:
                    status_placeholder.text(msg)

            results = asyncio.run(run_all_metrics(st.session_state.enriched_dataset, on_progress=_on_progress))
            st.session_state.metric_results = results
            st.success("Eval metrics complete.")

    if st.session_state.metric_results:
        summary = summarize_scores(st.session_state.metric_results)
        st.subheader("Summary")
        st.dataframe(
            [
                {
                    "metric": name,
                    "mean": f"{s['mean']:.3f}" if s["mean"] is not None else "N/A",
                    "verdict": s["verdict"],
                    "scored": f"{s['n_valid']}/{s['n']}",
                }
                for name, s in summary.items()
            ],
            width="stretch",
            hide_index=True,
        )

        st.subheader("Per-Sample Scores")
        rag_samples = st.session_state.enriched_dataset["rag_samples"]
        per_sample_rows = []
        for i, s in enumerate(rag_samples):
            row = {"id": s["id"], "domain": s["domain"]}
            for metric_name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"):
                scores = st.session_state.metric_results.get(metric_name, [])
                row[metric_name] = f"{scores[i]:.2f}" if i < len(scores) and scores[i] is not None else "N/A"
            tc_scores = st.session_state.metric_results.get("tool_correctness", [])
            row["tool_correctness"] = f"{tc_scores[i]:.2f}" if i < len(tc_scores) else "N/A"
            per_sample_rows.append(row)
        st.dataframe(per_sample_rows, width="stretch", hide_index=True)

# ── Tab 4: Multimodal (Track F2) ──────────────────────────────────────────

with tab_multimodal:
    multimodal_samples = golden.get("multimodal_samples", [])
    st.markdown(
        "RAGAS's metrics are text-only — none of them can look at an image. This is a "
        "standalone Gemini-vision judge scoring whether an answer accurately describes "
        "what's actually in the picture. Each sample is scored **twice** — once against "
        "a deliberately correct answer, once against a deliberately wrong one — the same "
        "Good/Bad discrimination check used for the RAGAS experiments. A working judge "
        "scores the good answer meaningfully higher than the bad one."
    )

    if not multimodal_samples:
        st.info("No multimodal_samples in golden_dataset.json.")
    else:
        for sample in multimodal_samples:
            with st.expander(f"#{sample['id']} — {sample['question']}"):
                col_img, col_text = st.columns([1, 1.4])
                with col_img:
                    st.image(sample["image_path"])
                    st.caption(sample["source"])
                with col_text:
                    st.markdown(f"**Reference:** {sample['reference_description']}")
                    st.success(f"Good answer: {sample['good_answer']}")
                    st.error(f"Bad answer: {sample['bad_answer']}")

        if st.button("▶️ Run Multimodal Validation", type="primary"):
            status_placeholder = st.empty()

            def _mm_progress(i, total):
                status_placeholder.progress(i / total, text=f"Scoring {i}/{total} (good + bad per sample)...")

            st.session_state.multimodal_results = run_multimodal_validation(
                multimodal_samples, on_progress=_mm_progress
            )
            st.success("Multimodal validation complete.")

    if st.session_state.multimodal_results:
        st.subheader("Discrimination Check")
        st.dataframe(
            [
                {
                    "id": r["id"],
                    "good_score": f"{r['good_score']:.2f}" if r["good_score"] is not None else "N/A",
                    "bad_score": f"{r['bad_score']:.2f}" if r["bad_score"] is not None else "N/A",
                    "discriminates": "✅" if r["discriminates"] else ("❌" if r["discriminates"] is False else "N/A"),
                }
                for r in st.session_state.multimodal_results
            ],
            width="stretch",
            hide_index=True,
        )
