"""
Track F2 — multimodal eval coverage.

RAGAS's collections metrics (Faithfulness, Answer Relevancy, etc.) are
text-only — none of them can take an image as input, so none of them can
answer "does this answer actually describe what's in the picture." This
module is a separate, standalone rubric-based judge for exactly that
question, deliberately NOT built on RAGAS's abatch_score() machinery
(which has no multimodal metric to extend).

Uses Gemini vision (the same model already integrated for captioning and
vision-native answering) as the judge, since the RAGAS/Groq judge pipeline
in metrics.py is text-only and structurally cannot look at an image.
"""

import json
import re

import logfire

from app.config import settings

_client = None


def _get_client():
    """Lazy-init, matching the app's own vision modules — importing this
    file never triggers an API call or the Gemini client construction."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


JUDGE_PROMPT_TEMPLATE = """You are grading whether an AI assistant's answer accurately describes an image, given a question about it.

QUESTION ASKED:
{question}

THE ASSISTANT'S ANSWER:
{answer}

Look at the attached image carefully. Score how accurately the answer describes what is actually visible in the image, on a 0.0-1.0 scale:
- 1.0 = every claim in the answer is visibly true in the image
- 0.5 = partially accurate — some claims match, some are wrong, vague, or unsupported by the image
- 0.0 = the answer describes something not visible in the image, or contradicts it

Respond with ONLY a JSON object, no other text: {{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
"""


def score_multimodal_faithfulness(image_bytes: bytes, mime_type: str, question: str, answer: str) -> dict | None:
    """
    Rubric-scored: does `answer` accurately describe what's actually in
    the image, given `question`? Returns {"score": float, "reasoning": str}
    or None on any failure (bad image, API error, unparseable judge
    response) — best-effort, matching every other eval/vision function in
    this codebase; a single failed judgment shouldn't abort a batch run.
    """
    try:
        from google.genai import types

        client = _get_client()
        prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)

        response = client.models.generate_content(
            model=settings.GEMINI_VISION_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
        )
        raw = (response.text or "").strip()

        # Judge models occasionally wrap JSON in a ```json fence despite
        # instructions not to — strip it rather than fail the whole score.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logfire.warning(f"🖼️ Multimodal judge returned no parseable JSON: {raw[:200]!r}")
            return None

        parsed = json.loads(match.group(0))
        score = float(parsed["score"])
        score = max(0.0, min(1.0, score))  # clamp — a judge hallucinating 1.5 shouldn't corrupt the mean
        return {"score": score, "reasoning": parsed.get("reasoning", "")}
    except Exception as e:
        logfire.warning(f"🖼️ Multimodal faithfulness judging failed: {e}")
        return None


def score_multimodal_samples(samples: list[dict], on_progress=None) -> list[dict | None]:
    """
    samples: list of {"image_path": str, "mime_type": str, "question": str, "answer": str}
    Returns one score dict (or None) per sample, same order.
    """
    results = []
    for i, sample in enumerate(samples):
        try:
            with open(sample["image_path"], "rb") as f:
                image_bytes = f.read()
            result = score_multimodal_faithfulness(
                image_bytes, sample["mime_type"], sample["question"], sample["answer"]
            )
        except Exception as e:
            logfire.warning(f"🖼️ Could not read image for multimodal scoring ({sample.get('image_path')}): {e}")
            result = None
        results.append(result)
        if on_progress:
            on_progress(i + 1, len(samples))
    return results


def run_multimodal_validation(multimodal_samples: list[dict], on_progress=None) -> list[dict]:
    """
    Scores BOTH good_answer and bad_answer for each golden_dataset.json
    multimodal_samples entry — the same "does the metric actually
    discriminate" check DOCS/10_EVALS.md describes for the RAGAS
    Good/Medium/Bad triples. A working judge should score good_answer near
    1.0 and bad_answer near 0.0; if they come back close together (or
    inverted), the judge prompt itself is broken, not just one sample.
    """
    results = []
    total = len(multimodal_samples) * 2
    step = 0

    for sample in multimodal_samples:
        try:
            with open(sample["image_path"], "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            logfire.warning(f"🖼️ Could not read image ({sample.get('image_path')}): {e}")
            results.append({"id": sample["id"], "good_score": None, "bad_score": None, "discriminates": None})
            step += 2
            if on_progress:
                on_progress(step, total)
            continue

        good = score_multimodal_faithfulness(image_bytes, sample["mime_type"], sample["question"], sample["good_answer"])
        step += 1
        if on_progress:
            on_progress(step, total)

        bad = score_multimodal_faithfulness(image_bytes, sample["mime_type"], sample["question"], sample["bad_answer"])
        step += 1
        if on_progress:
            on_progress(step, total)

        good_score = good["score"] if good else None
        bad_score = bad["score"] if bad else None
        discriminates = (good_score is not None and bad_score is not None and good_score > bad_score)

        results.append({
            "id": sample["id"],
            "good_score": good_score,
            "bad_score": bad_score,
            "discriminates": discriminates,
        })

    return results
