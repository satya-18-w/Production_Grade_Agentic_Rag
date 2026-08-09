import logfire

from app.config import settings

CAPTION_PROMPT = (
    "You are indexing a technical document for search. Describe this image in "
    "2-4 factual sentences: what it shows (diagram, screenshot, chart, table, "
    "photo), the key technical elements visible (labels, components, "
    "architecture, code, numbers), and what a reader would learn from it. "
    "Be specific and literal — do not speculate beyond what is visible."
)


# ── Primary: Gemini (google-genai SDK) ──────────────────────────────────────
#
# NOTE: uses the `google-genai` SDK (import path `google.genai`), not the
# older `google-generativeai` package already in requirements.txt for text
# embeddings. `google-generativeai` is EOL — no further updates or bug fixes
# — so new multimodal code is built on the current, supported SDK instead.

_client = None


def _get_gemini_client():
    """Lazy-init the Gemini client — same lazy-loading pattern used for the
    embedding model and FlashRank ranker, so this module can be imported
    without an API call firing at import time."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logfire.info(f"🖼️ Gemini vision client ready ({settings.GEMINI_VISION_MODEL}).")
    return _client


def _caption_with_gemini(image_bytes: bytes, mime_type: str) -> str | None:
    try:
        from google.genai import types

        client = _get_gemini_client()
        response = client.models.generate_content(
            model=settings.GEMINI_VISION_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                CAPTION_PROMPT,
            ],
        )
        caption = (response.text or "").strip()
        return caption or None
    except Exception as e:
        # Covers quota exhaustion (RESOURCE_EXHAUSTED / 429), rate limits,
        # network errors, and any other API failure alike — all of them mean
        # "no caption from Gemini right now," and all of them should fall
        # through to the local model rather than just giving up.
        logfire.warning(f"🖼️ Gemini captioning unavailable ({e}) — trying local SmolVLM fallback.")
        return None


# ── Fallback: local SmolVLM-2B (zero cost, no quota ceiling) ────────────────
#
# Only ever loaded if Gemini fails at least once — most runs never touch
# this while the Gemini free-tier quota holds. First use downloads the model
# weights (a few GB, cached by huggingface_hub afterwards) and is
# noticeably slower per image than Gemini on CPU — acceptable for an
# ingestion-time fallback, not for a latency-sensitive path.
#
# Requires transformers new enough to know the SmolVLM/Idefics3 architecture
# (see requirements.txt) — older transformers installs will raise a clear
# "unrecognized model type" error the first time this is actually invoked.

_smolvlm_model = None
_smolvlm_processor = None


def _get_smolvlm():
    global _smolvlm_model, _smolvlm_processor
    if _smolvlm_model is None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logfire.info(f"🖼️ Loading local SmolVLM fallback ({settings.SMOLVLM_MODEL_ID}) — first use only...")
        _smolvlm_processor = AutoProcessor.from_pretrained(settings.SMOLVLM_MODEL_ID)
        _smolvlm_model = AutoModelForVision2Seq.from_pretrained(
            settings.SMOLVLM_MODEL_ID,
            torch_dtype=torch.float32,  # CPU-safe; bfloat16 only helps on GPU
        )
        logfire.info("🖼️ SmolVLM fallback ready.")
    return _smolvlm_model, _smolvlm_processor


def _caption_with_smolvlm(image_bytes: bytes) -> str | None:
    try:
        import io
        from PIL import Image

        model, processor = _get_smolvlm()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")

        generated_ids = model.generate(**inputs, max_new_tokens=200)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # SmolVLM's decoded output includes the prompt template — keep only
        # the assistant's reply after it.
        if "Assistant:" in text:
            text = text.split("Assistant:", 1)[-1].strip()
        return text.strip() or None
    except Exception as e:
        logfire.error(f"🖼️ SmolVLM fallback captioning also failed: {e}")
        return None


# ── Public API ───────────────────────────────────────────────────────────────

def caption_image(image_bytes: bytes, mime_type: str) -> str | None:
    """
    Caption a single image — Gemini first, local SmolVLM-2B fallback if
    Gemini fails for any reason (quota exhausted, rate limited, network
    error). Ingestion keeps producing captions at zero marginal cost even
    after the Gemini free-tier daily quota runs out mid-run.

    Best-effort throughout: returns None only if both paths fail, so a
    captioning failure skips that one image rather than failing the whole
    document's ingestion.
    """
    caption = _caption_with_gemini(image_bytes, mime_type)
    if caption:
        return caption

    return _caption_with_smolvlm(image_bytes)
