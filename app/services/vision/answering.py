import logfire

from app.config import settings

# Same google-genai client pattern as captioning.py (see that module's
# comment for why: google-generativeai is EOL, google-genai is current).
# Kept as its own lazy singleton rather than sharing captioning.py's client
# global — matches the existing per-module lazy-singleton convention used
# throughout this codebase (embedding.py, ranking_service.py, image_embedding.py).
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logfire.info(f"🖼️ Gemini vision-answering client ready ({settings.GEMINI_VISION_MODEL}).")
    return _client


# Caps payload size/cost — most questions need at most one or two diagrams,
# not every image a query happened to retrieve.
MAX_VISION_IMAGES = 3


def answer_with_vision(prompt: str, images: list[tuple[bytes, str]]) -> str | None:
    """
    Generate an answer using Gemini multimodal — the prompt plus up to
    MAX_VISION_IMAGES actual images, not just their captions.

    Best-effort: returns None on any failure (quota exhausted, rate limited,
    network error, bad image data) so the caller can fall back to the
    existing text-only Groq/Portkey path — the same fallback shape already
    proven for caption_image() in captioning.py, just one layer up at
    generation time instead of ingestion time.
    """
    if not images:
        return None

    try:
        from google.genai import types

        client = _get_client()
        parts = [types.Part.from_bytes(data=data, mime_type=mime) for data, mime in images[:MAX_VISION_IMAGES]]
        parts.append(prompt)

        response = client.models.generate_content(
            model=settings.GEMINI_VISION_MODEL,
            contents=parts,
        )
        answer = (response.text or "").strip()
        return answer or None
    except Exception as e:
        logfire.warning(f"🖼️ Vision-native answering unavailable ({e}) — falling back to text-only response.")
        return None
