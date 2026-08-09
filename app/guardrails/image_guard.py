import logfire

from app.ingestion.loaders.images import MAX_FETCH_BYTES

# This is deliberately NOT a NeMo/Colang rail. Colang's intent-matching runs
# on text — an uploaded image has none, so the off-topic/jailbreak/dialog
# flows in app/guardrails/rails.py can't say anything about it. This module
# is the pragmatic v1 gate for the image-upload path (Phase 3, Track C):
# size and content-type discipline, plus Gemini's own default safety
# filtering downstream in app/services/vision/answering.py (not disabled
# anywhere in this codebase). A custom NeMo Python action for explicit
# content moderation is a documented stretch goal — see
# DOCS/13_MULTIMODAL_RAG.md — worth building once real usage shows this
# baseline isn't enough, not before.

# Same size discipline already applied to externally-fetched images at
# ingestion time (app/ingestion/loaders/images.py) — reused, not
# re-invented, so there's one cap to reason about instead of two.
MAX_UPLOAD_BYTES = MAX_FETCH_BYTES

_ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
}


def validate_uploaded_image(image_bytes: bytes, content_type: str) -> tuple[bool, str | None]:
    """
    Server-side gate for a user-uploaded image — run this before an upload
    ever reaches the vision-native answering path.

    Returns:
        (True,  None)    — passed, safe to proceed.
        (False, reason)  — rejected; `reason` is plain-language and safe to
                           show the user directly (mirrors the guardrail
                           gate's existing rail_response pattern).
    """
    if not image_bytes:
        return False, "No image data received."

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        logfire.warning(
            f"🖼️ Uploaded image rejected — exceeds {MAX_UPLOAD_BYTES} byte cap "
            f"({len(image_bytes)} bytes)."
        )
        return False, "That image is too large. Please upload something under 15 MB."

    normalized_type = (content_type or "").split(";")[0].strip().lower()
    if normalized_type not in _ALLOWED_CONTENT_TYPES:
        logfire.warning(f"🖼️ Uploaded image rejected — unsupported content type: {content_type!r}")
        return False, "That file doesn't look like a supported image type (PNG, JPEG, GIF, WEBP, or BMP)."

    # Audit trail — every upload event logged distinctly, mirroring the
    # guardrail-fired logging pattern in app/guardrails/rails.py, so image
    # uploads are visible in the same observability stream as every other
    # guardrail decision rather than being invisible to Logfire.
    logfire.info(f"🖼️ Image upload passed validation ({len(image_bytes)} bytes, {normalized_type}).")
    return True, None
