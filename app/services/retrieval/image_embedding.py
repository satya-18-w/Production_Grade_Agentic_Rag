import logfire

# CLIP: text and images share one embedding space, so a natural-language
# query can be compared directly against image vectors — this is the
# mechanism Phase 2's image-space search relies on. Zero cost, local,
# reuses sentence-transformers (already a dependency for FlashRank and the
# text-embedding fallback), no new package needed.
CLIP_MODEL_NAME = "clip-ViT-B-32"
CLIP_DIM = 512

_clip_model = None


def _get_clip_model():
    """Lazy-load CLIP — same lazy-loading pattern as the FlashRank ranker
    and the text embedding fallback, so importing this module never
    triggers a model download or API call on its own."""
    global _clip_model
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer
        logfire.info(f"🖼️ Loading CLIP model ({CLIP_MODEL_NAME}) for image embeddings...")
        _clip_model = SentenceTransformer(CLIP_MODEL_NAME)
        logfire.info("🖼️ CLIP model ready.")
    return _clip_model


def get_image_embedding_dim() -> int:
    return CLIP_DIM


def embed_image(image_bytes: bytes) -> list[float] | None:
    """
    Embed an image into CLIP space for indexing.

    Best-effort: returns None on failure (corrupt image data, model load
    failure) rather than raising — image embedding failing shouldn't fail
    the whole document's ingestion. The point still gets its "text" vector
    from the caption regardless; only the "image" named vector is skipped.
    """
    try:
        import io
        from PIL import Image

        model = _get_clip_model()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        vector = model.encode(image)
        return vector.tolist()
    except Exception as e:
        logfire.warning(f"🖼️ CLIP image embedding failed: {e}")
        return None


def embed_text_for_image_search(query: str) -> list[float]:
    """
    Embed a text query into the SAME CLIP space as embed_image(), enabling
    cross-modal search: a query like "show me the architecture diagram" is
    compared directly against image vectors, not just caption text.
    """
    model = _get_clip_model()
    vector = model.encode(query)
    return vector.tolist()
