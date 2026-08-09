import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.image_embedding import embed_text_for_image_search


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

class QdrantSearchError(Exception):
    """
    Raised when the search operation itself fails (auth, network, outage) —
    distinct from a legitimate zero-result search. Lets callers tell
    "the knowledge base is down" apart from "nothing matched this query"
    instead of both collapsing into an empty list.
    """


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface, targeting the "text" named
    vector (Phase 2 schema: every point has a "text" vector; image caption
    points additionally have an "image" vector — see search_by_image_similarity).

    Raises QdrantSearchError on failure — callers decide how to degrade
    (e.g. the retriever node surfaces a distinct "retrieval failed" status
    instead of silently proceeding as if the search legitimately found nothing).
    """
    try:
        query_vector = embed_query(query)

        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            using="text",
            limit=limit,
            with_payload=True # JSON
        )

        results = []
        for res in response.points:
            result = {
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score,
                "match_type": "text",
                "content_type": res.payload.get("content_type", "text"),
            }
            # Written at index time for image_caption points (see
            # app/ingestion/processor.py) but never read back until now —
            # needed so the UI can render the actual source image, not just
            # its caption text.
            if "image_path" in res.payload:
                result["image_path"] = res.payload["image_path"]
            if "location" in res.payload:
                result["location"] = res.payload["location"]
            results.append(result)

        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        raise QdrantSearchError(str(e)) from e


def search_by_image_similarity(query: str, limit: int = 5):
    """
    Phase 2 — image-space search. Embeds the query into CLIP space and
    searches the "image" named vector, so a visually-described query
    ("show me the architecture diagram") can match against actual image
    content rather than only caption wording. Only points with an "image"
    vector (image caption chunks) can be returned — plain text chunks have
    no "image" vector and are naturally excluded by Qdrant.

    Best-effort and additive, unlike search_enterprise_knowledge: returns
    [] on any failure (including a collection that predates the Phase 2
    schema and has no "image" vector at all) rather than raising, so a
    missing or unavailable image index degrades to text-only results
    instead of failing retrieval.
    """
    try:
        query_vector = embed_text_for_image_search(query)

        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            using="image",
            limit=limit,
            with_payload=True
        )

        results = []
        for res in response.points:
            result = {
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score,
                "match_type": "image",
                "content_type": res.payload.get("content_type", "text"),
            }
            if "image_path" in res.payload:
                result["image_path"] = res.payload["image_path"]
            if "location" in res.payload:
                result["location"] = res.payload["location"]
            results.append(result)

        return results
    except Exception as e:
        logfire.warning(f"⚠️ Image-space search failed/unavailable: {e}")
        return []
