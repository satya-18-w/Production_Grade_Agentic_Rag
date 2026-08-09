import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
    search_by_image_similarity,
    QdrantSearchError,
)
from app.services.retrieval.ranking_service import rerank_documents

# Phase 2: how many image-space candidates to pull alongside the primary
# text-space search. Kept modest — these are additive, not a replacement
# for text search, and still get cut down by the same reranking pass.
IMAGE_SEARCH_LIMIT = 5


def retrieve_node(state: AgentState):
    """
    Performs vector search (text-space and image-space) and semantic
    reranking for technical queries.
    """
    query = state["current_query"]


    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        try:
            raw_results = search_enterprise_knowledge(query, limit=15)
        except QdrantSearchError as e:
            logfire.error(f"Vector search unavailable — proceeding with no context: {e}")
            return {
                "documents": [],
                "image_sources": [],
                "status": "Knowledge base temporarily unavailable — answering without retrieved context.",
                "plan": state["plan"] + ["Retrieval: Failed (vector DB unavailable)"]
            }
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        # Phase 2: also search image-space so visually-described queries
        # ("show me the diagram of...") can surface image captions that
        # text-space search alone might rank low, matching against the
        # actual image content rather than only caption wording.
        # Best-effort — a failure here (or a collection that predates the
        # Phase 2 schema) degrades to text-only results, never raises.
        image_results = search_by_image_similarity(query, limit=IMAGE_SEARCH_LIMIT)
        if image_results:
            logfire.info(f"Retrieved {len(image_results)} candidate(s) from image-space search")

        # Merge + dedupe — a caption chunk has both a "text" and an "image"
        # vector, so it can legitimately surface from both searches.
        seen_content = set()
        merged_results = []
        for doc in raw_results + image_results:
            if doc["content"] in seen_content:
                continue
            seen_content.add(doc["content"])
            merged_results.append(doc)

        # content -> full record, so the full record (image_path, location,
        # content_type) can be recovered after reranking without changing
        # rerank_documents()'s plain-string signature — other callers depend
        # on that signature staying simple.
        content_to_record = {doc["content"]: doc for doc in merged_results}
        doc_contents = [doc['content'] for doc in merged_results]

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]

        # Surface which of the reranked chunks are images, so the UI can
        # render the actual source image next to its caption — not just the
        # caption text, which is all `documents` carries.
        image_sources = []
        for content in reranked_contents:
            record = content_to_record.get(content)
            if record and record.get("content_type") == "image_caption" and "image_path" in record:
                image_sources.append({
                    "image_path": record["image_path"],
                    "location": record.get("location", ""),
                    "caption_preview": content[:200],
                    "source": record.get("source", "Unknown"),
                })

    return {
        "documents": formatted_docs,
        "image_sources": image_sources,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }
