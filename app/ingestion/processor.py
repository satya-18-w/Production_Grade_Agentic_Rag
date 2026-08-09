import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim
from app.services.retrieval.image_embedding import embed_image, get_image_embedding_dim
from app.services.vision.captioning import caption_image
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.images import (
    extract_images_from_pdf,
    extract_images_from_pptx,
    extract_images_from_docx,
    extract_images_from_html,
)
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service",)

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"
IMAGE_SUBDIR = "images"

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def save_image_locally(image_bytes: bytes, ext: str, source_type: str, filename: str, idx: int) -> str:
    """Save an extracted image as processed_data/<source_type>/images/<filename>_<idx>.<ext>."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type, IMAGE_SUBDIR)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}_{idx}.{ext}")
    with open(dest, "wb") as f:
        f.write(image_bytes)
    return dest


def process_images(file_path: str, filename: str, source_type: str, ext: str) -> list[dict]:
    """
    Extract + caption images (PDF, PPTX, DOCX, HTML), save them locally, and
    return caption records shaped like a text chunk so they ride the same
    embed/index path as ordinary text. DOCX/HTML images are frequently
    external links (docs sourced from web articles) rather than embedded —
    those are fetched too, subject to ALLOW_EXTERNAL_IMAGE_FETCH and the
    safety limits in app/ingestion/loaders/images.py.

    Best-effort throughout: a captioning failure skips that one image
    rather than failing the whole document's ingestion.
    """
    if ext == "pdf":
        images = extract_images_from_pdf(file_path)
    elif ext == "pptx":
        images = extract_images_from_pptx(file_path)
    elif ext == "docx":
        images = extract_images_from_docx(file_path)
    elif ext in ("html", "htm"):
        images = extract_images_from_html(file_path)
    else:
        return []

    caption_chunks = []
    with logfire.span("Image Extraction & Captioning", count=len(images)):
        for idx, img in enumerate(images):
            caption = caption_image(img.data, img.mime_type)
            if not caption:
                continue

            local_path = save_image_locally(img.data, img.ext, source_type, filename, idx)
            caption_chunks.append({
                "text": f"[Image — {img.location}] {caption}",
                "content_type": "image_caption",
                "image_path": local_path,
                "location": img.location,
                # Internal-only (leading underscore) — the raw bytes needed
                # for the Phase 2 CLIP embedding step below. Stripped before
                # the processed_data JSON is saved (not JSON-serializable,
                # and image_path already points at the saved file).
                "_image_bytes": img.data,
            })

        if caption_chunks:
            logfire.info(f"Captioned {len(caption_chunks)} image(s) from {filename}.")

    return caption_chunks


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → caption images → save locally → embed → index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            # 2. Multimodal: extract + caption images (PDF, PPTX, DOCX, HTML —
            #    embedded and, for DOCX/HTML, externally-linked images too)
            image_chunks = process_images(file_path, filename, source_type, ext)

            if (not full_text or not full_text.strip()) and not image_chunks:
                logfire.warning(f"No text or images extracted from {filename} — skipping.")
                return

            # 3. Chunk text, then merge with image caption chunks into one
            #    uniform list — everything downstream (save/embed/index)
            #    treats a caption exactly like any other text chunk.
            text_chunks = chunk_text(full_text) if full_text and full_text.strip() else []
            all_chunks = [{"text": c, "content_type": "text"} for c in text_chunks] + image_chunks

            if not all_chunks:
                return

            # 4. Save processed metadata locally (strip internal-only fields
            #    — e.g. raw image bytes carried for the embedding step below
            #    — not JSON-serializable and not needed in the saved
            #    artifact since image_path already points at the saved file)
            json_chunks = [{k: v for k, v in c.items() if not k.startswith("_")} for c in all_chunks]
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": json_chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 5. Embed and index in Qdrant — named vectors: every point gets
            #    a "text" vector (the chunk text or caption); image caption
            #    points additionally get an "image" vector (Phase 2 — the
            #    actual image pixels embedded via CLIP, independent of how
            #    the caption happens to be worded).
            with logfire.span("Vectorizing & Indexing"):
                text_vectors = embed_texts([c["text"] for c in all_chunks])
                points = []
                image_vectors_added = 0
                for chunk, text_vector in zip(all_chunks, text_vectors):
                    payload = {
                        "text": chunk["text"],
                        "source": filename,
                        "source_type": source_type,
                        "content_type": chunk["content_type"],
                    }
                    vectors = {"text": text_vector}

                    if chunk["content_type"] == "image_caption":
                        payload["image_path"] = chunk["image_path"]
                        payload["location"] = chunk["location"]
                        image_vector = embed_image(chunk["_image_bytes"])
                        if image_vector:
                            vectors["image"] = image_vector
                            image_vectors_added += 1

                    points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vectors, payload=payload))

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )
                logfire.info(
                    f"Indexed {len(points)} points to Qdrant from {filename} "
                    f"({len(text_chunks)} text, {len(image_chunks)} image, "
                    f"{image_vectors_added} with CLIP image vectors)."
                )

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — named vectors: "text" (dimension resolved at
        # runtime after embedding model probe) + "image" (fixed CLIP dim).
        # Every point gets a "text" vector; only image-caption points also
        # get an "image" vector — Qdrant allows a point to define a subset
        # of a collection's named vectors.
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            text_dim = get_embedding_dim()
            image_dim = get_image_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    "text": models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
                    "image": models.VectorParams(size=image_dim, distance=models.Distance.COSINE),
                },
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"(text: {text_dim}-dim, image: {image_dim}-dim, Cosine)."
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")
