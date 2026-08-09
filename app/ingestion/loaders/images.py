import ipaddress
import os
import socket
from typing import NamedTuple, List, Optional, Tuple
from urllib.parse import urlparse

import logfire

from app.config import settings


class ExtractedImage(NamedTuple):
    data: bytes
    ext: str
    mime_type: str
    location: str  # human-readable locator, e.g. "page 3", "slide 5", "image 2"


# Skip tiny images — icons, bullets, logos, avatars, decorative dividers.
# These add captioning cost/noise without carrying real technical content.
# Conservative threshold; real diagrams/screenshots are almost always well
# above this (a confirmed 64x64 author-avatar in pods_autoscale.html is
# exactly the kind of thing this is meant to filter out).
MIN_IMAGE_BYTES = 3000

# External fetch safety limits — Kubernetes docs sourced from web articles
# (Medium, vendor docs) commonly keep images as external links rather than
# embedding them (confirmed: 20 of 20 images in cronjobs.docx, all 4 images
# across the true_data HTML files). Fetching them is opt-out, not opt-in,
# but stays bounded: public hosts only, size-capped, timeout-bounded.
MAX_FETCH_BYTES = 15 * 1024 * 1024  # 15 MB — generous for a diagram, blocks abuse
FETCH_TIMEOUT = 10  # seconds
_FETCH_USER_AGENT = "Mozilla/5.0 (compatible; EnterpriseRAGIngestion/1.0)"

_EXT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "webp": "image/webp",
    "svg": "image/svg+xml",
}
_MIME_TO_EXT = {v: k for k, v in _EXT_TO_MIME.items() if k != "jpg"}


# ── PDF ──────────────────────────────────────────────────────────────────────

def extract_images_from_pdf(file_path: str) -> List[ExtractedImage]:
    """
    Extract embedded raster images from a PDF using pypdf's built-in image
    support (page.images) — no new dependency; pypdf handles the filter
    decoding internally and returns ready-to-write bytes per image.
    """
    from pypdf import PdfReader

    images: List[ExtractedImage] = []
    with logfire.span("Image Extraction (PDF)", filename=file_path):
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                for img in page.images:
                    if len(img.data) < MIN_IMAGE_BYTES:
                        continue
                    ext = img.name.rsplit(".", 1)[-1].lower() if "." in img.name else "png"
                    mime = _EXT_TO_MIME.get(ext, "image/png")
                    images.append(ExtractedImage(data=img.data, ext=ext, mime_type=mime, location=f"page {page_num}"))
            logfire.info(f"Extracted {len(images)} image(s) from {file_path}.")
        except Exception as e:
            # Best-effort — a broken image stream shouldn't fail text extraction.
            logfire.warning(f"Image extraction failed for {file_path}: {e}")
    return images


# ── PPTX ─────────────────────────────────────────────────────────────────────

def extract_images_from_pptx(file_path: str) -> List[ExtractedImage]:
    """
    Extract embedded images from a PPTX using python-pptx's shape.image API.
    Note: slides built from native vector shapes (freeform/auto-shape, common
    for hand-drawn architecture diagrams) have no PICTURE shapes at all and
    will legitimately yield zero images — that's not a bug, there's simply
    nothing to extract without rendering the slide itself.
    """
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    images: List[ExtractedImage] = []
    with logfire.span("Image Extraction (PPTX)", filename=file_path):
        try:
            prs = Presentation(file_path)
            for slide_num, slide in enumerate(prs.slides, start=1):
                for shape in slide.shapes:
                    if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                        continue
                    image = shape.image
                    if len(image.blob) < MIN_IMAGE_BYTES:
                        continue
                    images.append(ExtractedImage(
                        data=image.blob,
                        ext=image.ext,
                        mime_type=image.content_type,
                        location=f"slide {slide_num}",
                    ))
            logfire.info(f"Extracted {len(images)} image(s) from {file_path}.")
        except Exception as e:
            logfire.warning(f"Image extraction failed for {file_path}: {e}")
    return images


# ── DOCX ─────────────────────────────────────────────────────────────────────

def extract_images_from_docx(file_path: str) -> List[ExtractedImage]:
    """
    Extract images referenced in a DOCX — both embedded and externally-linked.
    Docs sourced from web articles (Medium, vendor docs) tend to keep images
    as external links rather than embedding them, so external fetch is a
    first-class path here, not a fallback.
    """
    from docx import Document as DocxDocument

    images: List[ExtractedImage] = []
    with logfire.span("Image Extraction (DOCX)", filename=file_path):
        try:
            doc = DocxDocument(file_path)
            img_rels = [r for r in doc.part.rels.values() if "image" in r.reltype]

            for idx, rel in enumerate(img_rels, start=1):
                if rel.is_external:
                    fetched = _fetch_external_image(rel.target_ref)
                else:
                    try:
                        fetched = (rel.target_part.blob, rel.target_part.content_type)
                    except Exception:
                        fetched = None

                if fetched is None:
                    continue
                data, mime = fetched
                if len(data) < MIN_IMAGE_BYTES:
                    continue
                ext = _MIME_TO_EXT.get(mime, mime.split("/")[-1] if "/" in mime else "png")
                images.append(ExtractedImage(data=data, ext=ext, mime_type=mime, location=f"image {idx}"))

            logfire.info(f"Extracted {len(images)} image(s) from {file_path}.")
        except Exception as e:
            logfire.warning(f"Image extraction failed for {file_path}: {e}")
    return images


# ── HTML ─────────────────────────────────────────────────────────────────────

def extract_images_from_html(file_path: str) -> List[ExtractedImage]:
    """
    Extract images referenced by <img> tags in a local HTML file — data URIs
    decoded directly, external URLs fetched with safety limits, relative
    paths resolved against the HTML file's own directory.
    """
    from bs4 import BeautifulSoup

    images: List[ExtractedImage] = []
    base_dir = os.path.dirname(os.path.abspath(file_path))

    with logfire.span("Image Extraction (HTML)", filename=file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            for idx, img in enumerate(soup.find_all("img"), start=1):
                src = (img.get("src") or "").strip()
                if not src:
                    continue

                if src.startswith("data:"):
                    fetched = _decode_data_uri(src)
                elif src.startswith("http://") or src.startswith("https://"):
                    fetched = _fetch_external_image(src)
                else:
                    fetched = _read_local_image(os.path.join(base_dir, src), base_dir)

                if fetched is None:
                    continue
                data, mime = fetched
                if len(data) < MIN_IMAGE_BYTES:
                    continue
                ext = _MIME_TO_EXT.get(mime, mime.split("/")[-1] if "/" in mime else "png")
                images.append(ExtractedImage(data=data, ext=ext, mime_type=mime, location=f"image {idx}"))

            logfire.info(f"Extracted {len(images)} image(s) from {file_path}.")
        except Exception as e:
            logfire.warning(f"Image extraction failed for {file_path}: {e}")
    return images


# ── Shared helpers: external fetch (SSRF-guarded), data URI, local path ────────

def _is_safe_public_host(hostname: str) -> bool:
    """
    Basic SSRF guard for the external-image fetch path: reject hosts that
    resolve to private / loopback / link-local / reserved / multicast IPs.
    Image URLs come from document content (docx/html source files chosen for
    ingestion), not live user input — but the same hygiene applies regardless
    of source.
    """
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in addr_info:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _fetch_external_image(url: str) -> Optional[Tuple[bytes, str]]:
    """
    Fetch an external image URL with safety limits: http/https + public-host
    only, size-capped streaming download, request timeout, content-type
    validated as image/*. Returns (data, mime_type) or None on any
    failure/rejection — best-effort, never raises.
    """
    if not settings.ALLOW_EXTERNAL_IMAGE_FETCH:
        return None

    try:
        import requests

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return None
        if not _is_safe_public_host(parsed.hostname):
            logfire.warning(f"Refusing to fetch image from non-public host: {url}")
            return None

        resp = requests.get(
            url,
            timeout=FETCH_TIMEOUT,
            stream=True,
            headers={"User-Agent": _FETCH_USER_AGENT},
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            return None

        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > MAX_FETCH_BYTES:
                logfire.warning(f"Image fetch exceeded {MAX_FETCH_BYTES} byte cap, aborting: {url}")
                return None
            chunks.append(chunk)

        return b"".join(chunks), content_type
    except Exception as e:
        logfire.warning(f"External image fetch failed ({url}): {e}")
        return None


def _decode_data_uri(data_uri: str) -> Optional[Tuple[bytes, str]]:
    """Decode a base64 `data:` URI image into (data, mime_type)."""
    import base64

    try:
        header, b64data = data_uri.split(",", 1)
        mime = (header.split(";")[0].replace("data:", "").strip() or "image/png")
        return base64.b64decode(b64data), mime
    except Exception:
        return None


def _read_local_image(path: str, base_dir: str) -> Optional[Tuple[bytes, str]]:
    """
    Read a locally-referenced image (relative <img src> path) from disk.
    Resolves symlinks/`..` and confirms the result stays inside base_dir
    before opening it — a relative src in a source HTML file shouldn't be
    able to walk outside the directory it was found in.
    """
    try:
        real_base = os.path.realpath(base_dir)
        real_path = os.path.realpath(path)
        if real_path != real_base and not real_path.startswith(real_base + os.sep):
            return None
        if not os.path.isfile(real_path):
            return None
        with open(real_path, "rb") as f:
            data = f.read()
        ext = real_path.rsplit(".", 1)[-1].lower() if "." in real_path else "png"
        return data, _EXT_TO_MIME.get(ext, "image/png")
    except Exception:
        return None
