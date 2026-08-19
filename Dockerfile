# Single image, reused for three roles via docker-compose's `command:`
# override: the FastAPI backend, the Streamlit UI, and one-off ingestion
# runs. They share the exact same dependency set, so one image is simpler
# to build and keep in sync than three — the only per-role difference is
# the process that gets started, not what's installed.

FROM python:3.12-slim

# --- System dependencies ---------------------------------------------------
# libreoffice-impress + poppler-utils: Track E (PPTX vector-shape diagram
# rendering, app/ingestion/loaders/images.py::render_pptx_diagram_slides).
# The one loader in this codebase with a system-binary dependency — see
# DOCS/13_MULTIMODAL_RAG.md. Bundled into the main image rather than split
# into a separate ingestion-only image, so `docker compose up` gives a
# fully working multimodal pipeline with no extra setup.
#
# build-essential: a handful of this project's dependencies (numba/llvmlite
# pinned in requirements.txt, some ML packages) can need to compile C
# extensions if a prebuilt wheel isn't available for this exact base image;
# cheap insurance against an opaque pip build failure.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-impress \
    poppler-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first, separately from the app code, so
# `docker build` can reuse this (slow — torch, transformers, the LangChain
# ecosystem) layer on rebuilds that only change application code.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Documentation, not enforcement — the actual bound port is whatever
# `command:` in docker-compose.yml passes to uvicorn/streamlit for that
# service.
EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
