# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
import base64
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard, validate_uploaded_image

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    # Optional image upload (Phase 3, Track C) — base64 in the existing
    # JSON body rather than multipart. Costs ~33% payload inflation, worth
    # it for what's typically one image per chat message; avoids adding a
    # second request-handling path for a single attachment.
    image_base64: Optional[str] = None
    image_content_type: Optional[str] = None
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    try:
        # Gate 0: decode + validate any uploaded image BEFORE it enters the
        # graph — same "reject at the gate, before the expensive pipeline"
        # pattern as the text guardrail below (Gate 1). A rejected image
        # never reaches Qdrant, FlashRank, or any LLM.
        uploaded_image = None
        if request.image_base64:
            try:
                # validate=True — without it, b64decode silently drops
                # invalid characters instead of raising, which would let
                # malformed input past this check as corrupted bytes rather
                # than being caught here.
                image_bytes = base64.b64decode(request.image_base64, validate=True)
            except Exception:
                logfire.warning(f"🖼️ Image upload had invalid base64 | thread={thread_id}")
                return {
                    "question": q,
                    "answer": "That image could not be read. Please try uploading it again.",
                    "thought_process": ["Image Upload: Invalid encoding"],
                    "status": "Rejected — invalid image data.",
                    "sources": [],
                    "image_sources": []
                }

            content_type = request.image_content_type or "image/png"
            valid, reason = validate_uploaded_image(image_bytes, content_type)
            if not valid:
                logfire.warning(f"🖼️ Uploaded image rejected | thread={thread_id} | reason={reason}")
                return {
                    "question": q,
                    "answer": reason,
                    "thought_process": ["Image Upload: Rejected", reason],
                    "status": "Rejected by image guardrail.",
                    "sources": [],
                    "image_sources": []
                }

            uploaded_image = {"data": image_bytes, "mime_type": content_type}
            logfire.info(f"🖼️ Image upload accepted | thread={thread_id} | {len(image_bytes)} bytes")

        initial_state = {
            "messages": [{"role": "user", "content": q}],
            "current_query": q,
            "documents": [],
            "image_sources": [],
            "uploaded_image": uploaded_image,
            "plan": ["Start"],
            "status": "Initializing Graph..."
        }

        # Configuration for Memory (Thread ID)
        config = {"configurable": {"thread_id": thread_id}}

        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "image_sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
            "image_sources": final_output.get("image_sources", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
            "image_sources": []
        }
