# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
import base64
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
logfire_token = os.getenv("LOGFIRE_TOKEN")
if logfire_token:
    logfire.configure(token=logfire_token)
else:
    logfire.configure(send_to_logfire=False)

# Now safe to import app modules - logfire is already active
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.agents.graph import rag_agent
from app.auth import service as auth_service
from app.auth.dependencies import get_current_user
from app.auth.tokens import create_access_token
from app.config import settings
from app.guardrails import initialize_rails, guard, validate_uploaded_image

from pydantic import BaseModel
from typing import Optional


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rails()
    auth_service.init_auth_tables()
    yield


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)


def rate_limit_key(request: Request) -> str:
    """Rate-limit key. request.client.host is only the real client IP when
    this process is hit directly — behind any reverse proxy (Render,
    Railway, Fly, nginx, etc.) it's the proxy's own IP for every request,
    which would put all users in one shared bucket. Prefer X-Forwarded-For
    when present. Trusting it here assumes deployment behind a managed edge
    that sets/overwrites this header itself, rather than a setup that
    passes through whatever a client sends unfiltered."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ALLOWED_ORIGINS is "*" by default (fine for local/dev); set it to your
# real UI origin(s) — comma-separated — for a public deployment.
#
# allow_credentials is deliberately False: auth here is a bearer token the
# client attaches itself (Authorization header), not a cookie, so browsers
# never need `credentials: 'include'` to call this API. Leaving it True
# would also be a spec violation together with the "*" wildcard — browsers
# reject Access-Control-Allow-Origin: * on a credentialed response.
_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")] if settings.ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Auth
# ============================================================

class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateThreadRequest(BaseModel):
    title: Optional[str] = None


def _token_response(user_id: int, username: str) -> dict:
    return {
        "token": create_access_token(user_id, username),
        "user_id": user_id,
        "username": username,
    }


@app.post("/auth/signup")
@limiter.limit("5/minute")
def signup(request: Request, body: SignupRequest):
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    # bcrypt only hashes the first 72 bytes of its input and silently
    # ignores the rest — past that length, two different passwords sharing
    # the same first 72 bytes would both authenticate. Reject before that
    # point rather than hash something the user didn't actually set.
    if len(body.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be at most 72 bytes.")
    try:
        user_id = auth_service.signup(body.username, body.password)
    except auth_service.UsernameTakenError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _token_response(user_id, body.username.strip())


@app.post("/auth/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest):
    try:
        user_id = auth_service.login(body.username, body.password)
    except auth_service.InvalidCredentialsError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return _token_response(user_id, body.username.strip())


@app.get("/auth/threads")
def get_threads(current_user: dict = Depends(get_current_user)):
    return {"threads": auth_service.list_threads(current_user["user_id"])}


@app.post("/auth/threads")
def new_thread(body: CreateThreadRequest, current_user: dict = Depends(get_current_user)):
    thread_id = auth_service.create_thread(current_user["user_id"], body.title)
    return {"thread_id": thread_id}


# ============================================================
# Core RAG API
# ============================================================

class QueryRequest(BaseModel):
    q: str
    thread_id: str
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
@limiter.limit("30/minute")
def query(request: Request, body: QueryRequest, current_user: dict = Depends(get_current_user)):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = body.q
    thread_id = body.thread_id

    # A thread_id must have been issued to this user via POST /auth/threads
    # — otherwise any authenticated user could read/continue any other
    # user's conversation just by supplying their thread_id.
    owner_id = auth_service.thread_owner(thread_id)
    if owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Thread not found or not owned by this user.")

    try:
        # Gate 0: decode + validate any uploaded image BEFORE it enters the
        # graph — same "reject at the gate, before the expensive pipeline"
        # pattern as the text guardrail below (Gate 1). A rejected image
        # never reaches Qdrant, FlashRank, or any LLM.
        uploaded_image = None
        if body.image_base64:
            try:
                # validate=True — without it, b64decode silently drops
                # invalid characters instead of raising, which would let
                # malformed input past this check as corrupted bytes rather
                # than being caught here.
                image_bytes = base64.b64decode(body.image_base64, validate=True)
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

            content_type = body.image_content_type or "image/png"
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
            auth_service.touch_thread(thread_id, title_if_untitled=q)
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

        auth_service.touch_thread(thread_id, title_if_untitled=q)

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
