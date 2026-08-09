from typing import TypedDict, List, Optional, Annotated
import operator


# Bounds how much conversation history is rendered into the planner/responder
# prompts on every turn. The full history still persists via the Postgres
# checkpointer — this only windows what gets sent to the LLM, so prompt size
# and cost stay flat as a thread grows instead of scaling with total turns.
MAX_HISTORY_MESSAGES = 20


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[List[dict], operator.add]
    current_query: str
    documents: List[str]
    # Structured records for retrieved images (path, location, caption
    # preview, source) — separate from `documents` (plain prompt strings)
    # so the UI can render the actual image without restructuring how the
    # responder's prompt gets built. Every node that can skip retrieval
    # must explicitly reset this to [] — no reducer is defined, so an
    # unset key otherwise carries a previous turn's value forward.
    image_sources: List[dict]
    # A user-uploaded image for this turn (Phase 3, Track C) — {"data":
    # bytes, "mime_type": str} or None. main.py sets this fresh on every
    # /query call from the current request (never carried over from a
    # previous turn — an image attached once shouldn't silently keep
    # applying to later, unrelated messages in the same thread).
    uploaded_image: Optional[dict]
    plan: List[str]
    status: str
    final_answer: str


def format_history(messages: List[dict]) -> str:
    """
    Render the most recent MAX_HISTORY_MESSAGES as a role-prefixed transcript.
    Older turns are dropped from the prompt (they remain in Postgres) so
    long-running threads don't grow the planner/responder prompt without bound.
    """
    windowed = messages[-MAX_HISTORY_MESSAGES:] if len(messages) > MAX_HISTORY_MESSAGES else messages
    lines = []
    for msg in windowed:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) + ("\n" if lines else "")
