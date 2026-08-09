# 19 — Reliability Fixes

> **One-line summary:** Four reliability gaps found by a code audit — an unguarded exception in the guardrail gate, guardrail calls bypassing the LLM gateway, vector-search outages masquerading as empty results, and unbounded conversation memory — fixed and documented here.

---

## Why This Doc Exists

A build-status audit and a follow-up flaws review read every file in `app/` against what the rest of `DOCS/` claimed. Four issues stood out as things that only surface under real failure conditions or a long-running session — not bugs a quick manual test would catch. This doc is the record of what was wrong, what changed, and why, so the next person reading `08_GUARDRAILS.md` or `03_NODE_INTELLIGENCE.md` isn't misled by code that has since moved past them.

---

## Fix 1 — Guardrail Gate Now Fails Closed Instead of Crashing

### The Problem

`app/guardrails/rails.py` — `guard()` called `_rails.generate(...)` with no `try/except` around it. Any transient failure (a Groq rate-limit blip, a network timeout) raised straight up into `app/main.py`'s generic exception handler, which returned the same *"I encountered an internal error"* message it gives for a real backend crash. There was no way to tell, from the user's side or from Logfire's error logs, whether the safety gate itself had failed versus the rest of the pipeline.

### The Fix

```python
# app/guardrails/rails.py

with logfire.span("🛡️ Guardrails Check"):
    try:
        result = _rails.generate(messages=[{"role": "user", "content": message}])
    except Exception as e:
        # Fail CLOSED — a broken safety gate should block, not silently
        # let everything through. Logged distinctly from a pipeline
        # execution error so gate outages don't get buried under generic
        # "internal error" responses.
        logfire.error(f"🛡️ Guardrails check failed (gate unavailable) — failing closed: {e}")
        return True, (
            "I'm having trouble verifying that request right now. Please try again in a moment."
        )
```

**Why fail closed, not open?** A safety gate that silently lets requests through when it breaks is worse than one that occasionally over-blocks. Failing closed means a gate outage looks like "try again in a moment" to the user — annoying, but safe. Failing open would mean a Groq blip quietly disables the off-topic and jailbreak rails for the duration of the outage, with nothing in the logs to flag it as a security-relevant event.

**Why a distinct log message?** `logfire.error("🛡️ Guardrails check failed (gate unavailable)...")` is text-searchable and separable from `logfire.error("❌ Backend Execution Failed...")` in `main.py`. Before this fix, both looked identical in the dashboard — now a spike in gate failures is visible as its own signal, not folded into general error noise.

---

## Fix 2 — Guardrail LLM Calls Now Route Through the Portkey Gateway

### The Problem

Every other LLM call in the system — planner, responder — goes through Portkey (`get_langchain_llm()` or `portkey_client`) for retry, fallback, caching, and dashboard visibility (see [09_LLM_GATEWAY.md](09_LLM_GATEWAY.md)). The guardrail gate was the one exception: `initialize_rails()` built a raw `ChatGroq(api_key=settings.GROQ_API_KEY, ...)` directly. That meant the component sitting in front of *every single request* had no retry, no fallback target, and was invisible in the Portkey dashboard — the opposite of where resilience matters most.

### The Fix

A new gateway config and client function in `app/gateway/client.py`:

```python
# Guardrail gate config: primary target is the fast/cheap 8b model — intent
# classification at the gate doesn't need the 70B model's quality. Falls back
# to the 70B target only if the 8b target itself is unavailable.
GUARDRAIL_GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {"attempts": 2, "on_status_codes": [429, 503]},
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/llama-3.1-8b-instant"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile"}},
    ]
}

def get_guardrail_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG_2}/llama-3.1-8b-instant",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=GUARDRAIL_GATEWAY_CONFIG,
            metadata={"feature": "guardrails", "_user": "rag-system", "environment": "production"}
        )
    )
```

`app/guardrails/rails.py` now calls `get_guardrail_llm()` instead of building `ChatGroq` directly.

**Note the fallback order is deliberately reversed from the main RAG gateway config.** The main pipeline's `GATEWAY_CONFIG` (in `09_LLM_GATEWAY.md`) targets the 70B model first, falling back to 8b. The guardrail gate targets 8b first — it's a classification task, not generation, so the cheap/fast model is the right *primary* choice — and falls back to 70b only if 8b itself is unavailable, purely for gate availability.

**What this buys:** guardrail calls now show up in the Portkey dashboard alongside everything else, get 2 automatic retries on 429/503 before Fix 1's fail-closed path even triggers, and have a real fallback target instead of none.

---

## Fix 3 — Vector Search Failures No Longer Look Like Empty Results

### The Problem

`app/services/retrieval/qdrant_service.py` — `search_enterprise_knowledge()` caught every exception (auth failure, network partition, a genuine Qdrant outage) and returned `[]`. The retriever node, responder, and UI then behaved exactly as if the query legitimately had zero matching documents. A full vector-DB outage silently degraded to "the LLM answers from nothing" instead of surfacing as an error — with no signal that retrieval had failed rather than simply found nothing relevant.

### The Fix

A new exception type makes the distinction explicit:

```python
# app/services/retrieval/qdrant_service.py

class QdrantSearchError(Exception):
    """
    Raised when the search operation itself fails (auth, network, outage) —
    distinct from a legitimate zero-result search.
    """

def search_enterprise_knowledge(query: str, limit: int = 8):
    try:
        ...
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        raise QdrantSearchError(str(e)) from e
```

The retriever node catches it specifically and reports a distinct plan step:

```python
# app/agents/nodes/retriever.py

try:
    raw_results = search_enterprise_knowledge(query, limit=15)
except QdrantSearchError as e:
    logfire.error(f"Vector search unavailable — proceeding with no context: {e}")
    return {
        "documents": [],
        "status": "Knowledge base temporarily unavailable — answering without retrieved context.",
        "plan": state["plan"] + ["Retrieval: Failed (vector DB unavailable)"]
    }
```

**Why not just raise all the way up?** The graph still completes and the user still gets an answer (the responder can fall back to general knowledge) — this isn't a fail-closed situation like the guardrail gate, since answering "I don't have retrieval access right now, but here's what I know generally" is more useful than a hard error for a knowledge-lookup failure. What changed is that `plan` (shown in the UI as the reasoning trail) now says **"Retrieval: Failed (vector DB unavailable)"** instead of silently proceeding to "Context Retrieved" with zero documents — a legitimate empty search and an outage are no longer the same event.

---

## Fix 4 — Conversation Memory Is Now Windowed Before It Reaches a Prompt

### The Problem

`AgentState.messages` uses `Annotated[List[dict], operator.add]` — every turn appends, nothing ever trims. Both the planner and responder rebuilt the *entire* history into a prompt string on every call:

```python
# before — app/agents/nodes/planner.py and responder.py, duplicated
history = ""
for msg in state["messages"][:-1]:
    role = "User" if msg["role"] == "user" else "Assistant"
    history += f"{role}: {msg['content']}\n"
```

Since memory moved to a `PostgresSaver` checkpointer (persisting across restarts — see [03_NODE_INTELLIGENCE.md](03_NODE_INTELLIGENCE.md)), a long-running `thread_id` now keeps growing indefinitely. Latency and token cost scaled with *total* conversation length, not recent relevance, with no ceiling.

### The Fix

A shared, bounded history formatter in `app/agents/state.py`:

```python
# Bounds how much conversation history is rendered into the planner/responder
# prompts on every turn. The full history still persists via the Postgres
# checkpointer — this only windows what gets sent to the LLM, so prompt size
# and cost stay flat as a thread grows instead of scaling with total turns.
MAX_HISTORY_MESSAGES = 20

def format_history(messages: List[dict]) -> str:
    windowed = messages[-MAX_HISTORY_MESSAGES:] if len(messages) > MAX_HISTORY_MESSAGES else messages
    lines = []
    for msg in windowed:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) + ("\n" if lines else "")
```

Both `planner.py` and `responder.py` now call `format_history(state["messages"][:-1])` instead of maintaining their own copy of the same loop.

**Why 20 messages?** ~10 user/assistant turn pairs — enough for the planner to correctly classify follow-ups like "what about the second one" without needing the full thread, while keeping prompt size flat regardless of how long a conversation runs.

**Why not summarize instead of just windowing?** Summarization (compressing older turns into a running synopsis rather than dropping them) is the stronger fix and is on the roadmap — it requires an extra LLM call per truncation point, which is a larger change. Windowing is the immediate, zero-new-dependency fix: it bounds cost and latency today. The full history remains in Postgres regardless, so nothing is lost — only what gets rendered into the prompt is capped.

---

## Summary Table

| # | Flaw | File(s) | Fix |
|---|------|---------|-----|
| 1 | Guardrail gate crash on transient failure | `app/guardrails/rails.py` | `try/except` around `_rails.generate()`, fails closed, logged distinctly |
| 2 | Guardrail LLM bypassed the gateway | `app/gateway/client.py`, `app/guardrails/rails.py` | New `get_guardrail_llm()` — 8b primary / 70b fallback, routed through Portkey |
| 3 | Search failure indistinguishable from empty result | `app/services/retrieval/qdrant_service.py`, `app/agents/nodes/retriever.py` | New `QdrantSearchError`, caught with a distinct `plan` step |
| 4 | Unbounded conversation memory in prompts | `app/agents/state.py`, `planner.py`, `responder.py` | `MAX_HISTORY_MESSAGES = 20` + shared `format_history()` |

---

## See Also

- `DOCS/08_GUARDRAILS.md` — guardrail architecture (now reflects Portkey routing)
- `DOCS/09_LLM_GATEWAY.md` — the gateway config these fixes now route guardrail traffic through
- `DOCS/03_NODE_INTELLIGENCE.md` — planner/retriever/responder internals (now reflects the Postgres checkpointer and history windowing)
