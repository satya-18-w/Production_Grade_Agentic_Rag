import re

import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings, PORTKEY_GATEWAY_CONFIG_SLUG, PORTKEY_GUARDRAIL_CONFIG_SLUG

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(content: str) -> str:
    """
    Drop <think>...</think> reasoning traces some Groq models (e.g.
    qwen3.6) inline into `content` regardless of reasoning_format —
    that param has no effect on those models, so this is the only way
    to keep raw thinking text out of what the user/guardrails see.
    """
    if not content:
        return content
    return _THINK_BLOCK_RE.sub("", content).strip()


# Documents what PORTKEY_GATEWAY_CONFIG_SLUG (below) is configured to do on
# the Portkey dashboard — kept here so the strategy is visible in the repo,
# not because it's sent on the wire. This account has block_inline_config
# enabled, so Portkey rejects an inline dict passed as `config=` (400
# inline_config_blocked); every call below must reference the saved config
# by its 'pc-...' slug instead.
#   - Fallback: primary @RAG1/qwen/qwen3.6-27b → @RAG2/groq/compound on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": "@RAG1/qwen/qwen3.6-27b", "reasoning_format": "hidden"}},
        {"override_params": {"model": "@RAG2/groq/compound"}},
    ]
}

# A saved config slug (string) is sent as-is; an inline dict is JSON-encoded
# into the x-portkey-config header. Accounts with "block_inline_config"
# enabled reject the latter, so the slug takes precedence when configured.
if settings.PORTKEY_USE_GATEWAY_CONFIG:
    RESOLVED_GATEWAY_CONFIG = (
        settings.PORTKEY_GATEWAY_CONFIG_SLUG or GATEWAY_CONFIG
    )
else:
    RESOLVED_GATEWAY_CONFIG = None

if settings.PORTKEY_USE_GATEWAY_CONFIG:
    portkey_client = Portkey(api_key=settings.PORTKEY_API_KEY, config=RESOLVED_GATEWAY_CONFIG)
else:
    portkey_client = Portkey(api_key=settings.PORTKEY_API_KEY)


def _build_portkey_headers(feature: str, config=None) -> dict:
    """Build Portkey headers without inline config unless explicitly enabled."""
    if settings.PORTKEY_USE_GATEWAY_CONFIG:
        return createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=config,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production",
            },
        )

    # If inline config is disabled at the account level, sending an inline
    # dict causes a 400 inline_config_blocked. We omit config here and let
    # the Portkey account defaults apply.
    return createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        provider=settings.LLM_PROVIDER,
        metadata={
            "feature": feature,
            "_user": "rag-system",
            "environment": "production",
        },
    )


def _resolve_portkey_model(slug: str, model: str) -> str:
    # With a valid config slug, models are sent as `@slug/model` (Portkey route).
    # Without that config path, pass the upstream model name directly to avoid
    # "Following keys are not valid: <slug>" errors when the account has
    # different/unknown provider aliases.
    # Portkey config slugs can include all routing now, so the model is sent
    # as plain model name to avoid reliance on external slug aliases.
    return model


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=_resolve_portkey_model(settings.GROQ_SLUG, "qwen/qwen3.6-27b"),
        temperature=0,
        default_headers=_build_portkey_headers(feature, RESOLVED_GATEWAY_CONFIG)
    )

# Documents what PORTKEY_GUARDRAIL_CONFIG_SLUG (below) is configured to do
# on the Portkey dashboard — see the GATEWAY_CONFIG comment above for why
# the slug, not this dict, is what's actually sent. Primary target is the
# fast/cheap 8b model — intent classification at the gate doesn't need the
# 70B model's quality. Falls back to the 70B target only if the 8b target
# itself is unavailable. This keeps guardrail calls inside the same
# fallback/retry/observability gateway as every other LLM call in the
# system, instead of a raw ChatGroq client.
GUARDRAIL_GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": "@RAG2/groq/compound"}},
        {"override_params": {"model": "@RAG1/qwen/qwen3.6-27b", "reasoning_format": "hidden"}},
    ]
}

if settings.PORTKEY_USE_GATEWAY_CONFIG:
    RESOLVED_GUARDRAIL_GATEWAY_CONFIG = (
        settings.PORTKEY_GUARDRAIL_CONFIG_SLUG or GUARDRAIL_GATEWAY_CONFIG
    )
else:
    RESOLVED_GUARDRAIL_GATEWAY_CONFIG = None


def get_guardrail_llm() -> ChatOpenAI:
    """
    Portkey-backed ChatOpenAI for the NeMo Guardrails gate. Routes through the
    same gateway (fallback + retry + dashboard visibility) as every other LLM
    call in the system.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=_resolve_portkey_model(settings.GROQ_SLUG_2, "groq/compound"),
        temperature=0,
        default_headers=_build_portkey_headers("guardrails", RESOLVED_GUARDRAIL_GATEWAY_CONFIG)
    )


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
