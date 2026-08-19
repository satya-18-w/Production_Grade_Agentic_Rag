import logfire
from nemoguardrails import RailsConfig, LLMRails

from app.gateway import get_guardrail_llm, strip_reasoning
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Routes through the Portkey gateway (get_guardrail_llm) instead of a raw
    ChatGroq client, so the gate gets the same fallback/retry/dashboard
    visibility as every other LLM call in the system. Targets
    groq/compound first — fast intent classification — falling back to
    qwen/qwen3.6-27b only if the primary target is unavailable.
    """
    global _rails

    guard_llm = get_guardrail_llm()

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (Portkey-routed groq/compound).")



def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired, or the gate itself failed;
                                return this response immediately, skip the RAG pipeline.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

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

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        content = strip_reasoning(content)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
