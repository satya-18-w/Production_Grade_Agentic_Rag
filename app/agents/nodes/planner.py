from app.agents.state import AgentState, format_history
from app.gateway import get_langchain_llm
import logfire

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the recent conversation.
    """
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    # An uploaded image (Phase 3, Track C) always routes technical —
    # retrieval still runs so any complementary text context is available
    # alongside the image (see responder.py), but there's no ambiguity to
    # ask the LLM to resolve: the user attached something to ask about.
    # Skips the classification call entirely rather than trusting the LLM
    # not to misclassify an image-only message (e.g. "what is this?") as
    # conversational.
    if state.get("uploaded_image"):
        logfire.info("🖼️ Image uploaded — routing technical, skipping intent classification.")
        return {
            "current_query": user_message,
            "status": "Image uploaded — analyzing with any related context...",
            "plan": ["Intent: Technical (image upload)", f"Search Term: {user_message}"]
        }

    # Windowed conversation history (excluding the latest message) — bounded
    # by MAX_HISTORY_MESSAGES so prompt size doesn't grow with thread length.
    history = format_history(state["messages"][:-1])

    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """
    
    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")
    
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
            # Explicit reset — retriever_node is skipped on this branch, and
            # without a reducer LangGraph would otherwise carry a previous
            # technical turn's retrieved images into this conversational reply.
            "documents": [],
            "image_sources": [],
        }
    
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
