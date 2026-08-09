import mimetypes

import logfire
from app.agents.state import AgentState, format_history
from app.gateway import portkey_client, extract_cache_status
from app.services.vision.answering import answer_with_vision, MAX_VISION_IMAGES


def _load_vision_images(image_sources: list[dict]) -> list[tuple[bytes, str]]:
    """
    Read raw bytes for up to MAX_VISION_IMAGES retrieved images from disk.
    Best-effort — a missing or unreadable file just drops that one image
    rather than failing the whole answer; the vision-native path degrades
    to fewer images, and the text-only fallback still has the full context
    either way.
    """
    images = []
    for src in image_sources[:MAX_VISION_IMAGES]:
        path = src.get("image_path")
        if not path:
            continue
        try:
            mime, _ = mimetypes.guess_type(path)
            with open(path, "rb") as f:
                images.append((f.read(), mime or "image/png"))
        except Exception as e:
            logfire.warning(f"🖼️ Could not read retrieved image for vision-native answer ({path}): {e}")
    return images


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    # Windowed conversation history — bounded by MAX_HISTORY_MESSAGES so
    # prompt size and token cost don't grow with total thread length.
    history_str = format_history(state["messages"][:-1])

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        # Vision-native path — triggered by a retrieved image (image_sources,
        # Track A/B) or a user-uploaded image (uploaded_image, Track C).
        # Never true on the CONVERSATIONAL branch, since both are explicitly
        # reset there — see state.py / planner.py. Best-effort: any failure
        # (quota, rate limit, unreadable file) falls through to the existing
        # text-only path below unchanged — the same fallback shape already
        # proven for captioning.
        uploaded_image = state.get("uploaded_image")
        has_uploaded_image = uploaded_image is not None

        vision_images = []
        if uploaded_image:
            # Uploaded image goes first — it's the user's direct focus, and
            # answer_with_vision() caps the combined list at
            # MAX_VISION_IMAGES, so ordering decides what gets dropped if
            # retrieval also surfaced images.
            vision_images.append((uploaded_image["data"], uploaded_image["mime_type"]))
        vision_images.extend(_load_vision_images(state.get("image_sources", [])))

        if vision_images:
            vision_answer = answer_with_vision(prompt, vision_images)
            if vision_answer:
                logfire.info(f"🖼️ Response synthesised via vision-native path ({len(vision_images)} image(s)).")
                return {
                    "final_answer": vision_answer,
                    "status": "Response generated (vision-native).",
                    "plan": state["plan"] + ["Answer: Vision-native 🖼️"],
                    "messages": [{"role": "assistant", "content": vision_answer}]
                }
            logfire.info("🖼️ Vision-native path unavailable — falling back to text-only response.")
            if has_uploaded_image:
                # The text-only fallback model has no way to see the image —
                # say so explicitly rather than let it silently ignore or
                # hallucinate about an attachment it can't perceive.
                prompt += (
                    "\n\nNOTE: The user attached an image, but it could not be analyzed right now "
                    "(temporary issue). Acknowledge this honestly, answer using only the text "
                    "context above if relevant, and ask the user to describe the image or try "
                    "again shortly."
                )

        try:
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
