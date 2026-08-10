import os
import base64
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path, override=True)


# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=token)
    # logfire.instrument_requests() # Disabled due to OpenTelemetry bug on Windows: MeterProvider.get_meter() got multiple values for argument 'version'
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"
    


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")
    
    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warn(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")


# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input — accept_file lets a user attach a single image alongside
# their message (Phase 3, Track C); file_type restricts the browser-side
# picker to image formats, ahead of the server-side guardrail check in
# app/guardrails/image_guard.py.
if submission := st.chat_input(
    "Ask about your documentation...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "gif", "webp", "bmp"],
):
    user_text = submission.text
    uploaded_file = submission.files[0] if submission.files else None

    # START TRACE: User Interaction
    with logfire.span(
        "💬 User Chat Interaction",
        user_query=user_text,
        session_id=st.session_state.session_id,
        has_image=bool(uploaded_file),
    ):

        # v1: store a text marker rather than persisting raw image bytes in
        # session history — enough to show an image was part of the turn
        # without carrying bytes forward across Streamlit reruns.
        history_content = f"[Image attached] {user_text}" if uploaded_file else user_text
        st.session_state.messages.append({"role": "user", "content": history_content})
        with st.chat_message("user", avatar=USER_AVATAR):
            if uploaded_file:
                st.image(uploaded_file)
            st.markdown(user_text)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    # DISTRIBUTED TRACE: Calling Backend
                    with logfire.span("📡 Calling RAG Backend"):
                        # Get backend URL from env, or default to local if not set
                        base_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip()
                        if not base_url.startswith(("http://", "https://")):
                            base_url = f"http://{base_url}"
                        url = f"{base_url}/query"
                        payload = {"q": user_text, "thread_id": st.session_state.session_id}
                        if uploaded_file:
                            payload["image_base64"] = base64.b64encode(uploaded_file.getvalue()).decode("ascii")
                            payload["image_content_type"] = uploaded_file.type
                        response = requests.post(url, json=payload, timeout=60)
                        data = response.json()
                    
                    # Show Reasoning Steps from Backend
                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")
                    
                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)
                    
                    # --- SHOW SOURCES (NESTED EXPANDABLES) ---
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("📄 View Retrieved Context (Sources)"):
                            for i, source in enumerate(sources):
                                # Create a preview title for each chunk
                                preview = source[:100].replace("\n", " ") + "..."
                                with st.expander(f"Chunk {i+1}: {preview}"):
                                    st.info(source)

                    # --- SHOW RETRIEVED IMAGES ---
                    # image_path is a local filesystem path from wherever
                    # ingestion ran — works because the backend and this UI
                    # currently share a filesystem. Falls back to a caption
                    # note rather than crashing if the path isn't reachable
                    # (e.g. backend and UI split across machines later).
                    image_sources = data.get("image_sources", [])
                    if image_sources:
                        with st.expander(f"🖼️ Retrieved Images ({len(image_sources)})"):
                            for img in image_sources:
                                try:
                                    st.image(img["image_path"], caption=f"{img.get('source', '')} — {img.get('location', '')}")
                                except Exception:
                                    st.caption(f"🖼️ {img.get('source', '')} — {img.get('location', '')} (image unavailable)")
                                    st.info(img.get("caption_preview", ""))
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Offline.")
                    st.stop()

            # Final Answer Streaming
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")
            
            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)
            
            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")
