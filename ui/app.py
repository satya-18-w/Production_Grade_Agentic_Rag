import os
import base64
import streamlit as st
import requests
import time
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


# --- BACKEND URL ---
def _backend_url() -> str:
    base_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip()
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    return base_url


# --- AUTH STATE ---
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
    st.session_state.username = None
    st.session_state.user_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.auth_token}"}


def _logout():
    logfire.info(f"👋 User logged out: {st.session_state.username}")
    st.session_state.auth_token = None
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.thread_id = None
    st.session_state.messages = []


def _new_thread(title: str | None = None) -> bool:
    """Asks the backend for a fresh thread_id owned by the logged-in user.
    Returns False (and surfaces the error) if the token has expired."""
    try:
        r = requests.post(
            f"{_backend_url()}/auth/threads",
            json={"title": title},
            headers=_auth_headers(),
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Backend Offline: {e}")
        return False

    if r.status_code == 401:
        st.session_state.auth_token = None
        st.error("Your session expired. Please log in again.")
        return False
    if r.status_code != 200:
        st.error(f"Could not start a new chat: {r.status_code} - {r.text}")
        return False

    st.session_state.thread_id = r.json()["thread_id"]
    st.session_state.messages = []
    return True


# --- LOGIN / SIGNUP GATE ---
# Everything below the auth check requires a valid bearer token — /query
# and /auth/threads on the backend both reject unauthenticated requests.
if not st.session_state.auth_token:
    st.title("🤖 Enterprise Agentic Assistant")
    st.subheader("Sign in to continue")

    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            try:
                r = requests.post(
                    f"{_backend_url()}/auth/login",
                    json={"username": username, "password": password},
                    timeout=15,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Backend Offline: {e}")
                r = None
            if r is not None:
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.auth_token = data["token"]
                    st.session_state.username = data["username"]
                    st.session_state.user_id = data["user_id"]
                    logfire.info(f"✅ Login: {data['username']}")
                    st.rerun()
                else:
                    st.error(r.json().get("detail", "Login failed."))

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Username", key="signup_username")
            new_password = st.text_input(
                "Password (min 8 characters)", type="password", key="signup_password"
            )
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            try:
                r = requests.post(
                    f"{_backend_url()}/auth/signup",
                    json={"username": new_username, "password": new_password},
                    timeout=15,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Backend Offline: {e}")
                r = None
            if r is not None:
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.auth_token = data["token"]
                    st.session_state.username = data["username"]
                    st.session_state.user_id = data["user_id"]
                    logfire.info(f"✨ New account: {data['username']}")
                    st.rerun()
                else:
                    st.error(r.json().get("detail", "Sign up failed."))

    st.stop()

# A thread must exist before the chat below can call /query.
if not st.session_state.thread_id:
    if not _new_thread():
        st.stop()


# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Logged in as **{st.session_state.username}**")
    st.caption(f"Thread: {st.session_state.thread_id[:8]}")

    if st.button("🗑️ New Chat", width="stretch", type="primary"):
        if _new_thread():
            st.rerun()

    if st.button("Log out", width="stretch"):
        _logout()
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
        thread_id=st.session_state.thread_id,
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
                        url = f"{_backend_url()}/query"
                        payload = {"q": user_text, "thread_id": st.session_state.thread_id}
                        if uploaded_file:
                            payload["image_base64"] = base64.b64encode(uploaded_file.getvalue()).decode("ascii")
                            payload["image_content_type"] = uploaded_file.type
                        response = requests.post(url, json=payload, headers=_auth_headers(), timeout=60)
                        if response.status_code == 401:
                            _logout()
                            st.error("Your session expired. Please log in again.")
                            st.rerun()
                        if response.status_code != 200:
                            st.error(f"Backend Error: {response.status_code} - {response.text}")
                            st.stop()
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
