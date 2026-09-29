import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GOOGLEGEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # --- GEMINI VISION (multimodal image captioning — primary) ---
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")

    # --- LOCAL VISION FALLBACK (zero-cost, used when Gemini fails/quota exhausted) ---
    SMOLVLM_MODEL_ID = os.getenv("SMOLVLM_MODEL_ID", "HuggingFaceTB/SmolVLM-Instruct")

    # --- MULTIMODAL INGESTION ---
    # Kubernetes docs sourced from web articles (Medium, vendor docs) tend to
    # keep images as external links rather than embedding them — fetching is
    # on by default so those diagrams are actually captured. Set to "false"
    # to keep ingestion strictly local (embedded/local images only).
    ALLOW_EXTERNAL_IMAGE_FETCH = os.getenv("ALLOW_EXTERNAL_IMAGE_FETCH", "true").lower() == "true"

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("LLM_API_KEY"))
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_SLUG = os.getenv("GROQ_SLUG", "rag")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2", "brag")

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_USE_GATEWAY_CONFIG = os.getenv(
        "PORTKEY_USE_GATEWAY_CONFIG", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip() or "groq"

    # Saved Portkey config slugs (e.g. "pc-xxxxx"). When the Portkey account
    # has "block_inline_config" enabled, sending the fallback/cache/retry
    # config as an inline JSON dict is rejected — a saved config referenced
    # by slug must be used instead. Leave unset to keep sending the inline
    # dict (only works while block_inline_config is off on the account).
    PORTKEY_GATEWAY_CONFIG_SLUG = os.getenv("PORTKEY_GATEWAY_CONFIG_SLUG")
    PORTKEY_GUARDRAIL_CONFIG_SLUG = os.getenv("PORTKEY_GUARDRAIL_CONFIG_SLUG")

    # --- POSTGRES CHECKPOINTER (LangGraph persistent memory) ---
    POSTGRES_URI = os.getenv("POSTGRES_URI")

    # --- AUTH (JWT bearer tokens, app/auth/) ---
    AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
    AUTH_TOKEN_EXPIRE_MINUTES = int(os.getenv("AUTH_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))  # 7 days

    # --- API HARDENING ---
    # Comma-separated list of allowed browser origins for CORS. "*" is the
    # permissive default so local/dev setups work out of the box — restrict
    # this to your real UI origin(s) for a public deployment.
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", os.getenv("CORS_ORIGINS", "*"))

    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    @staticmethod
    def validate() -> None:
        """Fail fast at startup instead of deep inside a request. Mirrors
        the RuntimeError pattern app/agents/graph.py already uses for
        POSTGRES_URI, extended to every other required credential."""
        required = {
            "PORTKEY_API_KEY": Settings.PORTKEY_API_KEY,
            "QDRANT_API_KEY": Settings.QDRANT_API_KEY,
            "QDRANT_CLUSTER_ENDPOINT": Settings.QDRANT_URL,
            "GOOGLEGEMINI_API_KEY": Settings.GEMINI_API_KEY,
            "POSTGRES_URI": Settings.POSTGRES_URI,
            "AUTH_SECRET_KEY": Settings.AUTH_SECRET_KEY,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                f"{', '.join(missing)}. Set them in your .env file."
            )

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
PORTKEY_GATEWAY_CONFIG_SLUG = os.getenv("PORTKEY_GATEWAY_CONFIG_SLUG")
PORTKEY_GUARDRAIL_CONFIG_SLUG = os.getenv("PORTKEY_GUARDRAIL_CONFIG_SLUG")
settings = Settings()
settings.validate()
