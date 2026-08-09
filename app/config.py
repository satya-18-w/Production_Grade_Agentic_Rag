import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GOOGLEGEMINI_API_KEY")

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
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "RAG1"     # primary: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = "RAG"  # fallback: @brag/llama-3.1-8b-instant

    # --- POSTGRES CHECKPOINTER (LangGraph persistent memory) ---
    POSTGRES_URI = os.getenv("POSTGRES_URI")

    
    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()