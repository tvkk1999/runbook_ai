import os
from pathlib import Path

# === Directory Paths ===
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
IMAGES_DIR = UPLOADS_DIR / "images"

# Create the directory paths if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Ensure directories exist
for directory in [DATA_DIR, UPLOADS_DIR, EMBEDDINGS_DIR, IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# === Ollama LLM Configuration ===
GEMINI_API_KEY = "your_api_key_here"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# === Ollama LLM Configuration ===
OLLAMA_API_URL = "http://localhost:11434"  # Ollama local server endpoint
DEFAULT_MODEL_NAME = "llama3.1:8b"

# LLM call parameters for generation
LLM_PARAMS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 512,
    "stream": False,
}

# === Vector Store Configuration ===
VECTOR_STORE_COLLECTION_NAME = "runbook_docs"

# Embedding model name for sentence-transformers
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === Guardrails Configuration ===
# Example: max query length allowed (reflects guardrails.py limit)
MAX_QUERY_LENGTH = 2000

# Potentially add lists of harmful keywords or regex patterns here if used elsewhere


# === Logging & Debugging options ===
LOG_LEVEL = os.getenv("RUNBOOK_AI_LOG_LEVEL", "INFO")

# === Other Global Settings ===
# Any other config constants used app-wide, e.g., cache sizes, API keys (none here currently)