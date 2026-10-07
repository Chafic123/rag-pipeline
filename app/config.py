"""Application settings, all in one place.

Values come from environment variables (a .env file locally, real environment
variables in Docker), with sensible defaults. Every other module imports
`settings` from here instead of hard-coding model names, paths or keys.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains app/, data/, evals/ ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env if it exists (it does not exist inside Docker, which is fine).
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Models
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
    judge_model: str = os.getenv("JUDGE_MODEL", "gpt-4o")   # stronger than the generator

    # Data and vector store
    docs_dir: Path = Path(os.getenv("DOCS_DIR", PROJECT_ROOT / "data" / "docs"))
    qdrant_path: Path = Path(os.getenv("QDRANT_PATH", PROJECT_ROOT / "qdrant_data"))
    collection: str = os.getenv("COLLECTION", "documents")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "3"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))


settings = Settings()
