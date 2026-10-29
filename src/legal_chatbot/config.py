"""Application settings, read once from environment variables (and a local ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """All tunable parameters of the pipeline. Every field can be overridden by an env var."""

    groq_api_key: str | None
    groq_model: str
    temperature: float
    max_tokens: int
    request_timeout: int
    max_retries: int

    corpus_path: Path
    index_dir: Path
    embedding_model: str
    # all-MiniLM-L6-v2 truncates inputs at 256 word pieces, so chunks are kept small enough
    # to be embedded in full instead of being silently cut.
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_chars: int

    translation_model: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=_env_float("LLM_TEMPERATURE", 0.1),
            max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
            request_timeout=_env_int("LLM_TIMEOUT_SECONDS", 30),
            max_retries=_env_int("LLM_MAX_RETRIES", 3),
            corpus_path=Path(os.getenv("CUAD_JSON_PATH", "data/sample_contracts.json")),
            index_dir=Path(os.getenv("INDEX_DIR", "index_cuad.faiss")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=_env_int("CHUNK_SIZE", 1000),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
            top_k=_env_int("RETRIEVER_TOP_K", 5),
            max_context_chars=_env_int("MAX_CONTEXT_CHARS", 6000),
            translation_model=os.getenv("TRANSLATION_MODEL", "facebook/m2m100_418M"),
        )
