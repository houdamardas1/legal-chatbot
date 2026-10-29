"""Integration test of the FAISS layer with deterministic fake embeddings (no model download)."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from legal_chatbot.config import Settings

pytest.importorskip("faiss")
pytest.importorskip("langchain_community")

from legal_chatbot.rag import (
    build_vectorstore,
    load_corpus,
    load_vectorstore,
    make_retriever,
    save_vectorstore,
)
from tests.test_rag import SAMPLE


def test_build_save_load_and_retrieve(tmp_path):
    settings = Settings.from_env()
    embeddings = DeterministicFakeEmbedding(size=32)
    store = build_vectorstore(load_corpus(SAMPLE), settings, embeddings)
    save_vectorstore(store, tmp_path / "index")

    reloaded = load_vectorstore(tmp_path / "index", settings, embeddings)
    docs = make_retriever(reloaded, top_k=2).invoke("confidentiality")

    assert len(docs) == 2
    assert all(d.metadata["doc_id"].startswith("doc_") for d in docs)


def test_load_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-index"):
        load_vectorstore(tmp_path / "nope", Settings.from_env(), DeterministicFakeEmbedding(size=8))
