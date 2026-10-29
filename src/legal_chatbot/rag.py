"""Retrieval-augmented pipeline: corpus loading, FAISS indexing, retrieval and answering."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .extraction import (
    ContractField,
    ContractInfo,
    detect_fields,
    extract_json_object,
    format_answer,
)
from .prompts import EXTRACTION_PROMPT, QA_PROMPT

logger = logging.getLogger(__name__)


class CorpusFormatError(ValueError):
    """Raised when a corpus file does not match the expected JSON schema."""


class Retriever(Protocol):
    def invoke(self, query: str) -> list[Document]: ...


class TextLLM(Protocol):
    def invoke(self, prompt: str) -> str: ...


# --------------------------------------------------------------------------- corpus


def parse_corpus(data: Any) -> list[Document]:
    """Convert ``{"documents": [{"id", "text"}]}`` (or a bare list) into LangChain documents."""
    if isinstance(data, dict) and "documents" in data:
        records = data["documents"]
    elif isinstance(data, list):
        records = data
    else:
        raise CorpusFormatError(
            'Expected a JSON list of documents or an object with a "documents" key.'
        )
    if not isinstance(records, list):
        raise CorpusFormatError('"documents" must be a list.')

    documents = []
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        doc_id = record.get("id", position)
        documents.append(
            Document(
                page_content=text.strip(),
                metadata={"doc_id": f"doc_{doc_id}", "title": record.get("title")},
            )
        )
    if not documents:
        raise CorpusFormatError("The corpus does not contain any document with a non-empty text.")
    return documents


def load_corpus(source: str | Path | IO[Any]) -> list[Document]:
    """Load a corpus from a path or an open file (e.g. a Streamlit upload)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = json.load(source)
    return parse_corpus(data)


def split_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(list(documents))
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


# --------------------------------------------------------------------------- vector store
# Heavy dependencies (torch, faiss) are imported lazily so the rest of the package stays light.


def get_embeddings(settings: Settings):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(documents: list[Document], settings: Settings, embeddings=None):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    logger.info("Embedding %d chunks from %d documents", len(chunks), len(documents))
    return FAISS.from_documents(
        chunks,
        embeddings or get_embeddings(settings),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def save_vectorstore(vectorstore, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))


def load_vectorstore(index_dir: Path, settings: Settings, embeddings=None):
    from langchain_community.vectorstores import FAISS

    if not index_dir.exists():
        raise FileNotFoundError(
            f"No FAISS index at {index_dir}. Build one first with: legal-chatbot build-index"
        )
    # Only load indexes this project created: FAISS metadata is stored with pickle.
    return FAISS.load_local(
        str(index_dir), embeddings or get_embeddings(settings), allow_dangerous_deserialization=True
    )


def make_retriever(vectorstore, top_k: int):
    """Max-marginal-relevance retrieval: relevant chunks, but not five copies of the same clause."""
    return vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": top_k, "fetch_k": max(4 * top_k, 20)}
    )


# --------------------------------------------------------------------------- answering


@dataclass
class Source:
    doc_id: str
    snippet: str


@dataclass
class Answer:
    text: str
    mode: str  # "extraction" or "qa"
    sources: list[Source] = field(default_factory=list)
    fields: list[ContractField] = field(default_factory=list)
    latency_s: float = 0.0
    raw_llm_output: str | None = None


def build_context(docs: list[Document], max_chars: int) -> str:
    """Join retrieved chunks, tagged with their document id, within a character budget."""
    parts: list[str] = []
    used = 0
    for doc in docs:
        block = f"[{doc.metadata.get('doc_id', 'inconnu')}]\n{doc.page_content}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def _sources(docs: list[Document], snippet_chars: int = 240) -> list[Source]:
    seen: dict[str, Source] = {}
    for doc in docs:
        doc_id = str(doc.metadata.get("doc_id", "inconnu"))
        if doc_id not in seen:
            snippet = " ".join(doc.page_content.split())[:snippet_chars]
            seen[doc_id] = Source(doc_id=doc_id, snippet=snippet)
    return list(seen.values())


class ContractAssistant:
    """Routes a question to field extraction or open QA, both grounded on retrieved chunks."""

    def __init__(self, retriever: Retriever, llm: TextLLM, max_context_chars: int = 6000):
        self.retriever = retriever
        self.llm = llm
        self.max_context_chars = max_context_chars

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")
        start = time.perf_counter()
        docs = self.retriever.invoke(question)
        if not docs:
            answer = Answer(
                text="Aucun passage pertinent n'a été trouvé dans les documents.", mode="qa"
            )
        else:
            fields = detect_fields(question)
            answer = self._extract(question, docs, fields) if fields else None
            if answer is None:
                answer = self._answer(question, docs)
        answer.latency_s = time.perf_counter() - start
        return answer

    def _answer(self, question: str, docs: list[Document]) -> Answer:
        context = build_context(docs, self.max_context_chars)
        output = self.llm.invoke(QA_PROMPT.format(question=question, context=context))
        return Answer(text=str(output).strip(), mode="qa", sources=_sources(docs))

    def _extract(
        self, question: str, docs: list[Document], fields: list[ContractField]
    ) -> Answer | None:
        context = build_context(docs, self.max_context_chars)
        output = str(self.llm.invoke(EXTRACTION_PROMPT.format(context=context)))
        data = extract_json_object(output)
        if data is None:
            logger.info("Extraction returned no JSON, falling back to open QA")
            return None
        info = ContractInfo.from_llm_json(data)
        if info.is_empty():
            logger.info("Extraction returned only empty fields, falling back to open QA")
            return None
        return Answer(
            text=format_answer(info, fields),
            mode="extraction",
            sources=_sources(docs),
            fields=fields,
            raw_llm_output=output,
        )
