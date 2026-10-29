import io
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from legal_chatbot.rag import (
    ContractAssistant,
    CorpusFormatError,
    build_context,
    load_corpus,
    parse_corpus,
    split_documents,
)
from tests.conftest import FakeLLM, FakeRetriever

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_contracts.json"


class TestCorpus:
    def test_sample_corpus_loads(self):
        docs = load_corpus(SAMPLE)
        assert len(docs) == 3
        assert docs[0].metadata["doc_id"] == "doc_1"

    def test_accepts_bare_list_and_file_objects(self):
        raw = json.dumps([{"id": "a", "text": "hello"}, {"id": "b", "text": "  "}])
        docs = load_corpus(io.StringIO(raw))
        assert [d.metadata["doc_id"] for d in docs] == ["doc_a"]

    @pytest.mark.parametrize(
        "data", [{"foo": []}, "text", {"documents": "x"}, {"documents": [{"text": ""}]}]
    )
    def test_rejects_invalid_corpora(self, data):
        with pytest.raises(CorpusFormatError):
            parse_corpus(data)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path / "missing.json")

    def test_split_keeps_metadata_and_respects_size(self):
        docs = [Document(page_content="mot " * 2000, metadata={"doc_id": "doc_9"})]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 500 for c in chunks)
        assert all(c.metadata["doc_id"] == "doc_9" for c in chunks)


def test_build_context_tags_sources_and_respects_budget(contract_docs):
    context = build_context(contract_docs, max_chars=10_000)
    assert context.startswith("[doc_1]\n")
    assert "[doc_2]" in context
    assert len(build_context(contract_docs, max_chars=120)) <= 120


class TestContractAssistant:
    def test_open_question_uses_qa_prompt(self, contract_docs):
        llm = FakeLLM("La confidentialité dure cinq ans [doc_2].")
        answer = ContractAssistant(FakeRetriever(contract_docs), llm).ask(
            "Que dit la clause de confidentialité ?"
        )

        assert answer.mode == "qa"
        assert answer.text == "La confidentialité dure cinq ans [doc_2]."
        assert [s.doc_id for s in answer.sources] == ["doc_1", "doc_2"]  # de-duplicated
        assert "Que dit la clause de confidentialité ?" in llm.prompts[0]

    def test_field_question_uses_structured_extraction(self, contract_docs):
        llm = FakeLLM(
            '```json\n{"nom_client": "Northwind Analytics SARL", "montant": "48 000 EUR"}\n```'
        )
        answer = ContractAssistant(FakeRetriever(contract_docs), llm).ask("Quel est le montant ?")

        assert answer.mode == "extraction"
        assert answer.text == "Montant : 48 000 EUR."
        assert len(llm.prompts) == 1

    def test_falls_back_to_qa_when_extraction_is_not_json(self, contract_docs):
        llm = FakeLLM("désolé, pas de JSON", "Le client est Northwind.")
        answer = ContractAssistant(FakeRetriever(contract_docs), llm).ask(
            "Quel est le nom du client ?"
        )

        assert answer.mode == "qa"
        assert answer.text == "Le client est Northwind."
        assert len(llm.prompts) == 2

    def test_falls_back_to_qa_when_extraction_is_empty(self, contract_docs):
        llm = FakeLLM('{"nom_client": "_____"}', "Je ne trouve pas cette information.")
        answer = ContractAssistant(FakeRetriever(contract_docs), llm).ask(
            "Quel est le nom du client ?"
        )
        assert answer.mode == "qa"

    def test_no_documents(self):
        answer = ContractAssistant(FakeRetriever([]), FakeLLM()).ask("Quel est le montant ?")
        assert "Aucun passage" in answer.text

    def test_empty_question(self, contract_docs):
        with pytest.raises(ValueError):
            ContractAssistant(FakeRetriever(contract_docs), FakeLLM()).ask("   ")
