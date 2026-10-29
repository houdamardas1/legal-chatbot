import pytest
from langchain_core.documents import Document


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return list(self.docs)


class FakeLLM:
    """Returns canned responses in order and records the prompts it received."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


@pytest.fixture
def contract_docs():
    return [
        Document(
            page_content="This Service Agreement is entered into on 15 March 2023 by Northwind.",
            metadata={"doc_id": "doc_1"},
        ),
        Document(
            page_content="The Client shall pay a total fee of 48,000 EUR.",
            metadata={"doc_id": "doc_1"},
        ),
        Document(page_content="Confidentiality lasts five years.", metadata={"doc_id": "doc_2"}),
    ]
