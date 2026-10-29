from unittest.mock import MagicMock

import pytest
import requests

from legal_chatbot.config import Settings
from legal_chatbot.llm import GROQ_CHAT_URL, GroqAPIError, GroqLLM


def _response(status=200, body=None, text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body if body is not None else {}
    resp.text = text
    return resp


def _ok(content="Bonjour"):
    return _response(body={"choices": [{"message": {"content": content}}]})


@pytest.fixture
def session():
    return MagicMock()


@pytest.fixture
def llm(session):
    return GroqLLM(api_key="test-key", session=session, max_retries=3, backoff_base=0)


def test_successful_call_sends_expected_payload(llm, session):
    session.post.return_value = _ok("Réponse")
    llm.system_prompt = "Tu es juriste."

    assert llm.invoke("Question ?") == "Réponse"

    url = session.post.call_args.args[0]
    kwargs = session.post.call_args.kwargs
    assert url == GROQ_CHAT_URL
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["messages"] == [
        {"role": "system", "content": "Tu es juriste."},
        {"role": "user", "content": "Question ?"},
    ]
    assert kwargs["timeout"] == llm.request_timeout


@pytest.mark.parametrize("status", [429, 503])
def test_retries_transient_errors(llm, session, status):
    session.post.side_effect = [_response(status), _response(status), _ok("ok")]
    assert llm.invoke("q") == "ok"
    assert session.post.call_count == 3


def test_retries_network_errors(llm, session):
    session.post.side_effect = [requests.ConnectionError("down"), _ok("ok")]
    assert llm.invoke("q") == "ok"


def test_gives_up_after_max_retries(llm, session):
    session.post.return_value = _response(503)
    with pytest.raises(GroqAPIError, match="after 3 attempts"):
        llm.invoke("q")
    assert session.post.call_count == 3


def test_client_errors_are_not_retried(llm, session):
    session.post.return_value = _response(401, text="invalid api key")
    with pytest.raises(GroqAPIError, match="401"):
        llm.invoke("q")
    assert session.post.call_count == 1


def test_unexpected_body(llm, session):
    session.post.return_value = _response(body={"choices": []})
    with pytest.raises(GroqAPIError, match="Unexpected"):
        llm.invoke("q")


def test_api_key_is_not_leaked_in_repr(llm):
    assert "test-key" not in repr(llm)


def test_from_settings_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqAPIError, match="GROQ_API_KEY"):
        GroqLLM.from_settings(Settings.from_env())


def test_from_settings_reads_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_MODEL", "custom-model")
    llm = GroqLLM.from_settings(Settings.from_env())
    assert llm.model == "custom-model"
