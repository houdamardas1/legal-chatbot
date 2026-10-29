"""LangChain-compatible wrapper around the Groq chat completions API."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from langchain_core.language_models import LLM
from pydantic import Field, SecretStr

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Status codes worth retrying: rate limiting and transient server-side failures.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class GroqAPIError(RuntimeError):
    """Raised when the Groq API cannot produce a completion."""


class GroqLLM(LLM):
    """Minimal Groq client with timeouts and exponential-backoff retries.

    Implemented as a LangChain ``LLM`` so it plugs into any LangChain chain or runnable.
    """

    api_key: SecretStr
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 1024
    request_timeout: int = 30
    max_retries: int = 3
    backoff_base: float = 2.0
    system_prompt: str | None = None
    session: Any = Field(default=None, exclude=True)

    @classmethod
    def from_settings(cls, settings, **overrides: Any) -> GroqLLM:
        if not settings.groq_api_key:
            raise GroqAPIError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        params = {
            "api_key": settings.groq_api_key,
            "model": settings.groq_model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "request_timeout": settings.request_timeout,
            "max_retries": settings.max_retries,
        }
        params.update(overrides)
        return cls(**params)

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs: Any) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if stop:
            payload["stop"] = stop
        return self._post_with_retry(payload)

    def _post_with_retry(self, payload: dict[str, Any]) -> str:
        http = self.session or requests
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = http.post(
                    GROQ_CHAT_URL, headers=headers, json=payload, timeout=self.request_timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                logger.warning(
                    "Groq network error (attempt %d/%d): %s", attempt, self.max_retries, exc
                )
            else:
                if response.status_code in RETRYABLE_STATUS_CODES:
                    last_error = GroqAPIError(f"HTTP {response.status_code}")
                    logger.warning(
                        "Groq returned HTTP %d (attempt %d/%d)",
                        response.status_code,
                        attempt,
                        self.max_retries,
                    )
                elif response.status_code >= 400:
                    raise GroqAPIError(
                        f"Groq API error HTTP {response.status_code}: {response.text[:300]}"
                    )
                else:
                    return self._parse_completion(response.json())

            if attempt < self.max_retries:
                time.sleep(self.backoff_base**attempt)

        raise GroqAPIError(f"Groq API unavailable after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _parse_completion(body: dict[str, Any]) -> str:
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqAPIError(f"Unexpected Groq response format: {body!r}") from exc
