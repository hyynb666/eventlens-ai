"""OpenAI-compatible Chat Completions client using Python's standard library."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    """Raised when an LLM provider request fails or returns an invalid response."""


class OpenAICompatibleClient:
    """A small client for APIs implementing the OpenAI Chat Completions format."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 30,
    ) -> None:
        if not api_key.strip():
            raise ValueError("An API key is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Request a JSON-formatted chat completion and return its content."""
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise LLMError(f"The language model provider returned HTTP {error.code}.") from error
        except (URLError, TimeoutError, OSError) as error:
            raise LLMError("Could not connect to the language model provider.") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LLMError("The language model provider returned an invalid response.") from error

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("The language model provider returned an unexpected response.") from error
        if not isinstance(content, str):
            raise LLMError("The language model provider returned no text content.")
        return content


def client_from_environment() -> OpenAICompatibleClient | None:
    """Create a provider client when OPENAI_API_KEY is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAICompatibleClient(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        or "https://api.openai.com/v1",
    )
