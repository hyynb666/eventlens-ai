"""Provider-neutral LLM interface."""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """The minimal completion method needed by the question interpreter."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text response for the supplied prompts."""
        ...
