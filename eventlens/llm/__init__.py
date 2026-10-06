"""Small language-model interface used to interpret analytics questions."""

from eventlens.llm.base import LLMClient
from eventlens.llm.provider import LLMError, OpenAICompatibleClient, client_from_environment

__all__ = ["LLMClient", "LLMError", "OpenAICompatibleClient", "client_from_environment"]
