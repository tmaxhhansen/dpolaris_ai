"""
LLM provider abstraction for dPolaris AI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class LLMProviderError(RuntimeError):
    """Base provider error."""


class LLMUnavailableError(LLMProviderError):
    """Raised when the configured provider is unavailable."""


class LLMProvider(ABC):
    """Provider interface for text completion and streaming."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def enabled(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def disabled_reason(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> AsyncIterator[str]:
        raise NotImplementedError


class NullProvider(LLMProvider):
    """Disabled provider used when no LLM backend is configured."""

    def __init__(self, reason: str = "LLM provider is disabled"):
        self._reason = reason

    @property
    def name(self) -> str:
        return "none"

    @property
    def enabled(self) -> bool:
        return False

    @property
    def disabled_reason(self) -> str:
        return self._reason

    async def complete(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> str:
        return self._reason

    async def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> AsyncIterator[str]:
        yield self._reason


class AnthropicProvider(LLMProvider):
    """Anthropic-backed provider with lazy anthropic import."""

    def __init__(self, api_key: str):
        key = (api_key or "").strip()
        if not key:
            raise LLMUnavailableError(
                "LLM provider 'anthropic' requires ANTHROPIC_API_KEY."
            )
        try:
            import anthropic  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise LLMUnavailableError(
                "LLM provider 'anthropic' is unavailable because the anthropic package is not installed."
            ) from exc

        self._client = anthropic.Anthropic(api_key=key)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def enabled(self) -> bool:
        return True

    @property
    def disabled_reason(self) -> str:
        return ""

    async def complete(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
        )
        return response.content[0].text

    async def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> AsyncIterator[str]:
        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text

