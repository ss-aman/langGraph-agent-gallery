"""
src/utils/llm_factory.py — Provider-agnostic LLM factory.

Uses LangChain's `init_chat_model` so you can swap between OpenAI,
Anthropic, Google, Ollama, etc. by changing two env vars — no code changes.

Supported providers (set LLM_PROVIDER env var):
  openai          → requires OPENAI_API_KEY
  anthropic       → requires ANTHROPIC_API_KEY
  google_genai    → requires GOOGLE_API_KEY
  ollama          → requires local Ollama server (no API key)
  azure_openai    → requires AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)


def create_llm(
    *,
    temperature: float | None = None,
    model: str | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Create a ChatModel using the configured provider.

    Args:
        temperature: Override the configured temperature.
        model:       Override the configured model name.
        provider:    Override the configured provider.
        **kwargs:    Additional kwargs forwarded to init_chat_model.

    Returns:
        A LangChain BaseChatModel (provider-agnostic interface).

    Example:
        llm = create_llm(temperature=0.7)
        response = await llm.ainvoke([HumanMessage(content="Hello")])
    """
    _provider = provider or settings.llm_provider
    _model = model or settings.llm_model
    _temp = temperature if temperature is not None else settings.llm_temperature

    logger.debug("Creating LLM: provider=%s model=%s temperature=%s",
                 _provider, _model, _temp)

    return init_chat_model(
        model=_model,
        model_provider=_provider,
        temperature=_temp,
        **kwargs,
    )


def create_structured_llm(
    schema: type[BaseModel],
    *,
    temperature: float = 0.0,
    model: str | None = None,
    provider: str | None = None,
) -> Any:
    """
    Create an LLM bound to return a specific Pydantic model as output.

    Uses `with_structured_output()` — the underlying mechanism (tool calling
    vs JSON mode) is chosen automatically by LangChain based on the provider.

    Args:
        schema:      Pydantic BaseModel class defining the expected output.
        temperature: Should usually be 0 for structured extraction.

    Returns:
        A Runnable that accepts messages and returns an instance of `schema`.

    Example:
        extractor = create_structured_llm(IntentOutput)
        result: IntentOutput = await extractor.ainvoke(messages)
    """
    llm = create_llm(temperature=temperature, model=model, provider=provider)
    return llm.with_structured_output(schema)


# ── Cached instances for performance ─────────────────────────────────────────
# These are reused across graph invocations to avoid repeated construction.
# LRU cache with maxsize=None = effectively a singleton per unique argument set.

@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return a cached LLM instance for the given temperature."""
    return create_llm(temperature=temperature)
