from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


# Common typos / aliases mapped onto the canonical provider keys.
_PROVIDER_ALIASES = {
    "anthorpic": "anthropic",
    "anthropics": "anthropic",
    "claude": "anthropic",
    "gpt": "openai",
    "open-ai": "openai",
    "oai": "openai",
    "google": "gemini",
    "google-genai": "gemini",
    "gemini-pro": "gemini",
    "open-router": "openrouter",
    "router": "openrouter",
    "local": "ollama",
    "openai-compatible": "custom",
}

SUPPORTED_PROVIDERS = {
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map aliases like ``anthorpic`` -> ``anthropic`` and lower-case the key."""

    key = (value or "").strip().lower()
    return _PROVIDER_ALIASES.get(key, key)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Imports are lazy so the offline benchmark / tests never require the
    provider SDKs to be installed.
    """

    provider = normalize_provider(config.provider)
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {config.provider!r}. "
            f"Choose one of {sorted(SUPPORTED_PROVIDERS)}."
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "custom":
        # OpenAI-compatible endpoint (vLLM, LM Studio, Together, etc.).
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider == "openrouter":
        # OpenRouter speaks the OpenAI protocol behind a fixed base URL.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Unhandled provider {provider!r}")
