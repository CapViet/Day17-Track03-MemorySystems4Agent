from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - paths for the repo root, dataset directory, and state directory
    - compact-memory settings (token threshold + messages to keep)
    - provider settings for the main model and the judge model
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


# Reasonable default model per provider so a bare `LLM_PROVIDER=...` still works.
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

# Which env var holds the API key for each provider.
_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}


def _load_dotenv(root: Path) -> None:
    """Best-effort `.env` loading; silently skipped if python-dotenv is absent."""

    try:
        from dotenv import load_dotenv
    except Exception:
        return
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _base_url_for(provider: str) -> str | None:
    if provider == "custom":
        return os.getenv("CUSTOM_BASE_URL")
    if provider == "ollama":
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return None


def _provider_config(provider: str, model_name: str, temperature: float) -> ProviderConfig:
    key_env = _API_KEY_ENV.get(provider)
    api_key = os.getenv(key_env) if key_env else None
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=_base_url_for(provider),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated :class:`LabConfig`."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(provider, "gpt-4o-mini"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model_name = os.getenv(
        "JUDGE_MODEL", _DEFAULT_MODELS.get(judge_provider, model_name)
    )

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "700"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=_provider_config(provider, model_name, temperature),
        judge_model=_provider_config(judge_provider, judge_model_name, 0.0),
    )
