import os
from typing import Any

from langchain_ollama import ChatOllama


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def get_ollama_base_url(prefix: str | None = None) -> str:
    if prefix:
        prefixed_base_url = os.getenv(f"{prefix}_OLLAMA_BASE_URL")
        if prefixed_base_url:
            return prefixed_base_url

    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def get_ollama_api_key(prefix: str | None = None) -> str | None:
    if prefix:
        prefixed_api_key = os.getenv(f"{prefix}_OLLAMA_API_KEY")
        if prefixed_api_key:
            return prefixed_api_key

    return os.getenv("OLLAMA_API_KEY")


def make_chat_ollama(
    *,
    model: str,
    prefix: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> ChatOllama:
    resolved_base_url = base_url or get_ollama_base_url(prefix)
    resolved_api_key = api_key or get_ollama_api_key(prefix)

    client_kwargs = dict(kwargs.pop("client_kwargs", {}) or {})
    if resolved_api_key:
        headers = dict(client_kwargs.get("headers", {}) or {})
        headers["Authorization"] = f"Bearer {resolved_api_key}"
        client_kwargs["headers"] = headers

    return ChatOllama(
        model=model,
        base_url=resolved_base_url,
        client_kwargs=client_kwargs,
        **kwargs,
    )
