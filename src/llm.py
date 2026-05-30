"""Thin wrapper over the local Ollama chat model.

We call the `ollama` client directly (not langchain's ChatOllama) because the
Qwen3-family models emit a separate reasoning field that langchain's wrapper
currently drops, returning empty content. Calling Ollama with think=False gives
clean, fast, grounded answers. Embeddings still go through langchain (they work).
"""
import re
from functools import lru_cache

import ollama

from . import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@lru_cache(maxsize=1)
def get_client() -> "ollama.Client":
    return ollama.Client(host=config.OLLAMA_BASE_URL)


def _clean(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def chat(prompt: str, system: str = "") -> str:
    """Single-turn chat. Returns clean assistant text (reasoning disabled)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = get_client().chat(
            model=config.GEN_MODEL,
            messages=messages,
            think=False,
            options={"temperature": config.TEMPERATURE, "num_predict": config.NUM_PREDICT},
        )
        return _clean(resp.message.content)
    except TypeError:
        # older ollama client without think kwarg
        resp = get_client().chat(
            model=config.GEN_MODEL, messages=messages,
            options={"temperature": config.TEMPERATURE, "num_predict": config.NUM_PREDICT},
        )
        return _clean(resp.message.content)
