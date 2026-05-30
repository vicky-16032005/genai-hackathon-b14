"""English <-> Kannada translation layer.

Primary: deep-translator (Google) — fast and high quality for Kannada.
Fallback: the local Qwen LLM (works fully offline if the venue Wi-Fi drops).
Results are cached so repeated demo queries are instant.
"""
from functools import lru_cache

from . import llm


def _google(text: str, source: str, target: str) -> str:
    from deep_translator import GoogleTranslator
    # Google caps each call ~5000 chars; chunk on sentence boundaries.
    if len(text) <= 4500:
        return GoogleTranslator(source=source, target=target).translate(text)
    parts, buf = [], ""
    for sent in text.replace("\n", " \n").split(". "):
        if len(buf) + len(sent) > 4500:
            parts.append(GoogleTranslator(source=source, target=target).translate(buf))
            buf = ""
        buf += sent + ". "
    if buf.strip():
        parts.append(GoogleTranslator(source=source, target=target).translate(buf))
    return " ".join(parts)


def _llm_translate(text: str, target_name: str) -> str:
    return llm.chat(
        f"Translate the following text to {target_name}. "
        f"Output ONLY the translation, no notes.\n\n{text}",
        system="You are a precise translator.",
    )


@lru_cache(maxsize=512)
def translate(text: str, target: str = "kn", source: str = "en") -> str:
    """Translate `text`. target/source are ISO codes ('en','kn'). Safe — never raises."""
    if not text or not text.strip() or target == source:
        return text
    try:
        out = _google(text, source, target)
        if out and out.strip():
            return out
    except Exception:
        pass
    try:
        names = {"kn": "Kannada", "en": "English"}
        return _llm_translate(text, names.get(target, target))
    except Exception:
        return text  # last resort: return original rather than crash the demo


def to_english(text: str) -> str:
    return translate(text, target="en", source="kn")


def to_kannada(text: str) -> str:
    return translate(text, target="kn", source="en")
