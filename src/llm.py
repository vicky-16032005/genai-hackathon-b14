"""Pluggable generation backend.

  LLM_PROVIDER=ollama       -> fast local small model (default; great for the demo)
  LLM_PROVIDER=transformers -> self-contained HF model, deploys to HF Spaces /
                               Streamlit Cloud with no Ollama daemon.

Both return clean text (any stray <think> blocks are stripped).
"""
import re
from functools import lru_cache

from . import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


# --------------------------------------------------------------------------- #
#  Ollama backend
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _ollama_client():
    import ollama
    return ollama.Client(host=config.OLLAMA_BASE_URL)


def _ollama_chat(prompt: str, system: str) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    opts = {"temperature": config.TEMPERATURE, "num_predict": config.NUM_PREDICT}
    client = _ollama_client()
    try:  # think=False helps qwen3-family; harmless to attempt, retry without on error
        return _clean(client.chat(model=config.GEN_MODEL, messages=msgs,
                                  think=False, options=opts).message.content)
    except Exception:
        return _clean(client.chat(model=config.GEN_MODEL, messages=msgs,
                                  options=opts).message.content)


# --------------------------------------------------------------------------- #
#  Transformers backend (deployable, no Ollama)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.HF_GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.HF_GEN_MODEL,
        torch_dtype=(torch.float16 if torch.backends.mps.is_available()
                     or torch.cuda.is_available() else torch.float32),
    )
    if torch.backends.mps.is_available():
        model = model.to("mps")
    elif torch.cuda.is_available():
        model = model.to("cuda")
    return tok, model


def _hf_chat(prompt: str, system: str) -> str:
    tok, model = _hf()
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=config.NUM_PREDICT,
                         do_sample=config.TEMPERATURE > 0, temperature=max(config.TEMPERATURE, 0.01),
                         pad_token_id=tok.eos_token_id)
    return _clean(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


# --------------------------------------------------------------------------- #
#  Groq backend (free, fast hosted API — OpenAI-compatible)
# --------------------------------------------------------------------------- #
def _groq_chat(prompt: str, system: str) -> str:
    import requests
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    resp = requests.post(
        f"{config.GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": config.GROQ_MODEL, "messages": msgs,
              "temperature": config.TEMPERATURE, "max_tokens": config.NUM_PREDICT},
        timeout=60,
    )
    resp.raise_for_status()
    return _clean(resp.json()["choices"][0]["message"]["content"])


def chat(prompt: str, system: str = "") -> str:
    """Single-turn chat via the configured provider. Returns clean assistant text."""
    if config.LLM_PROVIDER == "groq":
        return _groq_chat(prompt, system)
    if config.LLM_PROVIDER == "transformers":
        return _hf_chat(prompt, system)
    return _ollama_chat(prompt, system)
