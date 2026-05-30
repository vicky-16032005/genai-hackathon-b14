"""RAGAS-style metrics computed with local models (PS-SC4 deliverable 4).

We implement the published RAGAS algorithms rather than importing the library,
because ragas 0.4.x has an import-time incompatibility with current
langchain-community. The algorithms here are faithful to RAGAS:

  * Context Recall  = fraction of ground-truth statements that are attributable
    to the retrieved context (LLM-judged statement attribution).
  * Answer Relevance = mean cosine similarity between the user's question and
    questions reverse-generated from the answer (embedding-based).

Both use the same local Ollama models as the app, so the eval is self-contained
and reproducible offline.
"""
import json
import re
import math

from . import llm, config
from .ingest import get_embeddings

_emb = None


def _judge(prompt: str, system: str = "") -> str:
    """Run the evaluation judge (config.JUDGE_MODEL via Ollama), independent of
    the app's generator so eval scores stay credible even with a small generator."""
    import ollama
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    opts = {"temperature": 0.0, "num_predict": 700}
    try:
        r = client.chat(model=config.JUDGE_MODEL, messages=msgs, think=False, options=opts)
    except Exception:
        r = client.chat(model=config.JUDGE_MODEL, messages=msgs, options=opts)
    return llm._clean(r.message.content)


def _embed(texts: list[str]) -> list[list[float]]:
    global _emb
    if _emb is None:
        _emb = get_embeddings()
    return _emb.embed_documents(texts)


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _extract_json(text: str):
    m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line).strip()


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    """Fraction of ground-truth statements attributable to retrieved context.

    Uses line-based output ('YES: ...' / 'NO: ...') instead of nested JSON, which
    small local models emit far more reliably (nested JSON frequently malforms and
    would otherwise yield false zeros).
    """
    ctx = "\n\n".join(contexts)
    prompt = (
        "Break the REFERENCE answer into atomic factual statements. For EACH statement, "
        "output exactly one line: 'YES: <statement>' if the CONTEXT supports it, or "
        "'NO: <statement>' if the CONTEXT does not. Output ONLY these lines.\n\n"
        f"CONTEXT:\n{ctx}\n\nREFERENCE:\n{ground_truth}"
    )
    out = _judge(prompt, system="You are a strict evaluation judge.")
    yes = no = 0
    for raw in out.splitlines():
        line = _strip_bullet(raw).upper()
        if line.startswith("YES"):
            yes += 1
        elif line.startswith("NO"):
            no += 1
    total = yes + no
    return yes / total if total else 0.0


def answer_relevance(question: str, answer: str, n: int = 3) -> float:
    """Mean cosine similarity between the question and questions generated from the answer."""
    prompt = (
        f"Generate {n} different questions that the ANSWER below directly and fully answers. "
        "Output ONE question per line and nothing else.\n\n"
        f"ANSWER:\n{answer}"
    )
    out = _judge(prompt, system="You generate questions for evaluation.")
    gen = []
    for raw in out.splitlines():
        q = _strip_bullet(raw)
        if q and ("?" in q or len(q.split()) >= 3):
            gen.append(q)
    gen = gen[:n]
    if not gen:
        return 0.0
    vecs = _embed([question] + gen)
    qv, gens = vecs[0], vecs[1:]
    return sum(_cosine(qv, g) for g in gens) / len(gens)
