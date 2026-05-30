"""Build a synthetic retriever-training set: citizen query -> scheme passage.

For each of the 20 schemes we ask the (still-local) LLM for several natural
citizen questions that the scheme answers, and pair each with that scheme's
Overview passage as the positive. In-batch negatives are supplied by the loss
during training.

IMPORTANT (honest eval): the 20 evaluation personas in data/personas.json are
NOT used here, so the before/after retrieval comparison measures generalisation
to unseen phrasings, not memorisation.
"""
import json
import re
from pathlib import Path

from src import config, llm

OUT = Path(config.ROOT) / "train" / "pairs.jsonl"
QUERIES_PER_SCHEME = 6


def overview_passage(s: dict) -> str:
    return (f"Scheme: {s['scheme_name']} | Category: {s.get('category','')} | Section: Overview\n"
            f"{s.get('summary','')}\n\nBenefits: {s.get('benefits','')}")


def gen_queries(s: dict, n: int) -> list[str]:
    prompt = (
        f"Scheme name: {s['scheme_name']}\n"
        f"Summary: {s.get('summary','')}\n"
        f"Eligibility: {s.get('eligibility_text','')}\n\n"
        f"Write {n} short, natural questions a citizen might ask that THIS scheme answers. "
        "Vary the phrasing (some about eligibility, some about benefits, some about how to "
        "apply). Do not name the scheme in the question. Output one question per line, nothing else."
    )
    out = llm.chat(prompt, system="You generate realistic citizen questions for a government-scheme assistant.")
    qs = []
    for line in out.splitlines():
        q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if len(q.split()) >= 3 and "?" in q:
            qs.append(q)
    return qs[:n]


def main():
    schemes = json.loads(Path(config.SCHEMES_JSON).read_text())["schemes"]
    pairs = []
    for i, s in enumerate(schemes, 1):
        pos = overview_passage(s)
        qs = gen_queries(s, QUERIES_PER_SCHEME)
        for q in qs:
            pairs.append({"query": q, "positive": pos, "scheme_id": s["scheme_id"]})
        print(f"[{i:>2}/{len(schemes)}] {s['scheme_id']:16s} -> {len(qs)} queries")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(pairs)} (query, passage) pairs -> {OUT}")


if __name__ == "__main__":
    main()
