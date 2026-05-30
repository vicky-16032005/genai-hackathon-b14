"""Honest, judge-independent retriever comparison on the 20 held-out personas.

Compares three embedders on the SAME 60 scheme passages:
  * Ollama nomic-embed-text       (previous default)
  * MiniLM base (un-fine-tuned)   (sentence-transformers/all-MiniLM-L6-v2)
  * MiniLM fine-tuned             (models/scheme-retriever — our trained model)

Metrics: Hit-rate@5 (did a passage from the expected scheme appear in top-5)
and MRR (mean reciprocal rank of the first correct passage). Deterministic —
no LLM judge — so the numbers reflect retrieval quality only.
"""
import json
from pathlib import Path

import numpy as np

from src import config
from src.ingest import _scheme_documents

K = config.TOP_K


def _norm(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def evaluate(name, embed_passages, embed_queries, docs, personas):
    P = _norm(np.array(embed_passages))
    sids = [d.metadata["scheme_id"] for d in docs]
    hits, rr = 0, []
    for p in personas:
        q = _norm(np.array([embed_queries[p["id"]]]))[0]
        order = np.argsort(-(P @ q))
        ranked_sids = [sids[i] for i in order]
        exp = set(p["expected_schemes"])
        top = ranked_sids[:K]
        hits += 1 if exp & set(top) else 0
        rank = next((i + 1 for i, s in enumerate(ranked_sids) if s in exp), None)
        rr.append(1.0 / rank if rank else 0.0)
    n = len(personas)
    return name, hits / n, sum(rr) / n


def main():
    docs = _scheme_documents()
    passages = [d.page_content for d in docs]
    personas = json.loads(Path(config.PERSONAS_JSON).read_text())["personas"]
    queries = {p["id"]: p["query"] for p in personas}
    rows = []

    # 1) Ollama nomic
    try:
        from langchain_ollama import OllamaEmbeddings
        emb = OllamaEmbeddings(model="nomic-embed-text", base_url=config.OLLAMA_BASE_URL)
        rows.append(evaluate("Ollama nomic-embed-text", emb.embed_documents(passages),
                             {i: emb.embed_query(q) for i, q in queries.items()}, docs, personas))
    except Exception as e:
        print("nomic skipped:", e)

    # 2) MiniLM base & 3) fine-tuned
    from sentence_transformers import SentenceTransformer
    for name, path in [("MiniLM base", config.ST_EMBED_BASE),
                       ("MiniLM fine-tuned (ours)", config.ST_EMBED_MODEL)]:
        st = SentenceTransformer(path)
        rows.append(evaluate(name, st.encode(passages, normalize_embeddings=False),
                             {i: st.encode(q) for i, q in queries.items()}, docs, personas))

    print("\n" + "=" * 64)
    print(f"{'Retriever':<32} {'Hit@'+str(K):>8} {'MRR':>8}")
    print("-" * 64)
    for name, hit, mrr in rows:
        print(f"{name:<32} {hit:>8.3f} {mrr:>8.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
