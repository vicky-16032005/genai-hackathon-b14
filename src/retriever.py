"""Load the FAISS index and retrieve scheme passages.

Supports FAISS metadata filtering (PS-SC4 spec: "FAISS with metadata filter").
Also offers flexible state-aware retrieval: a citizen in Karnataka should match
both Karnataka-specific schemes AND All-India schemes.
"""
from functools import lru_cache

from langchain_community.vectorstores import FAISS

from . import config
from .ingest import get_embeddings


@lru_cache(maxsize=1)
def load_vectorstore() -> FAISS:
    return FAISS.load_local(
        str(config.INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


def retrieve(query: str, k: int = config.TOP_K, metadata_filter: dict | None = None):
    """Return top-k (Document, score) pairs, optionally filtered by metadata."""
    vs = load_vectorstore()
    return vs.similarity_search_with_score(query, k=k, filter=metadata_filter)


def retrieve_for_state(query: str, state: str | None = None, k: int = config.TOP_K):
    """State-aware retrieval: keep All-India schemes plus the user's state."""
    docs = retrieve(query, k=k * 3 if state else k)
    if not state or state == "All India":
        return docs[:k]
    keep = []
    for doc, score in docs:
        states = doc.metadata.get("applicable_states", "")
        if "All India" in states or state in states or states == "":
            keep.append((doc, score))
    return (keep or docs)[:k]
