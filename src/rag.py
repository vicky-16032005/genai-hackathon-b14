"""RAG pipeline: retrieve scheme passages -> grounded, citation-accurate answer.

Returns the answer plus a deduplicated citation list (scheme name, section,
official link, source) so the UI can render clickable, verifiable citations —
the core PS-SC4 requirement of "instant, citation-accurate answers".
"""
from . import config, llm, retriever, translate

SYSTEM = (
    "You are the Smart City Government Scheme assistant. Answer ONLY from the "
    "provided context about Indian government welfare schemes. For every claim, "
    "cite the scheme using its [n] marker. When relevant, state the scheme name, "
    "eligibility, documents needed, and the official application link. "
    "If the answer is not in the context, say you don't have that information and "
    "suggest the closest scheme present. Be concise and factual. Never invent schemes, "
    "figures, or links."
)


def _format_context(hits) -> tuple[str, list[dict]]:
    blocks, citations, seen = [], [], set()
    for i, (doc, _score) in enumerate(hits, 1):
        m = doc.metadata
        blocks.append(f"[{i}] {doc.page_content}")
        key = (m.get("scheme_name"), m.get("section"))
        if key not in seen:
            seen.add(key)
            citations.append({
                "n": i,
                "scheme_name": m.get("scheme_name", ""),
                "section": m.get("section", ""),
                "application_link": m.get("application_link", ""),
                "source": m.get("source", ""),
                "category": m.get("category", ""),
            })
    return "\n\n".join(blocks), citations


def answer_query(query: str, lang: str = "en", state: str | None = None,
                 k: int = config.TOP_K) -> dict:
    """Run RAG. If lang=='kn', retrieve in English then translate the answer."""
    search_query = translate.to_english(query) if lang == "kn" else query

    hits = retriever.retrieve_for_state(search_query, state=state, k=k)
    if not hits:
        msg = "No relevant scheme documents were found in the knowledge base."
        return {"answer": translate.to_kannada(msg) if lang == "kn" else msg,
                "answer_en": msg, "citations": [], "contexts": []}

    context, citations = _format_context(hits)
    prompt = (
        f"Context (numbered scheme passages):\n{context}\n\n"
        f"Question: {search_query}\n\n"
        "Answer using only the context above, citing [n] markers."
    )
    answer_en = llm.chat(prompt, system=SYSTEM)
    answer = translate.to_kannada(answer_en) if lang == "kn" else answer_en

    return {
        "answer": answer,
        "answer_en": answer_en,
        "citations": citations,
        "contexts": [d.page_content for d, _ in hits],
    }
