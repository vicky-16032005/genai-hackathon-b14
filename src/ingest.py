"""Ingestion pipeline: scheme corpus (+ optional PDFs) -> chunks -> FAISS.

Each scheme becomes a few *section-labelled* documents (Overview / Eligibility /
Application) so retrieval returns the precise passage and every answer can cite
`Scheme name — Section`. Scheme-name metadata is attached to every chunk, which
is exactly the PS-SC4 deliverable: "ingest ... with scheme-name metadata".
"""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)


def _scheme_documents() -> list[Document]:
    data = json.loads(Path(config.SCHEMES_JSON).read_text())
    docs: list[Document] = []
    for s in data["schemes"]:
        base_meta = {
            "scheme_id": s["scheme_id"],
            "scheme_name": s["scheme_name"],
            "category": s.get("category", ""),
            "level": s.get("level", ""),
            "applicable_states": ", ".join(s.get("applicable_states", [])),
            "application_link": s.get("application_link", ""),
            "source": s.get("source", ""),
        }
        sections = {
            "Overview": f"{s.get('summary','')}\n\nBenefits: {s.get('benefits','')}",
            "Eligibility": s.get("eligibility_text", ""),
            "Application & Documents": (
                f"Documents needed: {', '.join(s.get('documents_needed', []))}.\n"
                f"How to apply: {s.get('how_to_apply','')}\n"
                f"Official application link: {s.get('application_link','')}"
            ),
        }
        for section, body in sections.items():
            if not body.strip():
                continue
            header = f"Scheme: {s['scheme_name']} | Category: {s.get('category','')} | Section: {section}\n"
            meta = dict(base_meta, section=section)
            docs.append(Document(page_content=header + body, metadata=meta))
    return docs


def _pdf_documents() -> list[Document]:
    """Optional: ingest any real scheme PDFs dropped into data/pdfs/."""
    pdf_dir = Path(config.PDF_DIR)
    if not pdf_dir.exists():
        return []
    from pypdf import PdfReader
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    docs: list[Document] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        name = pdf.stem.replace("_", " ")
        try:
            reader = PdfReader(str(pdf))
        except Exception as e:
            print(f"  ! skipped {pdf.name}: {e}")
            continue
        for pno, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            for chunk in splitter.split_text(text):
                docs.append(Document(
                    page_content=f"Scheme: {name} | Section: PDF page {pno}\n{chunk}",
                    metadata={"scheme_id": pdf.stem, "scheme_name": name,
                              "category": "PDF document", "level": "",
                              "applicable_states": "", "application_link": "",
                              "source": pdf.name, "section": f"PDF page {pno}"},
                ))
    return docs


def build_index() -> FAISS:
    docs = _scheme_documents() + _pdf_documents()
    schemes = {d.metadata["scheme_name"] for d in docs}
    print(f"Ingesting {len(docs)} chunks across {len(schemes)} schemes/documents...")
    vs = FAISS.from_documents(docs, get_embeddings())
    Path(config.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(str(config.INDEX_DIR))
    print(f"FAISS index saved -> {config.INDEX_DIR}  ({len(schemes)} schemes)")
    return vs


if __name__ == "__main__":
    build_index()
