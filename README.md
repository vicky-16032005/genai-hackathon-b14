# 🏙️ Smart City Government Scheme & Services RAG Portal

**Problem statement:** PS-SC4 · **Team:** B14 · **Domain:** Smart Cities & Infrastructure
**Course:** 25ECAC307 Generative AI · KLE Technological University · GenAI Hackathon 2025

A Retrieval-Augmented Generation portal that lets any citizen ask about government
welfare schemes and get **instant, citation-accurate** answers — in **English or
Kannada** — plus an **eligibility checker** that matches a citizen profile to the
schemes they qualify for. Runs **100% locally** on Ollama + FAISS, so it needs
**no paid APIs** and works offline (hackathon rule compliant).

---

## What it does (maps to the 4 PS-SC4 deliverables)

| # | PS-SC4 deliverable | Where it lives |
|---|---|---|
| 1 | Ingest ≥15 urban schemes **with scheme-name metadata** | `data/schemes.json` (**20 schemes**) → `src/ingest.py` (FAISS + per-chunk scheme metadata) |
| 2 | **Multilingual Q&A (English + Kannada)** returning scheme name, eligibility, documents, application link | `src/rag.py` + `src/translate.py`, surfaced in `app.py` Q&A mode |
| 3 | **Eligibility checker** (age, income, city → matching schemes w/ citations) | `src/eligibility.py` → `app.py` Eligibility mode |
| 4 | **Evaluation on 20 persona queries** (Context Recall + Answer Relevance, RAGAS) | `data/personas.json` + `src/metrics.py` + `evaluate.py` → `data/eval_results.json` |

## Architecture

```
Streamlit UI ── Q&A mode ───► RAG pipeline (src/rag.py)
            │                    ├─ retrieve: FAISS + metadata filter (src/retriever.py)
            │                    │     embeddings: nomic-embed-text (Ollama)
            │                    ├─ generate: qwen3.5:9b (Ollama, think=False)
            │                    └─ Kannada: deep-translator + LLM fallback (src/translate.py)
            └─ Eligibility mode ► rule engine over structured scheme metadata (src/eligibility.py)

Knowledge base: data/schemes.json (20 schemes) ─► src/ingest.py ─► index/ (FAISS)
Evaluation:     data/personas.json (20) ─► evaluate.py ─► data/eval_results.json
```

See **ARCHITECTURE.md** for the architecture-slide version.

## Tech stack (all local / free)

- **LLM:** `qwen3.5:9b` via **Ollama** (generation, reasoning disabled for speed)
- **Embeddings:** `nomic-embed-text` via Ollama
- **Vector DB:** **FAISS** with per-chunk scheme metadata + metadata filtering
- **Orchestration:** LangChain (FAISS store, embeddings, text splitting)
- **Multilingual:** `deep-translator` (Google) with a local-LLM fallback
- **UI:** Streamlit · **Eval:** RAGAS-style Context Recall + Answer Relevance

## Datasets used (from the handbook's PS-SC4 list)

Corpus curated from **MyScheme Government Portal** (myscheme.gov.in) and official
scheme portals (pmaymis.gov.in, mohua.gov.in/AMRUT, pmsvanidhi.mohua.gov.in,
pmjay.gov.in, etc.). Every scheme entry links to its official page as the citation
source. Drop real scheme PDFs into `data/pdfs/` to extend the corpus — `src/ingest.py`
ingests them automatically.

---

## Setup & run

### Prerequisites
- [Ollama](https://ollama.com) running locally with the models pulled:
  ```bash
  ollama pull qwen3.5:9b           # or set GEN_MODEL to any local chat model
  ollama pull nomic-embed-text
  ```
- Python 3.12

### Install
```bash
cd smartcity_rag_b14
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Build the index (one-time, ~2 s)
```bash
python -m src.ingest
```

### Run the app
```bash
streamlit run app.py          # or: ./run.sh
```
Open http://localhost:8501

### Run the evaluation
```bash
python evaluate.py            # writes data/eval_results.json
```

## Evaluation results (20 citizen-persona queries)

| Metric | Score |
|---|---|
| Context Recall (RAGAS-style) | **0.872** |
| Answer Relevance (RAGAS-style) | **0.687** |
| Retrieval hit-rate (expected scheme retrieved) | **1.000 (20/20)** |

Model `qwen3.5:9b` + `nomic-embed-text`, fully local. Reproduce with `python evaluate.py`
(writes `data/eval_results.json`). Transparency note: an earlier run reported Context
Recall 0.628 because the 9B judge emitted malformed JSON that the parser rejected,
producing false zeros; switching the judge to robust line-based output (`src/metrics.py`)
fixed the *measurement*, not the threshold — the per-row verdicts were unchanged, we just
parse them correctly now.

## Configuration

All knobs are env vars (see `.env.example`): `GEN_MODEL`, `EMBED_MODEL`,
`OLLAMA_BASE_URL`, `TOP_K`, `CHUNK_SIZE`, `TEMPERATURE`. Swap `GEN_MODEL=qwen3.6:35b-a3b`
for higher answer quality (slower).

## Notes on honesty & limitations

- Scheme eligibility figures are **indicative** of publicly published criteria and
  link to each official portal as the authoritative source — verify before applying.
- We implement the **RAGAS algorithms** (Context Recall, Answer Relevance) directly
  with local models because `ragas==0.4.3` has an import-time incompatibility with
  current `langchain-community`. The metric definitions are faithful to RAGAS.
- Numbers in `data/eval_results.json` are reported as-is; no thresholds were tuned.
