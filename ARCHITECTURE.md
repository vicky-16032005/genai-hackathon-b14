# Architecture Slide — PS-SC4 Smart City Scheme RAG Portal (Team B14)

## One-liner
A fully-local RAG portal that gives citizens **citation-accurate** answers about
government schemes in **English & Kannada**, plus a **rule-based eligibility checker**.

## System diagram

```
┌──────────────────────────── Streamlit UI (app.py) ────────────────────────────┐
│   Mode: [💬 RAG Q&A]   [✅ Eligibility checker]        Language: English / ಕನ್ನಡ │
└───────────────┬─────────────────────────────────────────────┬──────────────────┘
                │                                               │
     ┌──────────▼───────────┐                       ┌───────────▼─────────────┐
     │  RAG pipeline         │                       │  Eligibility engine     │
     │  (src/rag.py)         │                       │  (src/eligibility.py)   │
     │  1 translate q→EN     │                       │  rule filter on age /   │
     │  2 retrieve top-k     │                       │  income / state /gender │
     │  3 LLM answer + [n]   │                       │  → ranked schemes +     │
     │  4 translate→Kannada  │                       │    citations            │
     └──────┬──────────┬─────┘                       └───────────┬─────────────┘
            │          │                                         │
   ┌────────▼───┐  ┌───▼─────────────┐               ┌───────────▼─────────────┐
   │ FAISS index│  │ Ollama qwen3.5:9b│               │ structured scheme       │
   │ + metadata │  │  (generation)    │               │ metadata (eligibility)  │
   │(retriever) │  └──────────────────┘               └─────────────────────────┘
   └────┬───────┘
        │ embeddings: nomic-embed-text (Ollama)
   ┌────▼────────────────────────────────────────┐
   │ Ingestion (src/ingest.py)                    │
   │ data/schemes.json (20 schemes) + data/pdfs/  │
   │ → section-labelled chunks w/ scheme-name meta│
   └──────────────────────────────────────────────┘

Evaluation (evaluate.py + src/metrics.py): 20 personas → Context Recall, Answer Relevance
```

## Key design decisions

1. **Section-labelled chunks** (Overview / Eligibility / Application) per scheme →
   precise retrieval and clean `Scheme — Section` citations.
2. **Scheme-name metadata on every chunk** → satisfies the PS-SC4 metadata
   requirement and powers state-aware + metadata-filtered retrieval.
3. **Deterministic eligibility engine** (no LLM) → exact, verifiable, hallucination-free
   matching; the LLM is used only for the open-ended Q&A.
4. **100% local (Ollama + FAISS)** → no paid APIs, offline-capable, rule-compliant.
5. **Grounded prompting** → "answer ONLY from context, cite [n], never invent" →
   citation-accurate answers.
6. **Kannada via translation layer** → English retrieval/generation, then translate,
   so retrieval quality stays high and the answer is bilingual.

## Data flow for one Q&A query
`question → (kn→en) → FAISS top-k (state-aware) → numbered context → qwen3.5:9b →
cited answer → (en→kn) → UI with clickable official links`

## Evaluation method
- **Context Recall:** LLM decomposes each reference answer into atomic statements and
  checks attribution to retrieved context. (RAGAS algorithm, local LLM judge.)
- **Answer Relevance:** reverse-generate questions from the answer, embed with
  nomic-embed, mean cosine similarity to the original question. (RAGAS algorithm.)
- **Retrieval hit-rate:** deterministic — was the expected scheme actually retrieved.
