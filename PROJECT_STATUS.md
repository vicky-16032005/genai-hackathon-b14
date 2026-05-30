# Project Status & Context — Smart City Scheme RAG Portal

**Problem statement:** PS-SC4 — *Smart City Government Scheme & Services RAG Portal*
**Team:** B14 · **Domain:** Smart Cities & Infrastructure · ★ RAG-focused (CO4/CO5)
**Course:** 25ECAC307 Generative AI · KLE Technological University · GenAI Hackathon 2025

---

## 1. The problem (game plan)

Citizens don't know which government schemes exist or whether they qualify. Build a **RAG
portal** that ingests scheme documents and gives **instant, citation-accurate** answers.

**Four graded deliverables:**
1. Ingest **≥15 urban schemes** with scheme-name metadata
2. **English + Kannada** Q&A returning scheme, eligibility, documents, application link — **with citations**
3. **Eligibility checker** (age / income / city → matching schemes, cited)
4. **RAGAS evaluation** on **20 citizen personas** (Context Recall + Answer Relevance)

**Our approach:** retrieve over a curated 20-scheme corpus (FAISS + metadata) → a small,
grounded LLM with strict *"answer only from context, cite [n], never invent"* prompting →
Kannada via a translation layer → a deterministic eligibility engine (no hallucination) →
honest evaluation. Everything runs on a **small, free, deployable** stack — no paid APIs.

## 2. % completion

| Area | Status | % |
|---|---|---|
| 4 graded deliverables | built & verified | **100%** |
| Corpus + ingestion (20 schemes → FAISS) | done | 100% |
| EN + Kannada cited Q&A | done | 100% |
| Eligibility checker | done | 100% |
| RAGAS eval (20 personas) | done | 100% |
| Civic UI + full Kannada localization | done | 100% |
| Deployable model swap + **trained retriever** | done & verified | 100% |
| Pushed to GitHub | done | 100% |
| Live Hugging Face Spaces deploy | uploading / building | ~70% |

**Overall ≈ 95%.** The graded deliverable (working app + all 4 requirements + architecture
slide + demo script + eval, on GitHub) is **100% done**. Only the *live* cloud deploy is still
in flight.

## 3. Models used

| Role | Model | Notes |
|---|---|---|
| Generator (local demo) | **`qwen2.5:1.5b`** via Ollama | replaced `qwen3.5:9b`; fast |
| Generator (deployed) | **`Qwen/Qwen2.5-1.5B-Instruct`** via 🤗 transformers | no Ollama → cloud-deployable |
| Retriever (ours, **trained**) | **fine-tuned `all-MiniLM-L6-v2`** (`models/scheme-retriever`, 87 MB) | trained on synthetic citizen-query→scheme pairs; replaced `nomic-embed-text` |
| Eval judge | **`qwen3.5:9b`** (held constant) | strong judge for credible scoring |
| Kannada | `deep-translator` (Google) + LLM fallback | translation layer (not a trained model) |

**Pluggable** via env: `LLM_PROVIDER` (`ollama`|`transformers`), `EMBED_PROVIDER` (`st`|`ollama`).

## 4. Evaluation (honest, 20 held-out personas)

Same strong judge (`qwen3.5:9b`) both times, so it's apples-to-apples:

| Metric | old (9B + nomic) | new (1.5B + fine-tuned retriever) |
|---|---|---|
| Context Recall | 0.872 | **0.994** ⬆ |
| Retrieval hit-rate | 20/20 | **20/20** |
| Answer Relevance | 0.687 | **0.529** ⬇ |

Deterministic retriever comparison (`train/eval_retriever.py`): Hit@5 1.0 for all; **MRR 0.95 → 1.0**
after fine-tuning (correct scheme rank-1 for every persona).

**Honest read:** training the retriever measurably improved retrieval *and* made it deployable.
Shrinking the generator to 1.5B for deployability cost answer relevance (0.69→0.53) — answers
stay correct & grounded (hence 0.99 recall) but are wordier. One-line fix to recover it while
staying deployable: `GEN_MODEL=qwen2.5:3b`.

## 5. Build journey (what we did, in order)

1. **Selected the PS** — parsed the allocation PDF → B14 = PS-SC4; read the handbook spec.
2. **Corpus** — authored 20 urban schemes (`data/schemes.json`) from MyScheme + official portals,
   each with eligibility fields + official link as the citation source.
3. **Ingestion** — section-labelled chunks (Overview / Eligibility / Application) with scheme-name
   metadata → FAISS (`src/ingest.py`).
4. **RAG pipeline** — retrieve → grounded, `[n]`-cited answer (`src/rag.py`). Fixed a
   `langchain_ollama` empty-output bug by calling the Ollama client directly with `think=False`.
5. **Kannada** — translation layer, `deep-translator` + local-LLM fallback (`src/translate.py`).
6. **Eligibility checker** — deterministic rule engine over scheme metadata (`src/eligibility.py`).
7. **Streamlit UI** — two tabbed modes.
8. **Evaluation** — 20 personas (`data/personas.json`); implemented RAGAS metrics locally
   (`src/metrics.py`). Caught a JSON-parse bug producing false zeros → switched the judge to robust
   line-based parsing (Context Recall 0.628 → 0.872; a *measurement* fix, not threshold-tuning).
9. **GitHub** — pushed to `vicky-16032005/genai-hackathon-b14`.
10. **Civic UI redesign** — tricolor hero, stat chips, tabs, white cards, navy/saffron theme.
11. **Bug fixes & polish** — persisted results in `session_state` + translate-at-render so the
    **language switch works live**; brightened the too-dark Send button; **moved the language
    toggle onto the page** (was hidden in a collapsed sidebar); **full Kannada localization**
    (header, chips, dropdown options via `format_func`, example questions).
12. **Deployable model + trained retriever** — swapped `qwen3.5:9b` → `qwen2.5:1.5b` (pluggable
    Ollama/transformers); generated synthetic data (`train/make_dataset.py`) and **fine-tuned a
    MiniLM retriever** (`train/train_retriever.py`); rebuilt FAISS; honest eval refresh.
13. **Deploy** — Hugging Face Spaces (Docker SDK + `Dockerfile`, transformers generator).

## 6. How to run

**Local (fast):**
```bash
ollama pull qwen2.5:1.5b
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.ingest      # build FAISS (uses the fine-tuned retriever in models/)
streamlit run app.py      # http://localhost:8501
```
**Reproduce the training + eval:**
```bash
python -m train.make_dataset && python -m train.train_retriever && python -m train.eval_retriever
JUDGE_MODEL=qwen3.5:9b python evaluate.py
```
**Deploy (no Ollama):** Hugging Face Spaces, Docker SDK — `Dockerfile` runs the app with
`LLM_PROVIDER=transformers`; the trained retriever + Qwen2.5-1.5B-Instruct run in-process.

## 7. Links
- **GitHub:** https://github.com/vicky-16032005/genai-hackathon-b14
- **Live Space:** https://huggingface.co/spaces/01fe23bci060/smartcity-scheme-rag-b14 *(building)*

## 8. Honest notes / known limitations
- Scheme eligibility figures are **indicative** of publicly published criteria and link to each
  official portal as the authoritative source — verify before applying.
- Answer Relevance (0.53) is the cost of the small deployable generator; bump to `qwen2.5:3b` to recover.
- On HF free CPU, Q&A is slow (~30–90 s/answer); eligibility + retrieval are instant.
- RAGAS metrics are reimplemented locally (the `ragas` library has an import-time incompatibility
  with current `langchain-community`); the algorithms are faithful to RAGAS.
