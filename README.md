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

## Tech stack (small, free, deployable)

- **LLM (pluggable):** small instruct model — `qwen2.5:1.5b` via **Ollama** (fast local) or
  `Qwen/Qwen2.5-1.5B-Instruct` via **transformers** (`LLM_PROVIDER=transformers`) so it
  deploys to HF Spaces / Streamlit Cloud with no Ollama daemon.
- **Retriever (trained, ours):** a **fine-tuned `all-MiniLM-L6-v2`** (`models/scheme-retriever`,
  ~90 MB, CPU) — self-contained and deployable, replacing the Ollama `nomic-embed-text` dependency.
- **Vector DB:** **FAISS** with per-chunk scheme metadata + metadata filtering
- **Orchestration:** LangChain · **Multilingual:** `deep-translator` (Google) + local-LLM fallback
- **UI:** Streamlit · **Eval:** RAGAS-style Context Recall + Answer Relevance + a deterministic
  retriever comparison

## The trained retriever (deployable)

We fine-tuned the embedding retriever — the lever that actually moves RAG quality — rather than the
generator (which RAG already grounds). Synthetic *citizen-query → scheme* pairs were generated from
the corpus (`train/make_dataset.py`), then `all-MiniLM-L6-v2` was fine-tuned with
`MultipleNegativesRankingLoss` (`train/train_retriever.py`). The 20 eval personas were **held out**.

Honest, judge-free comparison on those held-out personas (`train/eval_retriever.py`):

| Retriever | Hit@5 | MRR |
|---|---|---|
| Ollama `nomic-embed-text` (old) | 1.000 | 0.950 |
| `all-MiniLM-L6-v2` base | 1.000 | 0.950 |
| **fine-tuned MiniLM (ours)** | 1.000 | **1.000** |

Hit@5 was already saturated (20 clean schemes ⇒ easy recall), so training couldn't raise it; the
fine-tune improved **ranking (MRR 0.95 → 1.0** — correct scheme rank-1 for all 20) and, more
importantly, made the retriever a **deployable, Ollama-free** artifact. Reproduce:
`python -m train.make_dataset && python -m train.train_retriever && python -m train.eval_retriever`.

## Datasets used (from the handbook's PS-SC4 list)

Corpus curated from **MyScheme Government Portal** (myscheme.gov.in) and official
scheme portals (pmaymis.gov.in, mohua.gov.in/AMRUT, pmsvanidhi.mohua.gov.in,
pmjay.gov.in, etc.). Every scheme entry links to its official page as the citation
source. Drop real scheme PDFs into `data/pdfs/` to extend the corpus — `src/ingest.py`
ingests them automatically.

---

## Setup & run

### Prerequisites
- Python 3.12
- **Local (fast):** [Ollama](https://ollama.com) with the small model:
  ```bash
  ollama pull qwen2.5:1.5b         # generator (override with GEN_MODEL)
  ```
  Embeddings use the fine-tuned MiniLM in `models/scheme-retriever` — no Ollama needed for retrieval.
- **Deploy (no Ollama, e.g. HF Spaces / Streamlit Cloud):** set `LLM_PROVIDER=transformers`
  and the app loads `Qwen/Qwen2.5-1.5B-Instruct` + the local MiniLM retriever entirely in-process.

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

Deployed stack — generator `qwen2.5:1.5b`, our fine-tuned MiniLM retriever — judged by a
held-constant strong model (`qwen3.5:9b`, set via `JUDGE_MODEL`) so the numbers are credible
and comparable to the earlier baseline:

| Metric | old (9B + nomic) | **new (1.5B + fine-tuned retriever)** |
|---|---|---|
| Context Recall (RAGAS-style) | 0.872 | **0.994** ⬆ |
| Retrieval hit-rate | 20/20 | **20/20** |
| Answer Relevance (RAGAS-style) | 0.687 | **0.529** ⬇ |

**Honest read:** the trained retriever raised Context Recall (0.872 → 0.994) and ranking
(MRR 0.95 → 1.0) — the right context is now retrieved for ~every query. Shrinking the generator
9B → 1.5B for deployability cost answer relevance (0.687 → 0.529): the small model's answers stay
*correct and grounded* (hence recall ≈ 1.0) but are wordier / less tightly on-question. To recover
relevance while staying deployable, set `GEN_MODEL=qwen2.5:3b` or `LLM_PROVIDER=transformers` with a
larger HF model. Reproduce: `JUDGE_MODEL=qwen3.5:9b python evaluate.py`.

*Methodology note:* an even earlier run mis-reported Context Recall (0.628) because the judge emitted
malformed JSON the parser rejected (false zeros); switching to robust line-based parsing
(`src/metrics.py`) fixed the *measurement*, not any threshold.

## Configuration

All knobs are env vars (see `.env.example`): `LLM_PROVIDER` (`ollama`|`transformers`),
`GEN_MODEL`, `HF_GEN_MODEL`, `EMBED_PROVIDER` (`st`|`ollama`), `ST_EMBED_MODEL`, `JUDGE_MODEL`,
`TOP_K`, `TEMPERATURE`. For higher answer quality while staying deployable, set
`GEN_MODEL=qwen2.5:3b` (Ollama) or point `HF_GEN_MODEL` at a larger HF model.

## Notes on honesty & limitations

- Scheme eligibility figures are **indicative** of publicly published criteria and
  link to each official portal as the authoritative source — verify before applying.
- We implement the **RAGAS algorithms** (Context Recall, Answer Relevance) directly
  with local models because `ragas==0.4.3` has an import-time incompatibility with
  current `langchain-community`. The metric definitions are faithful to RAGAS.
- Numbers in `data/eval_results.json` are reported as-is; no thresholds were tuned.
