# 5-Minute Demo Script — PS-SC4 (Team B14)

> Goal: show all 4 deliverables working in 5 minutes. Have `streamlit run app.py`
> already open at http://localhost:8501 and Ollama running before you start.

### 0:00 — Hook (20s)
> "Citizens miss out on welfare schemes because they don't know what exists or
> whether they qualify. Our Smart City Scheme RAG Portal answers any scheme
> question with **citations**, in **English and Kannada**, and tells a citizen
> exactly **which schemes they're eligible for** — on a **small, deployable, no-paid-API**
> stack."

### 0:20 — Deliverable 1: ingestion + corpus (30s)
- Point to the sidebar: small `qwen2.5:1.5b` generator + our **fine-tuned MiniLM retriever**,
  FAISS, "No paid APIs · deployable".
- Mention: **20 urban schemes** ingested with scheme-name metadata (PMAY-U, AMRUT,
  Swachh Bharat, PM SVANidhi, Ayushman Bharat, + Karnataka's Gruha Lakshmi / Anna Bhagya).
- "`python -m src.ingest` builds the FAISS index in ~2 seconds; drop PDFs into
  `data/pdfs/` and they're ingested too."

### 0:50 — Deliverable 2: citation-accurate Q&A in English (75s)
- Click the example: **"What health insurance does the government provide and who is eligible?"**
- Show: the answer names **Ayushman Bharat PM-JAY**, the **Rs 5 lakh** cover, eligibility,
  and the **official link** — each backed by a **[n] citation**.
- Expand **"Retrieved context"** → "every sentence is grounded in a retrieved scheme
  passage; the prompt forbids inventing schemes or links."

### 2:05 — Deliverable 2 (cont.): Kannada (45s)
- Switch **Answer language → Kannada (ಕನ್ನಡ)**.
- Ask: **"How does a woman get a free LPG connection?"**
- Show the **Kannada answer** (Ujjwala Yojana) + "Show English answer" expander + citations.

### 2:50 — Deliverable 3: eligibility checker (75s)
- Sidebar → **✅ Eligibility checker**; set **State = Karnataka**.
- Enter: **Age 35**, **Income 2,40,000**, **Gender Female**, needs = **Street vendor**.
- Click **Find my schemes** → show ranked matches: **PM SVANidhi** and **e-Shram** on top,
  plus **Gruha Lakshmi** (Karnataka women) — each card shows **why eligible**, documents,
  an **Apply link**, and a **citation** to the official source.
- "This is a deterministic rule engine over structured metadata — exact, no hallucination."

### 4:05 — Deliverable 4: evaluation (40s)
- Open `data/eval_results.json` (or run `python evaluate.py`).
- Read out: **Context Recall 0.99 · retrieval hit-rate 20/20 · Answer Relevance 0.53**
  over **20 held-out citizen personas** (RAGAS-style, judged by a held-constant strong model).
- "And we **trained** a model: our fine-tuned MiniLM retriever lifted Context Recall 0.87 → 0.99
  and ranking MRR 0.95 → 1.0 (`train/eval_retriever.py`) — and it's a deployable ~90 MB CPU model
  that drops the Ollama dependency for embeddings."

### 4:45 — Close (15s)
> "Citation-accurate, bilingual, eligibility-aware, evaluated — and with a trained retriever +
> a small pluggable LLM, it's small enough to deploy with no paid APIs. Ready for any city's PDFs."

---
## Backup answers if Wi-Fi drops
Everything except `deep-translator` is offline. If Google translate is unreachable,
the Kannada path automatically falls back to the local Qwen model — just demo it and
mention the fallback. The English Q&A, eligibility checker, and eval are fully offline.
