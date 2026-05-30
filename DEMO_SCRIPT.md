# 5-Minute Demo Script — PS-SC4 (Team B14)

> Goal: show all 4 deliverables working in 5 minutes. Have `streamlit run app.py`
> already open at http://localhost:8501 and Ollama running before you start.

### 0:00 — Hook (20s)
> "Citizens miss out on welfare schemes because they don't know what exists or
> whether they qualify. Our Smart City Scheme RAG Portal answers any scheme
> question with **citations**, in **English and Kannada**, and tells a citizen
> exactly **which schemes they're eligible for** — running **100% locally**, no
> paid APIs."

### 0:20 — Deliverable 1: ingestion + corpus (30s)
- Point to the sidebar: `qwen3.5:9b`, `nomic-embed-text`, FAISS, "No paid APIs".
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
- Read out: **avg Context Recall**, **avg Answer Relevance**, **retrieval hit-rate** over
  **20 citizen-persona queries** — the RAGAS metrics PS-SC4 asks for, computed with local models.

### 4:45 — Close (15s)
> "Citation-accurate, bilingual, eligibility-aware, evaluated — and entirely local
> and free. Ready to extend to any city's scheme PDFs."

---
## Backup answers if Wi-Fi drops
Everything except `deep-translator` is offline. If Google translate is unreachable,
the Kannada path automatically falls back to the local Qwen model — just demo it and
mention the fallback. The English Q&A, eligibility checker, and eval are fully offline.
