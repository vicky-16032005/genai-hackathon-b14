"""Smart City Government Scheme & Services RAG Portal — Streamlit UI.

PS-SC4 · Team B14 · KLE Technological University GenAI Hackathon.
Two modes: (1) citation-accurate RAG Q&A in English/Kannada,
(2) eligibility checker. 100% local stack (Ollama + FAISS), no paid APIs.
"""
import streamlit as st

from src import config, rag, eligibility, translate

st.set_page_config(page_title="Smart City Scheme RAG Portal", page_icon="🏙️", layout="wide")

# ---------- sidebar ----------
with st.sidebar:
    st.title("🏙️ Scheme RAG Portal")
    st.caption("PS-SC4 · Team B14")
    mode = st.radio("Mode", ["💬 Ask a question (RAG)", "✅ Eligibility checker"])
    lang_label = st.radio("Answer language", list(config.LANGS.keys()))
    lang = config.LANGS[lang_label]
    state = st.selectbox(
        "Your state / city",
        ["All India", "Karnataka", "Maharashtra", "Tamil Nadu", "Delhi", "Uttar Pradesh"],
        help="Karnataka unlocks state schemes (Gruha Lakshmi, Anna Bhagya) plus all central schemes.",
    )
    top_k = st.slider("Passages to retrieve (k)", 3, 10, config.TOP_K)
    st.divider()
    st.markdown(
        f"**Stack (all local):**\n\n"
        f"- LLM: `{config.GEN_MODEL}` via Ollama\n"
        f"- Embeddings: `{config.EMBED_MODEL}`\n"
        f"- Vector DB: FAISS + metadata\n"
        f"- Kannada: deep-translator + LLM fallback\n\n"
        f"✅ No paid APIs · runs offline"
    )

# ---------- helpers ----------
def render_citations(citations):
    if not citations:
        return
    st.markdown("##### 📚 Citations")
    for c in citations:
        link = c["application_link"] or "#"
        st.markdown(
            f"**[{c['n']}] {c['scheme_name']}** — _{c['section']}_  ·  "
            f"[Official link]({link})  ·  source: `{c['source']}`"
        )


def t(text_en):
    """Translate static UI strings when Kannada is selected."""
    return translate.to_kannada(text_en) if lang == "kn" else text_en


# ---------- Q&A mode ----------
if mode.startswith("💬"):
    st.header("💬 Ask about any government scheme")
    st.caption("Answers are generated only from ingested scheme documents, with citations.")

    examples = [
        "What housing scheme can I apply for and what documents do I need?",
        "I am a 65-year-old below poverty line. What pension schemes exist?",
        "How does a woman get a free LPG connection?",
        "What health insurance does the government provide and who is eligible?",
    ]
    st.session_state.setdefault("q", "")
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state["q"] = ex  # set before the keyed widget is created

    query = st.text_input("Your question (type in English or Kannada):", key="q")
    if st.button("🔎 Ask", type="primary") and query.strip():
        with st.spinner("Retrieving scheme documents and generating a cited answer..."):
            try:
                res = rag.answer_query(query, lang=lang, state=state, k=top_k)
                st.markdown("### Answer")
                st.markdown(res["answer"])
                if lang == "kn":
                    with st.expander("Show English answer"):
                        st.markdown(res["answer_en"])
                render_citations(res["citations"])
                with st.expander("🔍 Retrieved context (what the model read)"):
                    for i, ctx in enumerate(res["contexts"], 1):
                        st.text(f"[{i}] {ctx}\n")
            except Exception as e:
                st.error(f"Generation failed: {e}\n\nIs Ollama running? `ollama serve`")

# ---------- Eligibility mode ----------
else:
    st.header("✅ Which schemes am I eligible for?")
    st.caption("Enter your details — we match you against all schemes and cite each one.")

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", min_value=0, max_value=120, value=30)
    income = c2.number_input("Annual household income (Rs)", min_value=0, value=200000, step=10000)
    gender = c3.selectbox("Gender", ["Any", "Female", "Male"])
    interests = st.multiselect(
        "Your situation / needs (optional — improves ranking)",
        eligibility.ALL_INTERESTS,
    )

    if st.button("✅ Find my schemes", type="primary"):
        results = eligibility.check(age=age, annual_income=income, state=state,
                                    gender=gender, interests=interests)
        st.success(f"Found **{len(results)}** schemes you may be eligible for "
                   f"(showing strongest matches first).")
        show = results if interests else results[:10]
        for r in show:
            with st.container(border=True):
                title = r["scheme_name"]
                if lang == "kn":
                    title = f"{title}\n\n{translate.to_kannada(r['scheme_name'])}"
                st.markdown(f"#### {title}")
                st.caption(f"{r['category']} · {r['level']} scheme")
                if r["why"]:
                    st.markdown(" ".join(f"`✓ {w}`" for w in r["why"]))
                benefits = translate.to_kannada(r["benefits"]) if lang == "kn" else r["benefits"]
                st.markdown(f"**Benefits:** {benefits}")
                st.markdown(f"**Documents needed:** {', '.join(r['documents_needed'])}")
                st.markdown(
                    f"[🔗 Apply / official page]({r['application_link']})  ·  "
                    f"citation: `{r['source']}`"
                )

st.divider()
st.caption("Smart City Government Scheme & Services RAG Portal · PS-SC4 · Team B14 · "
           "KLE Tech GenAI Hackathon 2025 · No paid APIs — Ollama + FAISS, fully local.")
