"""Smart City Government Scheme & Services RAG Portal — Streamlit UI.

PS-SC4 · Team B14 · KLE Technological University GenAI Hackathon.
Civic / government theme. Two modes (tabs): citation-accurate RAG Q&A in
English/Kannada, and an eligibility checker. 100% local stack (Ollama + FAISS).

Results are persisted in session_state and translated at render time, so the
language toggle re-renders instantly in the chosen language (no LLM re-run).
"""
from html import escape as _esc
from pathlib import Path

import streamlit as st

from src import config, rag, eligibility, translate


st.set_page_config(page_title="Smart City Scheme Portal", page_icon="🏛️",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Building the scheme index…")
def _ensure_index():
    """Build the FAISS index on first boot if absent (e.g. fresh cloud deploy)."""
    if not (Path(config.INDEX_DIR) / "index.faiss").exists():
        from src import ingest
        ingest.build_index()
    return True


_ensure_index()

# --------------------------------------------------------------------------- #
#  Civic / government theme (forced light, India-flag accents)
# --------------------------------------------------------------------------- #
CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] { background:#FFFFFF; }
[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility:hidden; height:0; }
.block-container { padding-top:1.1rem; padding-bottom:2rem; max-width:1080px; }
section[data-testid="stSidebar"] { background:#F4F7FB; border-right:1px solid #E3E8EF; }
html, body, .stMarkdown, p, span, label, li { color:#1A1F36; }

.tricolor { height:5px; border-radius:4px; margin-bottom:12px;
  background:linear-gradient(90deg,#FF9933 0 33%, #ffffff 33% 66%, #138808 66% 100%);
  border:1px solid #EEE; }

.hero { background:linear-gradient(120deg,#F2860B 0%, #E06D14 45%, #138808 100%);
  border-radius:16px; padding:24px 30px; color:#fff; box-shadow:0 8px 26px rgba(11,61,145,.16); }
.hero .gov { font-size:.74rem; letter-spacing:.16em; text-transform:uppercase; opacity:.92; }
.hero h1 { font-family:Georgia,'Times New Roman',serif; font-size:2.05rem; font-weight:700;
  margin:.18rem 0 .25rem; color:#fff; line-height:1.15; }
.hero p { margin:0; font-size:1.02rem; opacity:.97; color:#fff; }

.chips { display:flex; gap:9px; flex-wrap:wrap; margin:14px 0 2px; }
.chip { background:#F4F7FB; border:1px solid #E3E8EF; color:#0B3D91; padding:6px 13px;
  border-radius:999px; font-size:.85rem; font-weight:600; }
.chip b { color:#138808; }

[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:14px;
  border:1px solid #E3E8EF !important; box-shadow:0 2px 12px rgba(20,30,60,.05);
  transition:box-shadow .15s ease; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow:0 4px 18px rgba(20,30,60,.09); }

.sname { font-size:1.1rem; font-weight:700; color:#13203B; margin:0 0 1px; }
.scat { color:#5A6B86; font-size:.85rem; margin-bottom:7px; }
.badge { display:inline-block; padding:2px 9px; border-radius:6px; font-size:.72rem;
  font-weight:700; letter-spacing:.02em; vertical-align:middle; }
.badge.central { background:#E7F0FF; color:#0B3D91; }
.badge.state { background:#E9F7EC; color:#138808; }
.why { display:inline-block; background:#E9F7EC; color:#0E6B1F; border:1px solid #BFE6C7;
  padding:3px 10px; border-radius:999px; font-size:.78rem; margin:3px 4px 0 0; }
.cite { display:inline-block; background:#F4F7FB; border:1px solid #E3E8EF; color:#33415A;
  padding:3px 9px; border-radius:6px; font-size:.78rem; margin:4px 4px 0 0; }
.cite a { color:#0B3D91; text-decoration:none; font-weight:600; }
.kn { color:#5A6B86; font-size:.96rem; margin-bottom:2px; }
.ans-h { font-weight:700; color:#0B3D91; font-size:.82rem; letter-spacing:.08em;
  text-transform:uppercase; margin-bottom:4px; }

/* primary action button — brighter, inviting (was too-dark navy) */
.stButton>button[kind="primary"] {
  background:linear-gradient(180deg,#3B82F6 0%, #2563EB 100%);
  border:0; color:#fff; font-weight:700; border-radius:10px; padding:.55rem 1.1rem;
  box-shadow:0 4px 14px rgba(37,99,235,.34); transition:all .15s ease; }
.stButton>button[kind="primary"]:hover {
  background:linear-gradient(180deg,#4F90FF 0%, #2F73F0 100%);
  box-shadow:0 7px 20px rgba(37,99,235,.45); transform:translateY(-1px); color:#fff; }
.stButton>button[kind="primary"]:active { transform:translateY(0); }
/* secondary (example) buttons — light, clean */
.stButton>button[kind="secondary"] { border-radius:10px; border:1px solid #D6DEEC;
  color:#1A2A4A; background:#FBFCFE; font-weight:500; }
.stButton>button[kind="secondary"]:hover { border-color:#2563EB; color:#2563EB; background:#fff; }
[data-testid="stLinkButton"] a { border-radius:9px; border:1px solid #2563EB !important;
  color:#2563EB !important; font-weight:600; }

[data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid #E3E8EF; }
button[data-baseweb="tab"] { font-size:1rem; font-weight:600; padding:8px 4px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  Sidebar — controls (sets `lang`)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Advanced")
    top_k = st.slider("Passages to retrieve (k)", 3, 10, config.TOP_K)
    st.divider()
    st.markdown(
        "**Stack — small & deployable**\n\n"
        f"- LLM `{config.GEN_MODEL}` · {config.LLM_PROVIDER}\n"
        "- Retriever: fine-tuned MiniLM (ours)\n"
        "- FAISS + metadata · LangChain\n\n"
        "✅ No paid APIs · deployable"
    )
    st.caption("PS-SC4 · Team B14 · KLE Tech")


# Determine language up-front by reading the toggle's stored value, so the
# header, chips and every label below localize in one consistent pass. (The
# toggle widget itself is rendered a little lower, under the chips.)
lang = "kn" if st.session_state.get("langsel") == "ಕನ್ನಡ" else "en"


def L(text_en: str) -> str:
    """Localize a UI string to the selected language (cached). Also used as a
    selectbox/multiselect format_func: the displayed label is translated while
    the returned option *value* stays English for the matching logic."""
    return translate.to_kannada(text_en) if lang == "kn" else text_en


# --------------------------------------------------------------------------- #
#  Header (fully localized)
# --------------------------------------------------------------------------- #
st.markdown('<div class="tricolor"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero">'
    f'<div class="gov">{_esc(L("Government of India · Smart Cities Mission"))}</div>'
    f'<h1>🏛️ {_esc(L("Smart City Scheme & Services Portal"))}</h1>'
    f'<p>{_esc(L("Find every government scheme you qualify for — with citation-accurate answers in English and Kannada."))}</p>'
    '</div>',
    unsafe_allow_html=True,
)
_chips = [
    ("📚", f'<b>20</b> {_esc(L("schemes"))}'),
    ("🗣️", f'{_esc(L("English"))} + ಕನ್ನಡ'),
    ("🔖", _esc(L("Cited answers"))),
    ("🎯", f'{_esc(L("Context Recall"))} <b>0.99</b>'),
    ("🔒", _esc(L("No paid APIs"))),
]
st.markdown('<div class="chips">'
            + "".join(f'<span class="chip">{e}&nbsp; {t}</span>' for e, t in _chips)
            + '</div>', unsafe_allow_html=True)
st.write("")

# Language + region controls — prominent on the page (never hidden in a
# collapsible sidebar). The selected value was already read above.
_lc, _sc, _sp = st.columns([1.5, 1.7, 1.8])
with _lc:
    st.radio("🌐 Language  /  ಭಾಷೆ", ["English", "ಕನ್ನಡ"], horizontal=True, key="langsel")
with _sc:
    state = st.selectbox(
        L("📍 Your state / city"),
        ["All India", "Karnataka", "Maharashtra", "Tamil Nadu", "Delhi", "Uttar Pradesh"],
        format_func=L,
        help="Karnataka unlocks state schemes (Gruha Lakshmi, Anna Bhagya) + all central schemes.",
    )


st.write("")
st.session_state.setdefault("q", "")
st.session_state.setdefault("qa", None)      # last Q&A result (English, translated at render)
st.session_state.setdefault("elig", None)    # last eligibility result list (English)


def cite_chips(citations):
    if not citations:
        return ""
    chips = "".join(
        f'<span class="cite"><a href="{c["application_link"] or "#"}" target="_blank">'
        f'[{c["n"]}] {c["scheme_name"]}</a> · {c["section"]}</span>'
        for c in citations
    )
    return f'<div class="ans-h">📚 {L("Sources")}</div><div>{chips}</div>'


tab_qa, tab_elig = st.tabs([f"💬  {L('Ask a question')}", f"✅  {L('Eligibility checker')}"])

# --------------------------------------------------------------------------- #
#  Q&A tab
# --------------------------------------------------------------------------- #
with tab_qa:
    st.markdown(f"#### {L('Ask about any government scheme')}")
    st.caption(L("Answers come only from ingested scheme documents — every claim is cited."))

    examples = [
        "What housing scheme can I apply for and what documents do I need?",
        "I am a 65-year-old below poverty line. What pension schemes exist?",
        "How does a woman get a free LPG connection?",
        "What health insurance does the government provide and who is eligible?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        label = L(ex)
        if cols[i % 2].button(label, key=f"ex{i}", use_container_width=True):
            st.session_state["q"] = label

    query = st.text_input(L("Your question (type in English or Kannada):"), key="q",
                          placeholder=L("e.g. I run a small shop — which loan scheme can help me?"))
    go = st.button(f"🔎  {L('Get a cited answer')}", type="primary")

    if go and query.strip():
        with st.spinner(L("Retrieving scheme documents and generating a cited answer…")):
            try:
                # always retrieve/generate in English; translate at render time
                src = translate.to_english(query) if lang == "kn" else query
                st.session_state["qa"] = rag.answer_query(src, lang="en", state=state, k=top_k)
            except Exception as e:
                st.session_state["qa"] = None
                st.error(f"Generation failed: {e}\n\nIs Ollama running?  `ollama serve`")

    qa = st.session_state.get("qa")
    if qa:
        answer = translate.to_kannada(qa["answer_en"]) if lang == "kn" else qa["answer_en"]
        with st.container(border=True):
            st.markdown(f'<div class="ans-h">{L("Answer")}</div>', unsafe_allow_html=True)
            st.markdown(answer)
            if lang == "kn":
                with st.expander(L("Show English answer")):
                    st.markdown(qa["answer_en"])
            if qa["citations"]:
                st.markdown(cite_chips(qa["citations"]), unsafe_allow_html=True)
        with st.expander("🔍 " + L("Retrieved context (what the model read)")):
            for i, ctx in enumerate(qa["contexts"], 1):
                st.text(f"[{i}] {ctx}\n")

# --------------------------------------------------------------------------- #
#  Eligibility tab
# --------------------------------------------------------------------------- #
with tab_elig:
    st.markdown(f"#### {L('Which schemes am I eligible for?')}")
    st.caption(L("Enter your details — we match you against all schemes and cite each one."))

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input(L("Age"), min_value=0, max_value=120, value=30)
        income = c2.number_input(L("Annual household income (₹)"), min_value=0, value=200000, step=10000)
        gender = c3.selectbox(L("Gender"), ["Any", "Female", "Male"], format_func=L)
        interests = st.multiselect(L("Your situation / needs (optional — improves ranking)"),
                                   eligibility.ALL_INTERESTS, format_func=L)
        if st.button(f"✅  {L('Find my schemes')}", type="primary"):
            st.session_state["elig"] = {
                "results": eligibility.check(age=age, annual_income=income, state=state,
                                             gender=gender, interests=interests),
                "interests": interests,
            }

    def render_scheme_cards(rows, translating):
        for r in rows:
            badge_cls = "state" if r["level"] == "State" else "central"
            why_html = "".join(f'<span class="why">✓ {L(w)}</span>' for w in r["why"]) or \
                       f'<span class="why">✓ {L("Generally available")}</span>'
            kn = f'<div class="kn">{translate.to_kannada(r["scheme_name"])}</div>' if translating else ""
            benefits = translate.to_kannada(r["benefits"]) if translating else r["benefits"]
            with st.container(border=True):
                st.markdown(
                    f'<div class="sname">{r["scheme_name"]}</div>{kn}'
                    f'<div class="scat">{r["category"]} · '
                    f'<span class="badge {badge_cls}">{r["level"]}</span></div>'
                    f'<div>{why_html}</div>'
                    f'<div style="margin-top:9px"><b>{L("Benefits")}:</b> {benefits}</div>'
                    f'<div style="margin-top:5px"><b>{L("Documents needed")}:</b> '
                    f'{", ".join(r["documents_needed"])}</div>'
                    f'<div style="margin-top:7px"><span class="cite">{L("citation")}: {r["source"]}</span></div>',
                    unsafe_allow_html=True,
                )
                if r["application_link"]:
                    st.link_button(f"🔗  {L('Apply / official page')}", r["application_link"])

    elig = st.session_state.get("elig")
    if elig is not None:
        results = elig["results"]
        st.success(L(f"You may be eligible for {len(results)} schemes (strongest matches first)."))
        show = results if elig["interests"] else results[:10]
        if lang == "kn":
            with st.spinner(L("Translating to Kannada…")):
                render_scheme_cards(show, True)
        else:
            render_scheme_cards(show, False)

st.divider()
st.caption("Smart City Government Scheme & Services RAG Portal · PS-SC4 · Team B14 · "
           "KLE Tech GenAI Hackathon 2025 · Ollama + FAISS, fully local.")
