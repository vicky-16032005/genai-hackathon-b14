"""Central configuration for the Smart City Scheme RAG Portal (PS-SC4, Team B14).

All model names and paths are env-overridable so the same code runs on a laptop
with Ollama or on any judge's machine without edits.
"""
import os
from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = ROOT / "index"
SCHEMES_JSON = DATA_DIR / "schemes.json"
PERSONAS_JSON = DATA_DIR / "personas.json"

# ---- models (small + deployable; no paid APIs — hackathon rule compliant) ----
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Generation — pluggable. 'ollama' = fast local small model; 'transformers' =
# self-contained HF model that deploys to HF Spaces / Streamlit Cloud (no Ollama).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
GEN_MODEL = os.getenv("GEN_MODEL", "qwen2.5:1.5b")              # small Ollama model (replaces qwen3.5:9b)
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # transformers backend
# Evaluation judge — defaults to the generator, but can be overridden with a
# stronger local model (e.g. qwen3.5:9b) for credible, generator-independent scoring.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GEN_MODEL)

# Embeddings — pluggable. 'st' = our fine-tuned sentence-transformers retriever
# (deployable, ~90MB, CPU); 'ollama' = nomic-embed-text fallback.
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "st")
ST_EMBED_BASE = os.getenv("ST_EMBED_BASE", "sentence-transformers/all-MiniLM-L6-v2")
ST_EMBED_MODEL = os.getenv("ST_EMBED_MODEL", str(ROOT / "models" / "scheme-retriever"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")     # ollama embedding fallback

# ---- retrieval ----
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ---- generation ----
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "700"))

LANGS = {"English": "en", "Kannada (ಕನ್ನಡ)": "kn"}
