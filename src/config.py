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

# ---- models (100% local via Ollama; no paid APIs — hackathon rule compliant) ----
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEN_MODEL = os.getenv("GEN_MODEL", "qwen3.5:9b")          # generation LLM
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # embeddings

# ---- retrieval ----
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ---- generation ----
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "700"))

LANGS = {"English": "en", "Kannada (ಕನ್ನಡ)": "kn"}
