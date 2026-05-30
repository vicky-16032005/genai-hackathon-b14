#!/usr/bin/env bash
# One-command launcher for the Smart City Scheme RAG Portal (PS-SC4, Team B14).
set -e
cd "$(dirname "$0")"

# activate venv if present
[ -d .venv ] && source .venv/bin/activate

# build the FAISS index if it doesn't exist yet
if [ ! -f index/index.faiss ]; then
  echo "Building FAISS index..."
  python -m src.ingest
fi

echo "Starting app at http://localhost:8501  (Ctrl+C to stop)"
streamlit run app.py
