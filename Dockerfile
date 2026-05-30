# Hugging Face Space (Docker SDK) — runs the Streamlit app on port 7860.
FROM python:3.12-slim

# non-root user per HF Spaces convention (writable HOME for caches + index)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    LLM_PROVIDER=transformers \
    EMBED_PROVIDER=st \
    HF_GEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    TOKENIZERS_PARALLELISM=false \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# CORS/XSRF off so the app renders inside the HF Spaces iframe
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
