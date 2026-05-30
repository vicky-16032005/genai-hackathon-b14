import os

# FAISS and PyTorch both ship their own OpenMP runtime; on macOS the duplicate
# triggers "OMP: Error #15" and aborts. This is the documented workaround and
# must be set before either library is imported, so it lives in the package init.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
