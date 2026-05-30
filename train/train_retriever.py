"""Fine-tune a small, deployable retriever (sentence-transformers / MiniLM) on the
synthetic citizen-query -> scheme-passage pairs.

Output: models/scheme-retriever/  (~90 MB, CPU-friendly, no Ollama needed —
this is the deployable trained artifact that replaces Ollama's nomic-embed-text.)
"""
import json
from pathlib import Path

from src import config

BASE = config.ST_EMBED_BASE
OUT = Path(config.ROOT) / "models" / "scheme-retriever"
PAIRS = Path(config.ROOT) / "train" / "pairs.jsonl"


def main():
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    pairs = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(pairs)} training pairs from {PAIRS.name}")

    model = SentenceTransformer(BASE)
    examples = [InputExample(texts=[p["query"], p["positive"]]) for p in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    loss = losses.MultipleNegativesRankingLoss(model)

    print(f"Fine-tuning {BASE} for 3 epochs on {len(examples)} pairs…")
    model.fit(train_objectives=[(loader, loss)], epochs=3, warmup_steps=10,
              show_progress_bar=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))
    print(f"\nSaved fine-tuned retriever -> {OUT}")


if __name__ == "__main__":
    main()
