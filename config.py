import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # MANUAL_RAG_OFFLINE=1 -> no model downloads (hash embeddings, no reranker). Smoke tests only.
    if os.environ.get("MANUAL_RAG_OFFLINE") == "1":
        cfg["embeddings"]["backend"] = "hash"
        cfg["reranker"]["backend"] = "none"
    return cfg
