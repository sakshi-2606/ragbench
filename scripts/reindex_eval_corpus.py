"""
Rebuild an eval collection from the corpus, driven by a config file.

Reads data/eval_corpus_v2.jsonl (chunks now carry a stable `id`) and (re)builds
the collection named in the given config, using that config's embedder. The
chunk id is written into the Qdrant payload as `chunk_id`, so every retrieved
chunk reports which corpus chunk it is — needed for ranking metrics.

    python scripts/reindex_eval_corpus.py configs/query.yaml       # openai small -> demo_k8s_helm
    python scripts/reindex_eval_corpus.py configs/query_bge.yaml   # local bge    -> demo_k8s_helm_bge

Qdrant up; OPENAI_API_KEY set for the openai config.
"""

import json
import os
import sys

from ragbench.utils.config import RunConfig
from ragbench.utils.resources import get_embedder
from ragbench.store.qdrant_store import QdrantStore

CORPUS = "data/eval_corpus_v2.jsonl"


def load_corpus(path):
    texts, payloads = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                texts.append(row["text"])
                payloads.append({
                    "text": row["text"],
                    "source": row.get("source", "unknown"),
                    "chunk_id": row.get("id"),
                })
    return texts, payloads


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("RAGBENCH_CONFIG", "configs/query.yaml")
    cfg = RunConfig.from_yaml(cfg_path)
    r = cfg.retrieval

    texts, payloads = load_corpus(CORPUS)
    embedder = get_embedder(r.embed_kind, r.embed_model)
    store = QdrantStore(collection_name=r.collection, vector_size=embedder.dim)
    store.create_collection(recreate=True)

    if hasattr(embedder, "embed_texts"):
        vectors = embedder.embed_texts(texts)
    else:
        vectors = embedder.embed_documents(texts)

    store.upsert(vectors, payloads)
    print(f"Indexed {len(texts)} chunks into '{r.collection}' "
          f"(model={r.embed_model}, kind={r.embed_kind}, dim={embedder.dim}) from {CORPUS}")
    print(f"config: {cfg_path}  (config_version={cfg.version})")


if __name__ == "__main__":
    main()