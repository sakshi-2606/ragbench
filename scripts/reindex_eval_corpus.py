"""
Rebuild the eval collection from a corpus file.

Reads data/eval_corpus_v1.jsonl and (re)builds the `demo_k8s_helm` collection
with text-embedding-3-small — i.e. the exact collection + embedding model the
service config (configs/query.yaml) queries. This replaces the ad-hoc 3-doc
index_and_search_demo.py for the eval path and makes the corpus reproducible
from a file.

Run from repo root (Qdrant up, OPENAI_API_KEY set):
    python scripts/reindex_eval_corpus.py
"""

import json

from ragbench.store.qdrant_store import QdrantStore
from ragbench.embed.providers import OpenAIEmbeddingProvider

CORPUS = "data/eval_corpus_v1.jsonl"
COLLECTION = "demo_k8s_helm"              # must match configs/query.yaml
EMBED_MODEL = "text-embedding-3-small"    # must match configs/query.yaml


def load_corpus(path):
    texts, payloads = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                texts.append(row["text"])
                payloads.append({"text": row["text"], "source": row.get("source", "unknown")})
    return texts, payloads


def main():
    texts, payloads = load_corpus(CORPUS)
    embedder = OpenAIEmbeddingProvider(model=EMBED_MODEL)
    store = QdrantStore(collection_name=COLLECTION, vector_size=embedder.dim)

    store.create_collection(recreate=True)      # drop + recreate, clean rebuild
    vectors = embedder.embed_texts(texts)       # one batched embedding call
    store.upsert(vectors, payloads)

    print(f"Indexed {len(texts)} chunks into '{COLLECTION}' (dim={embedder.dim}) from {CORPUS}")


if __name__ == "__main__":
    main()