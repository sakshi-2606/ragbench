"""
Cached, reusable pipeline resources.

The embedder, the Qdrant client, and the cross-encoder reranker are expensive
to build — the reranker especially (it loads a model). Before this, run_rag
rebuilt all three on EVERY call, which is wasteful for a single /query and
genuinely slow across a batch /eval (the reranker model would reload once per
item).

functools.lru_cache memoizes each getter by its arguments: the first call with
a given (kind, model) builds the object, every later call with the same
arguments returns the exact same instance. So each distinct resource is built
once per process and reused.

Note: one cache per process. Under a multi-worker server each worker keeps its
own cache, which is fine — they just warm up independently.
"""

from __future__ import annotations

from functools import lru_cache

from ragbench.embed.providers import OpenAIEmbeddingProvider
from ragbench.embed.local_provider import LocalSTEmbeddingProvider
from ragbench.store.qdrant_store import QdrantStore
from ragbench.rerank.providers import CrossEncoderReranker


@lru_cache(maxsize=None)
def get_embedder(embed_kind: str, embed_model: str):
    if embed_kind == "openai":
        return OpenAIEmbeddingProvider(model=embed_model)
    elif embed_kind == "local":
        return LocalSTEmbeddingProvider(embed_model)
    raise ValueError("embed_kind must be 'openai' or 'local'")


@lru_cache(maxsize=None)
def get_store(collection: str, vector_size: int):
    return QdrantStore(collection_name=collection, vector_size=vector_size)


@lru_cache(maxsize=None)
def get_reranker(rerank_model: str):
    # the expensive one — loads a model on construction
    return CrossEncoderReranker(model=rerank_model)