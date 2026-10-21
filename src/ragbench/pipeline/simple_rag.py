import time
from typing import Dict, Any

from ragbench.utils.resources import get_embedder, get_store, get_reranker


def ms():
    return int(time.perf_counter() * 1000)


def run_rag(
    question: str,
    collection: str,
    embed_model: str = "text-embedding-3-small",
    dense_top_k: int = 10,
    use_rerank: bool = True,
    rerank_model: str = "BAAI/bge-reranker-base",
    rerank_top_n: int = 3,
    embed_kind: str = "openai"
) -> Dict[str, Any]:

    # resources are now built once and reused (see utils/resources.py)
    embedder = get_embedder(embed_kind, embed_model)
    store = get_store(collection, embedder.dim)

    t1 = ms()
    qvec = embedder.embed_query(question)
    t_embed_q = ms() - t1

    t2 = ms()
    dense = store.search(qvec, top_k=dense_top_k)
    t_retrieve = ms() - t2

    candidates = [(r.payload.get("text", ""), dict(r.payload), getattr(r, "score", None)) for r in dense]
    context_chunks = [c[0] for c in candidates]

    t_rerank = 0
    reranked = []
    if use_rerank:
        rr = get_reranker(rerank_model)
        t3 = ms()
        reranked = rr.rerank(question, candidates, top_n=rerank_top_n)
        t_rerank = ms() - t3
        context_chunks = [it.text for it in reranked]

    return {
        "question": question,
        "embed_model": embed_model,
        "dense_top_k": dense_top_k,
        "use_rerank": use_rerank,
        "rerank_model": rerank_model if use_rerank else None,
        "rerank_top_n": rerank_top_n if use_rerank else 0,
        "timings_ms": {
            "embed_query": t_embed_q,
            "retrieve": t_retrieve,
            "rerank": t_rerank,
            "total_retrieval": t_embed_q + t_retrieve + t_rerank,
        },
        "dense_results": [{"text": c[0], "payload": c[1], "score": c[2]} for c in candidates],
        # surface the reranked items (with scores) so the per-query record
        # can show the before/after of reranking. base_score = original dense score.
        "reranked_results": [
            {
                "text": it.text,
                "payload": it.payload,
                "dense_score": it.base_score,
                "rerank_score": it.rerank_score,
            }
            for it in reranked
        ],
        "context_chunks": context_chunks,
        "top_dense_score": candidates[0][2] if candidates else 0,
        "top_rerank_score": reranked[0].rerank_score if reranked else None
    }