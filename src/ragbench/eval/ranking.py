"""
Retrieval ranking metrics.

Given the ranked list of retrieved chunk IDs and the set of relevant chunk IDs
for a query, compute precision@k, recall@k, MRR, and NDCG@k (binary relevance).

  precision@k : fraction of the top-k that are relevant
  recall@k    : fraction of the relevant chunks that appear in the top-k
  MRR         : reciprocal of the rank of the first relevant chunk (0 if none)
  NDCG@k      : DCG@k / ideal DCG@k, with gain 1 for a relevant chunk

Returns None when there are no relevant chunks (e.g. unanswerable items) — those
are excluded from ranking, since recall is undefined without a relevant set.
"""

from __future__ import annotations

import math
from typing import List, Optional

KS = [3, 5, 10]


def ranking_metrics(ranked_ids: List[str], relevant_ids: List[str], ks=KS) -> Optional[dict]:
    rel = set(relevant_ids)
    if not rel:
        return None

    # MRR — rank of the first relevant hit
    mrr = 0.0
    for i, cid in enumerate(ranked_ids):
        if cid in rel:
            mrr = 1.0 / (i + 1)
            break

    precision, recall, ndcg = {}, {}, {}
    for k in ks:
        topk = ranked_ids[:k]
        hits = sum(1 for cid in topk if cid in rel)
        precision[str(k)] = hits / k
        recall[str(k)] = hits / len(rel)

        dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(topk) if cid in rel)
        ideal_hits = min(len(rel), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        ndcg[str(k)] = (dcg / idcg) if idcg > 0 else 0.0

    return {"n_relevant": len(rel), "mrr": mrr,
            "precision": precision, "recall": recall, "ndcg": ndcg}