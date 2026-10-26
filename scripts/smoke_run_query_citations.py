"""Batch 2 (WS1 citations) — verify run_query threads chunk ids into the answer
and parses the cited ids back onto the record.

Run from repo root (Qdrant up, collection indexed):
    python scripts/smoke_run_query_citations.py configs/query.yaml
"""

import sys

from ragbench.utils.config import RunConfig
from ragbench.pipeline.run_query import run_query

cfg = RunConfig.from_yaml(sys.argv[1] if len(sys.argv) > 1 else "configs/query.yaml")

for q in ["How do I scale a Kubernetes Deployment?",
          "How do I roll back a Helm release?"]:
    rec = run_query(q, cfg)
    ctx = rec.reranked_chunks or rec.retrieved_chunks
    ctx_ids = [c.payload.get("chunk_id") for c in ctx]
    print(f"\nQ: {q}")
    print(f"A: {rec.answer}")
    print(f"context ids shown : {ctx_ids}")
    print(f"cited_chunk_ids   : {rec.cited_chunk_ids}")
    print(f"answer_prompt_ver : {rec.answer_prompt_version}")
    fabricated = [c for c in rec.cited_chunk_ids if c not in ctx_ids]
    print(f"fabricated (not in context): {fabricated}")
