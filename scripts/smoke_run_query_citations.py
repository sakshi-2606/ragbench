"""Batch 2 (WS1 citations) — verify run_query threads chunk ids into the answer
and parses the cited ids back onto the record.

Pulls the first few ANSWERABLE items (those with relevance labels) straight from
the eval set, so the questions are ones the corpus can actually answer and we see
real citations flow. Also prints relevant ids next to cited ids — a preview of
the citation scoring coming in batch 3.

Run from repo root (Qdrant up, collection indexed):
    python scripts/smoke_run_query_citations.py configs/query.yaml
"""

import json
import sys

from ragbench.utils.config import RunConfig
from ragbench.pipeline.run_query import run_query

cfg = RunConfig.from_yaml(sys.argv[1] if len(sys.argv) > 1 else "configs/query.yaml")

items = []
with open("benchmarks/evalset_v2.jsonl") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("relevant_chunk_ids"):   # answerable + labelled
            items.append(row)
        if len(items) >= 3:
            break

for row in items:
    rec = run_query(row["question"], cfg)
    ctx = rec.reranked_chunks or rec.retrieved_chunks
    ctx_ids = [c.payload.get("chunk_id") for c in ctx]
    rel = row["relevant_chunk_ids"]
    print(f"\n[{row['id']}] {row['question']}")
    print(f"A: {rec.answer}")
    print(f"context ids shown : {ctx_ids}")
    print(f"relevant ids      : {rel}")
    print(f"cited_chunk_ids   : {rec.cited_chunk_ids}")
    print(f"fabricated        : {[c for c in rec.cited_chunk_ids if c not in ctx_ids]}")
    print(f"answer_prompt_ver : {rec.answer_prompt_version}")