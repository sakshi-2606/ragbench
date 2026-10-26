"""
Benchmark/eval item schema.

Each eval item carries two orthogonal tags (slice, difficulty) and now a
`relevant_chunk_ids` list — the ground-truth set of corpus chunk IDs that are
relevant to the question. Ranking metrics (precision@k, recall@k, MRR, NDCG)
score retrieved chunk IDs against this set. Empty for unanswerable items (no
relevant chunk exists); one ID for single-hop; several for multi-hop /
conflicting-context.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Slice(str, Enum):
    single_hop = "single_hop"
    multi_hop = "multi_hop"
    unanswerable = "unanswerable"
    conflicting_context = "conflicting_context"
    adversarial = "adversarial"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class BenchItem(BaseModel):
    id: str
    question: str
    gold_answer: Optional[str] = None
    must_contain: List[str] = []
    gold_doc_contains: List[str] = []
    slice: Optional[Slice] = None
    difficulty: Optional[Difficulty] = None
    # ground-truth relevant corpus chunk IDs (for retrieval ranking metrics)
    relevant_chunk_ids: List[str] = []