"""
Benchmark/eval item schema.

Each eval item now declares two orthogonal tags:
  - slice:      the KIND of challenge (single_hop, multi_hop, unanswerable,
                conflicting_context, adversarial)
  - difficulty: how hard it is (easy, medium, hard)

Both are Optional for now so the legacy benchmarks/sample.jsonl (untagged) still
loads during the transition. Once evalset_v1 is authored and frozen, the manifest
step will assert every item in a versioned set is fully tagged.
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
    # eval-rigor tags (Optional during transition; required for a frozen set)
    slice: Optional[Slice] = None
    difficulty: Optional[Difficulty] = None