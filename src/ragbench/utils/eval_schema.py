"""
Structured result schema for a full eval run (the /eval counterpart to the
per-query QueryRecord).

An eval run is: for each dataset item, run the query AND score it against its
gold answer. So each item result wraps the full QueryRecord plus the
gold-answer-dependent flags that /query deliberately leaves out. The report
then rolls those up into aggregate rates.

Scoring mirrors run_benchmark.py exactly — no new metrics in this slice.
Each item result now also carries its slice + difficulty tags, so per-slice /
per-difficulty breakdowns can be computed later.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from ragbench.utils.run_schema import QueryRecord


class EvalItemResult(BaseModel):
    id: str
    question: str
    gold_answer: Optional[str] = None
    is_answerable: bool                 # gold_answer is not None

    # tags carried from the dataset item (for per-slice / per-difficulty views)
    slice: Optional[str] = None
    difficulty: Optional[str] = None

    record: QueryRecord                 # the full per-query record

    # --- scoring flags (mirror run_benchmark.py) ---
    faithful: Optional[bool] = None
    refused: bool = False
    # answerable items only:
    answer_success: Optional[bool] = None
    false_refusal: Optional[bool] = None
    # unanswerable items only:
    hallucination: Optional[bool] = None
    correct_refusal: Optional[bool] = None


class EvalAggregates(BaseModel):
    n_items: int
    n_answerable: int
    n_unanswerable: int

    # rates (each computed over the relevant subset, as in run_benchmark)
    faithfulness_rate: float            # over all items
    answer_success_rate: float          # over answerable
    false_refusal_rate: float           # over answerable
    hallucination_rate: float           # over unanswerable
    correct_refusal_rate: float         # over unanswerable

    # ops rollups
    avg_total_latency_ms: float
    total_cost_usd: float


class EvalReport(BaseModel):
    eval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset: str                        # path to the eval dataset
    config_version: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    aggregates: EvalAggregates
    items: List[EvalItemResult] = []