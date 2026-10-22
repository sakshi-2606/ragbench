"""
Structured result schema for a full eval run (the /eval counterpart to the
per-query QueryRecord).

The report carries the overall aggregates plus per-slice and per-difficulty
breakdowns, so "faithfulness 1.0" can be split into "faithfulness on multi_hop
vs conflicting_context", etc. It is stamped with both provenance keys —
config_version and dataset_version + dataset_hash — so any result is reproducible.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ragbench.utils.run_schema import QueryRecord


class EvalItemResult(BaseModel):
    id: str
    question: str
    gold_answer: Optional[str] = None
    is_answerable: bool                 # gold_answer is not None

    slice: Optional[str] = None
    difficulty: Optional[str] = None

    record: QueryRecord                 # the full per-query record

    # --- scoring flags (mirror run_benchmark.py) ---
    faithful: Optional[bool] = None
    refused: bool = False
    answer_success: Optional[bool] = None
    false_refusal: Optional[bool] = None
    hallucination: Optional[bool] = None
    correct_refusal: Optional[bool] = None


class GroupMetrics(BaseModel):
    """Metrics for one group (a slice, or a difficulty level).

    A rate is None when it doesn't apply to this group — i.e. the subset it
    would be computed over is empty (e.g. hallucination_rate for an all-answerable
    slice). This keeps per-group reporting free of meaningless blanks.
    """
    n: int
    faithfulness_rate: float                        # over all items in the group
    answer_success_rate: Optional[float] = None     # over answerable in the group
    false_refusal_rate: Optional[float] = None      # over answerable in the group
    hallucination_rate: Optional[float] = None      # over unanswerable in the group
    correct_refusal_rate: Optional[float] = None    # over unanswerable in the group
    avg_total_latency_ms: float
    total_cost_usd: float


class EvalAggregates(BaseModel):
    n_items: int
    n_answerable: int
    n_unanswerable: int

    faithfulness_rate: float
    answer_success_rate: float
    false_refusal_rate: float
    hallucination_rate: float
    correct_refusal_rate: float

    avg_total_latency_ms: float
    total_cost_usd: float


class EvalReport(BaseModel):
    eval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset: str
    dataset_version: Optional[str] = None
    dataset_hash: Optional[str] = None
    config_version: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    aggregates: EvalAggregates
    by_slice: Dict[str, GroupMetrics] = {}
    by_difficulty: Dict[str, GroupMetrics] = {}
    items: List[EvalItemResult] = []