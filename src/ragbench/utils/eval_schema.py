"""
Structured result schema for a full eval run (the /eval counterpart to the
per-query QueryRecord).

The report carries the overall aggregates, per-slice and per-difficulty
breakdowns, and a `confidence` block (Wilson CIs for each rate). It is stamped
with both provenance keys — config_version and dataset_version + dataset_hash.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ragbench.utils.run_schema import QueryRecord


class EvalItemResult(BaseModel):
    id: str
    question: str
    gold_answer: Optional[str] = None
    is_answerable: bool

    slice: Optional[str] = None
    difficulty: Optional[str] = None

    record: QueryRecord

    faithful: Optional[bool] = None
    refused: bool = False
    answer_success: Optional[bool] = None
    false_refusal: Optional[bool] = None
    hallucination: Optional[bool] = None
    correct_refusal: Optional[bool] = None


class GroupMetrics(BaseModel):
    n: int
    faithfulness_rate: float
    answer_success_rate: Optional[float] = None
    false_refusal_rate: Optional[float] = None
    hallucination_rate: Optional[float] = None
    correct_refusal_rate: Optional[float] = None
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
    # Wilson CIs for each rate (aggregate + per-slice + per-difficulty).
    confidence: Dict[str, Any] = {}
    items: List[EvalItemResult] = []