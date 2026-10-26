"""
Structured result schema for a full eval run.

Carries overall aggregates, per-slice/per-difficulty breakdowns, Wilson CIs
(`confidence`), retrieval ranking metrics (`retrieval`) on the dense top-10,
and context precision/recall (`context`) on the reranked context the LLM
actually sees. Stamped with config_version and dataset_version + dataset_hash.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ragbench.utils.run_schema import QueryRecord


class RetrievalMetrics(BaseModel):
    n_relevant: int
    mrr: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    ndcg: Dict[str, float]


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

    # retrieval ranking on the dense top-10 (vs relevant_chunk_ids)
    retrieval: Optional[RetrievalMetrics] = None
    # context quality on the RERANKED context actually fed to the LLM
    context_precision: Optional[float] = None   # relevant / context size
    context_recall: Optional[float] = None      # relevant kept / relevant total

    # answer quality on ANSWERABLE items (judged; None for unanswerable)
    correctness: Optional[str] = None   # correct / partially_correct / incorrect
    relevance: Optional[str] = None     # relevant / partial / irrelevant

    # citation accuracy on ANSWERED, labelled items (None for refusals/unanswerable)
    citation_validity: Optional[float] = None    # cited that are in context / cited
    citation_precision: Optional[float] = None   # cited that are relevant / cited
    citation_recall: Optional[float] = None      # relevant that were cited / relevant


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
    confidence: Dict[str, Any] = {}
    retrieval: Dict[str, Any] = {}        # dense-ranking summary
    context: Dict[str, Any] = {}          # reranked-context precision/recall summary
    answer_quality: Dict[str, Any] = {}   # correctness/relevance summary (answerable)
    citations: Dict[str, Any] = {}        # citation validity/precision/recall summary
    items: List[EvalItemResult] = []