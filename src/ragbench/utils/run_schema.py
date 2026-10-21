"""
Per-query structured record for the RAG service.

This extends the original retrieval-only schema into the FULL end-to-end
record returned by POST /query: metadata + retrieval + rerank + answer +
judge + metrics (per-stage latency, tokens, cost).

Note: the old `RunRecord` / `Timing` models (retrieval-only) are replaced by
the models below. Nothing in the benchmark path imported them, so this is an
extension, not a breaking change — but if any import of `RunRecord` or
`Timing` surfaces when you run the smoke test, flag it and we'll add a shim.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    """One retrieved chunk. `dense_score` is the vector-similarity score;
    `rerank_score` is filled only for chunks that survived reranking."""
    text: str
    payload: Dict[str, Any] = {}
    dense_score: Optional[float] = None
    rerank_score: Optional[float] = None


# --------------------------------------------------------------------------- #
# Metrics (cross-cutting)
# --------------------------------------------------------------------------- #
class StageLatency(BaseModel):
    """Milliseconds spent in each stage. `answer_ms` and `judge_ms` are the
    two that the benchmark never measured before this slice."""
    embed_query_ms: int = 0
    retrieve_ms: int = 0
    rerank_ms: int = 0
    answer_ms: int = 0
    judge_ms: int = 0
    total_ms: int = 0


class TokenUsage(BaseModel):
    """Token counts pulled from the OpenAI `usage` object on each LLM call."""
    answer_prompt_tokens: int = 0
    answer_completion_tokens: int = 0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    total_tokens: int = 0


class QueryMetrics(BaseModel):
    latency: StageLatency
    tokens: TokenUsage
    total_cost_usd: float = 0.0


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
class JudgeVerdict(BaseModel):
    """What the faithfulness judge returns, plus the refusal flag derived from
    `is_refusal`. The gold-answer-dependent metrics (answer_success,
    false_refusal, hallucination) are intentionally NOT here — they belong to
    eval, since a live query has no gold answer."""
    faithful: Optional[bool] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    refused: bool = False


# --------------------------------------------------------------------------- #
# Top-level record
# --------------------------------------------------------------------------- #
class QueryRecord(BaseModel):
    # --- metadata ---
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    config_version: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- retrieval ---
    embedding_model: str
    dense_top_k: int
    retrieved_chunks: List[RetrievedChunk] = []

    # --- rerank (populated only when use_rerank is True) ---
    use_rerank: bool = False
    rerank_model: Optional[str] = None
    rerank_top_n: int = 0
    reranked_chunks: List[RetrievedChunk] = []

    # --- answer ---
    answer: str
    answer_model: str

    # --- judge ---
    judge: JudgeVerdict
    judge_model: str

    # --- metrics ---
    metrics: QueryMetrics