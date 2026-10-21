"""
End-to-end orchestration for a single query.

This lifts the four-stage chain that currently lives inside run_benchmark.py's
loop (retrieve -> rerank -> answer -> judge) into one reusable function, and
fills the two gaps that benchmark left open:
  - it TIMES the answer and judge stages (not just retrieval), and
  - it computes COST from the token usage via estimate_cost.

It returns a fully-populated QueryRecord. The FastAPI /query endpoint (batch 3)
will be a thin wrapper that just calls this and returns the record as JSON.
"""

from __future__ import annotations

import time

from ragbench.pipeline.simple_rag import run_rag
from ragbench.generation.answer import generate_answer
from ragbench.eval.judge import judge_faithfulness
from ragbench.eval.refusal import is_refusal
from ragbench.eval.pricing import estimate_cost
from ragbench.utils.config import RunConfig
from ragbench.utils.run_schema import (
    QueryRecord,
    RetrievedChunk,
    JudgeVerdict,
    QueryMetrics,
    StageLatency,
    TokenUsage,
)


def _ms() -> int:
    return int(time.perf_counter() * 1000)


def run_query(question: str, config: RunConfig) -> QueryRecord:
    # ---- stages 1 & 2: retrieve (+ optional rerank) ----
    # run_rag already times embed/retrieve/rerank internally.
    retr = run_rag(
        question=question,
        collection=config.retrieval.collection,
        embed_model=config.retrieval.embed_model,
        dense_top_k=config.retrieval.dense_top_k,
        use_rerank=config.rerank.enabled,
        rerank_model=config.rerank.model,
        rerank_top_n=config.rerank.top_n,
        embed_kind=config.retrieval.embed_kind,
    )
    context_chunks = retr["context_chunks"]

    # ---- stage 3: answer (timed here — this is the gap benchmark left) ----
    t = _ms()
    answer, ans_usage = generate_answer(
        question=question,
        context_chunks=context_chunks,
        model=config.answer.model,
    )
    answer_ms = _ms() - t

    # ---- stage 4: judge (timed here too) ----
    t = _ms()
    verdict, judge_usage = judge_faithfulness(
        question=question,
        answer=answer,
        context_chunks=context_chunks,
        model=config.judge.model,
    )
    judge_ms = _ms() - t

    refused = is_refusal(answer)

    # ---- cost (never computed before this slice) ----
    answer_cost = estimate_cost(
        config.answer.model,
        ans_usage.get("prompt_tokens", 0),
        ans_usage.get("completion_tokens", 0),
    )
    judge_cost = estimate_cost(
        config.judge.model,
        judge_usage.get("prompt_tokens", 0),
        judge_usage.get("completion_tokens", 0),
    )

    # ---- assemble the record ----
    retrieved_chunks = [
        RetrievedChunk(
            text=r["text"],
            payload=r.get("payload", {}),
            dense_score=r.get("score"),
        )
        for r in retr["dense_results"]
    ]
    reranked_chunks = [
        RetrievedChunk(
            text=r["text"],
            payload=r.get("payload", {}),
            dense_score=r.get("dense_score"),
            rerank_score=r.get("rerank_score"),
        )
        for r in retr.get("reranked_results", [])
    ]

    latency = StageLatency(
        embed_query_ms=retr["timings_ms"]["embed_query"],
        retrieve_ms=retr["timings_ms"]["retrieve"],
        rerank_ms=retr["timings_ms"]["rerank"],
        answer_ms=answer_ms,
        judge_ms=judge_ms,
        total_ms=retr["timings_ms"]["total_retrieval"] + answer_ms + judge_ms,
    )
    tokens = TokenUsage(
        answer_prompt_tokens=ans_usage.get("prompt_tokens", 0),
        answer_completion_tokens=ans_usage.get("completion_tokens", 0),
        judge_prompt_tokens=judge_usage.get("prompt_tokens", 0),
        judge_completion_tokens=judge_usage.get("completion_tokens", 0),
        total_tokens=ans_usage.get("total_tokens", 0) + judge_usage.get("total_tokens", 0),
    )
    metrics = QueryMetrics(
        latency=latency,
        tokens=tokens,
        total_cost_usd=answer_cost + judge_cost,
    )

    return QueryRecord(
        question=question,
        config_version=config.version,
        embedding_model=config.retrieval.embed_model,
        dense_top_k=config.retrieval.dense_top_k,
        retrieved_chunks=retrieved_chunks,
        use_rerank=config.rerank.enabled,
        rerank_model=config.rerank.model if config.rerank.enabled else None,
        rerank_top_n=config.rerank.top_n if config.rerank.enabled else 0,
        reranked_chunks=reranked_chunks,
        answer=answer,
        answer_model=config.answer.model,
        judge=JudgeVerdict(
            faithful=verdict.get("faithful"),
            confidence=verdict.get("confidence"),
            rationale=verdict.get("rationale"),
            refused=refused,
        ),
        judge_model=config.judge.model,
        metrics=metrics,
    )