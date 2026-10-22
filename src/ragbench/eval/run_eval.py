"""
End-to-end eval orchestration.

Lifts the scoring + aggregation loop out of run_benchmark.py into one reusable
function that returns a structured EvalReport (and optionally writes result
files for the dashboard to read later).

Items are loaded via load_evalset(), which validates tags and verifies the
dataset's content hash against its manifest — so every report is stamped with
the exact frozen dataset (version + hash) it ran on. For each item it calls
run_query(), then scores it exactly as run_benchmark.py does.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from statistics import mean
from typing import List, Optional

from ragbench.bench.schema import BenchItem
from ragbench.pipeline.run_query import run_query
from ragbench.utils.config import RunConfig
from ragbench.eval.dataset import load_evalset
from ragbench.utils.eval_schema import (
    EvalItemResult,
    EvalAggregates,
    EvalReport,
)


def _score_item(bi: BenchItem, record) -> EvalItemResult:
    is_answerable = bi.gold_answer is not None
    refused = record.judge.refused
    faithful = record.judge.faithful

    answer_success = false_refusal = hallucination = correct_refusal = None
    if is_answerable:
        false_refusal = refused
        answer_success = (not refused) and bool(faithful)
    else:
        hallucination = not refused
        correct_refusal = refused

    return EvalItemResult(
        id=bi.id,
        question=bi.question,
        gold_answer=bi.gold_answer,
        is_answerable=is_answerable,
        slice=bi.slice.value if bi.slice else None,
        difficulty=bi.difficulty.value if bi.difficulty else None,
        record=record,
        faithful=faithful,
        refused=refused,
        answer_success=answer_success,
        false_refusal=false_refusal,
        hallucination=hallucination,
        correct_refusal=correct_refusal,
    )


def _rate(flags: List[bool]) -> float:
    """Mean over a subset; 0.0 when the subset is empty (matches run_benchmark)."""
    return float(mean(1 if f else 0 for f in flags)) if flags else 0.0


def _aggregate(items: List[EvalItemResult]) -> EvalAggregates:
    answerable = [it for it in items if it.is_answerable]
    unanswerable = [it for it in items if not it.is_answerable]

    return EvalAggregates(
        n_items=len(items),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        faithfulness_rate=_rate([bool(it.faithful) for it in items]),
        answer_success_rate=_rate([it.answer_success for it in answerable]),
        false_refusal_rate=_rate([it.false_refusal for it in answerable]),
        hallucination_rate=_rate([it.hallucination for it in unanswerable]),
        correct_refusal_rate=_rate([it.correct_refusal for it in unanswerable]),
        avg_total_latency_ms=(
            float(mean(it.record.metrics.latency.total_ms for it in items)) if items else 0.0
        ),
        total_cost_usd=sum(it.record.metrics.total_cost_usd for it in items),
    )


def _persist(report: EvalReport, persist_dir: str) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(os.path.join(persist_dir, f"eval_{ts}.json"), "w") as f:
        f.write(report.model_dump_json(indent=2))

    agg = report.aggregates
    with open(os.path.join(persist_dir, f"eval_summary_{ts}.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in agg.model_dump().items():
            f.write(f"{k},{v}\n")


def run_eval(
    dataset_path: str,
    config: RunConfig,
    persist_dir: Optional[str] = None,
) -> EvalReport:
    bench_items, meta = load_evalset(dataset_path)

    items: List[EvalItemResult] = []
    for bi in bench_items:
        record = run_query(bi.question, config)
        items.append(_score_item(bi, record))

    report = EvalReport(
        dataset=dataset_path,
        dataset_version=meta.get("version"),
        dataset_hash=meta.get("hash"),
        config_version=config.version,
        aggregates=_aggregate(items),
        items=items,
    )

    if persist_dir:
        _persist(report, persist_dir)

    return report