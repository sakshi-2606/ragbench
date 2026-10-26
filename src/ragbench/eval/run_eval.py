"""
End-to-end eval orchestration.

Loads a frozen, tagged eval set, runs run_query() on each item, scores it, and
rolls up: overall aggregates, per-slice/difficulty breakdowns, Wilson CIs,
retrieval ranking (dense top-10), and context precision/recall (reranked context).
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional

from ragbench.bench.schema import BenchItem
from ragbench.pipeline.run_query import run_query
from ragbench.utils.config import RunConfig
from ragbench.eval.dataset import load_evalset
from ragbench.eval.bootstrap import compute_confidence
from ragbench.eval.ranking import ranking_metrics
from ragbench.eval.judge import judge_answer_quality, ANSWER_QUALITY_PROMPT_VERSION
from ragbench.utils.eval_schema import (
    EvalItemResult,
    EvalAggregates,
    EvalReport,
    GroupMetrics,
    RetrievalMetrics,
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

    # retrieval ranking on the dense top-10
    ranked_ids = [c.payload.get("chunk_id") for c in record.retrieved_chunks]
    rm = ranking_metrics(ranked_ids, bi.relevant_chunk_ids)
    retrieval = RetrievalMetrics(**rm) if rm else None

    # context precision/recall on the RERANKED context actually fed to the LLM
    context_precision = context_recall = None
    rel = set(bi.relevant_chunk_ids)
    if rel:
        ctx_chunks = record.reranked_chunks or record.retrieved_chunks
        ctx_ids = [c.payload.get("chunk_id") for c in ctx_chunks]
        if ctx_ids:
            hits = sum(1 for cid in ctx_ids if cid in rel)
            context_precision = hits / len(ctx_ids)
            context_recall = hits / len(rel)

    # answer quality — judged, answerable items only. Reuse the item's judge
    # model so correctness/relevance use the same judge as faithfulness.
    correctness = relevance = None
    if is_answerable and bi.gold_answer:
        aq, _ = judge_answer_quality(
            bi.question, record.answer, bi.gold_answer, model=record.judge_model
        )
        correctness = aq["correctness"]
        relevance = aq["relevance"]

    # citation accuracy — answered (non-refused), labelled items only.
    # C = cited, X = ids the model saw (context), R = relevant (gold).
    citation_validity = citation_precision = citation_recall = None
    if rel and not refused:
        C = set(record.cited_chunk_ids)
        ctx_chunks = record.reranked_chunks or record.retrieved_chunks
        X = {c.payload.get("chunk_id") for c in ctx_chunks}
        citation_recall = len(C & rel) / len(rel)   # 0 if answered but cited nothing
        if C:
            citation_precision = len(C & rel) / len(C)
            citation_validity = len(C & X) / len(C)

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
        retrieval=retrieval,
        context_precision=context_precision,
        context_recall=context_recall,
        correctness=correctness,
        relevance=relevance,
        citation_validity=citation_validity,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
    )


def _rate(flags) -> float:
    flags = list(flags)
    return float(mean(1 if f else 0 for f in flags)) if flags else 0.0


def _opt_rate(flags) -> Optional[float]:
    flags = list(flags)
    return _rate(flags) if flags else None


def _group_metrics(items: List[EvalItemResult]) -> GroupMetrics:
    answerable = [it for it in items if it.is_answerable]
    unanswerable = [it for it in items if not it.is_answerable]
    return GroupMetrics(
        n=len(items),
        faithfulness_rate=_rate(bool(it.faithful) for it in items),
        answer_success_rate=_opt_rate(it.answer_success for it in answerable),
        false_refusal_rate=_opt_rate(it.false_refusal for it in answerable),
        hallucination_rate=_opt_rate(it.hallucination for it in unanswerable),
        correct_refusal_rate=_opt_rate(it.correct_refusal for it in unanswerable),
        avg_total_latency_ms=(
            float(mean(it.record.metrics.latency.total_ms for it in items)) if items else 0.0
        ),
        total_cost_usd=sum(it.record.metrics.total_cost_usd for it in items),
    )


def _aggregate(items: List[EvalItemResult]) -> EvalAggregates:
    answerable = [it for it in items if it.is_answerable]
    unanswerable = [it for it in items if not it.is_answerable]
    return EvalAggregates(
        n_items=len(items),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        faithfulness_rate=_rate(bool(it.faithful) for it in items),
        answer_success_rate=_rate(it.answer_success for it in answerable),
        false_refusal_rate=_rate(it.false_refusal for it in answerable),
        hallucination_rate=_rate(it.hallucination for it in unanswerable),
        correct_refusal_rate=_rate(it.correct_refusal for it in unanswerable),
        avg_total_latency_ms=(
            float(mean(it.record.metrics.latency.total_ms for it in items)) if items else 0.0
        ),
        total_cost_usd=sum(it.record.metrics.total_cost_usd for it in items),
    )


def _breakdown(items: List[EvalItemResult], key) -> Dict[str, GroupMetrics]:
    names = sorted({key(it) for it in items if key(it) is not None})
    return {name: _group_metrics([it for it in items if key(it) == name]) for name in names}


def _confidence(items: List[EvalItemResult]) -> dict:
    item_dicts = [
        {
            "is_answerable": it.is_answerable, "faithful": it.faithful,
            "answer_success": it.answer_success, "false_refusal": it.false_refusal,
            "hallucination": it.hallucination, "correct_refusal": it.correct_refusal,
            "slice": it.slice, "difficulty": it.difficulty,
        }
        for it in items
    ]
    return compute_confidence(item_dicts)


def _avg_retrieval(rms: List[Optional[RetrievalMetrics]]) -> Optional[dict]:
    rms = [r for r in rms if r is not None]
    if not rms:
        return None
    n = len(rms)
    ks = list(rms[0].precision.keys())
    return {
        "n": n,
        "mrr": sum(r.mrr for r in rms) / n,
        "precision": {k: sum(r.precision[k] for r in rms) / n for k in ks},
        "recall": {k: sum(r.recall[k] for r in rms) / n for k in ks},
        "ndcg": {k: sum(r.ndcg[k] for r in rms) / n for k in ks},
    }


def _retrieval_summary(items: List[EvalItemResult]) -> dict:
    slices = sorted({it.slice for it in items if it.slice})
    return {
        "overall": _avg_retrieval([it.retrieval for it in items]),
        "by_slice": {s: _avg_retrieval([it.retrieval for it in items if it.slice == s]) for s in slices},
    }


def _avg_context(sub: List[EvalItemResult]) -> Optional[dict]:
    ps = [it.context_precision for it in sub if it.context_precision is not None]
    rs = [it.context_recall for it in sub if it.context_recall is not None]
    if not ps:
        return None
    return {"n": len(ps), "precision": sum(ps) / len(ps), "recall": sum(rs) / len(rs)}


def _context_summary(items: List[EvalItemResult]) -> dict:
    slices = sorted({it.slice for it in items if it.slice})
    return {
        "overall": _avg_context(items),
        "by_slice": {s: _avg_context([it for it in items if it.slice == s]) for s in slices},
    }


_CORRECTNESS_BUCKETS = ["correct", "partially_correct", "incorrect"]
_RELEVANCE_BUCKETS = ["relevant", "partial", "irrelevant"]


def _avg_answer_quality(sub: List[EvalItemResult]) -> Optional[dict]:
    rows = [it for it in sub if it.correctness is not None]
    if not rows:
        return None
    n = len(rows)
    c = Counter(it.correctness for it in rows)
    r = Counter(it.relevance for it in rows)
    return {
        "n": n,
        "correctness": {b: c.get(b, 0) / n for b in _CORRECTNESS_BUCKETS},
        "relevance": {b: r.get(b, 0) / n for b in _RELEVANCE_BUCKETS},
        "correct_rate": c.get("correct", 0) / n,     # strict: partial excluded
        "relevant_rate": r.get("relevant", 0) / n,   # strict: partial excluded
    }


def _answer_quality_summary(items: List[EvalItemResult]) -> dict:
    slices = sorted({it.slice for it in items if it.slice})
    return {
        "prompt_version": ANSWER_QUALITY_PROMPT_VERSION,
        "overall": _avg_answer_quality(items),
        "by_slice": {s: _avg_answer_quality([it for it in items if it.slice == s]) for s in slices},
    }


def _mean_opt(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _avg_citations(sub: List[EvalItemResult]) -> Optional[dict]:
    # recall population = answered, labelled items (recall is defined there)
    scored = [it for it in sub if it.citation_recall is not None]
    if not scored:
        return None
    cited = [it for it in scored if it.citation_precision is not None]
    return {
        "n": len(scored),              # answered + labelled
        "n_cited": len(cited),         # of those, how many cited >= 1 source
        "validity": _mean_opt(it.citation_validity for it in cited),
        "precision": _mean_opt(it.citation_precision for it in cited),
        "recall": _mean_opt(it.citation_recall for it in scored),
    }


def _citation_summary(items: List[EvalItemResult]) -> dict:
    slices = sorted({it.slice for it in items if it.slice})
    return {
        "overall": _avg_citations(items),
        "by_slice": {s: _avg_citations([it for it in items if it.slice == s]) for s in slices},
    }


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
        by_slice=_breakdown(items, lambda it: it.slice),
        by_difficulty=_breakdown(items, lambda it: it.difficulty),
        confidence=_confidence(items),
        retrieval=_retrieval_summary(items),
        context=_context_summary(items),
        answer_quality=_answer_quality_summary(items),
        citations=_citation_summary(items),
        items=items,
    )

    if persist_dir:
        _persist(report, persist_dir)

    return report