"""
Bootstrap confidence intervals for eval metrics.

A rate (e.g. faithfulness) is a mean over a finite set of items. The bootstrap
estimates how much that mean would wobble if we'd sampled different items:
resample the items WITH replacement to the same size, recompute the mean,
repeat n_boot times, and take percentiles of the resulting distribution.

Non-parametric — no distributional assumptions. This captures *sampling*
uncertainty (which items we happened to include). It does NOT capture the
generator's run-to-run nondeterminism — that needs repeated eval runs.
"""

from __future__ import annotations

import random
from typing import List, Optional


def bootstrap_ci(flags, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Optional[dict]:
    """CI for a rate = mean of boolean flags. Returns {point, low, high, n} or
    None for an empty input."""
    flags = [1 if f else 0 for f in flags]
    n = len(flags)
    if n == 0:
        return None
    point = sum(flags) / n

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total = 0
        for _ in range(n):
            total += flags[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot) - 1]
    return {"point": point, "low": lo, "high": hi, "n": n}


def paired_diff_ci(flags_a, flags_b, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Optional[dict]:
    """CI for the difference (B - A) of two rates measured on the SAME items.

    Paired: we resample item indices once per iteration and compute both rates
    on that same resample, so item-to-item variation cancels. `crosses_zero`
    tells you whether the difference is distinguishable from noise.
    """
    a = [1 if f else 0 for f in flags_a]
    b = [1 if f else 0 for f in flags_b]
    n = len(a)
    if n == 0 or len(b) != n:
        return None
    point = sum(b) / n - sum(a) / n

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ma = sum(a[i] for i in idx) / n
        mb = sum(b[i] for i in idx) / n
        diffs.append(mb - ma)
    diffs.sort()

    lo = diffs[int((1 - ci) / 2 * n_boot)]
    hi = diffs[int((1 + ci) / 2 * n_boot) - 1]
    return {"diff": point, "low": lo, "high": hi, "n": n, "crosses_zero": lo <= 0 <= hi}


# --- confidence block for a whole eval (aggregate + per-slice + per-difficulty) ---

_METRICS = [
    ("faithfulness_rate", "all", "faithful"),
    ("answer_success_rate", "answerable", "answer_success"),
    ("false_refusal_rate", "answerable", "false_refusal"),
    ("hallucination_rate", "unanswerable", "hallucination"),
    ("correct_refusal_rate", "unanswerable", "correct_refusal"),
]


def _group_cis(items: List[dict], n_boot, ci, seed) -> dict:
    ans = [it for it in items if it["is_answerable"]]
    un = [it for it in items if not it["is_answerable"]]
    subsets = {"all": items, "answerable": ans, "unanswerable": un}
    out = {}
    for name, subset_key, flag in _METRICS:
        subset = subsets[subset_key]
        out[name] = bootstrap_ci([it[flag] for it in subset], n_boot, ci, seed) if subset else None
    return out


def compute_confidence(items: List[dict], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    """items: list of per-item result dicts (the EvalReport 'items' shape)."""
    slices = sorted({it["slice"] for it in items if it.get("slice")})
    diffs = sorted({it["difficulty"] for it in items if it.get("difficulty")})
    return {
        "n_boot": n_boot,
        "ci": ci,
        "seed": seed,
        "aggregate": _group_cis(items, n_boot, ci, seed),
        "by_slice": {s: _group_cis([it for it in items if it["slice"] == s], n_boot, ci, seed) for s in slices},
        "by_difficulty": {d: _group_cis([it for it in items if it["difficulty"] == d], n_boot, ci, seed) for d in diffs},
    }