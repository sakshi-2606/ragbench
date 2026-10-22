"""
Confidence intervals for eval metrics.

Two methods, each for what it's good at:
  - Wilson score interval  -> single PROPORTIONS (rates). Analytic, honest at the
    boundaries: 8/8 gives a wide interval like [0.68, 1.0], not [1.0, 1.0].
  - Bootstrap               -> things without a clean closed form: paired metric
    DIFFERENCES (config A vs B on the same items) for rates and for continuous
    values like latency/cost.

Both capture *sampling* uncertainty (which items we happened to include). Neither
captures the generator's run-to-run nondeterminism — that needs repeated runs.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist, mean
from typing import List, Optional


def wilson_ci(flags, ci: float = 0.95) -> Optional[dict]:
    """Wilson score interval for a proportion = mean of boolean flags."""
    flags = [1 if f else 0 for f in flags]
    n = len(flags)
    if n == 0:
        return None
    p = sum(flags) / n
    z = NormalDist().inv_cdf((1 + ci) / 2)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return {"point": p, "low": max(0.0, center - half), "high": min(1.0, center + half),
            "n": n, "method": "wilson"}


def bootstrap_ci(flags, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Optional[dict]:
    """Percentile bootstrap CI for a rate. (Prefer wilson_ci for plain proportions.)"""
    flags = [1 if f else 0 for f in flags]
    n = len(flags)
    if n == 0:
        return None
    point = sum(flags) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(flags[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"point": point, "low": means[int((1 - ci) / 2 * n_boot)],
            "high": means[int((1 + ci) / 2 * n_boot) - 1], "n": n, "method": "bootstrap"}


def _paired(a, b, n_boot, ci, seed, binary):
    if binary:
        a = [1 if f else 0 for f in a]
        b = [1 if f else 0 for f in b]
    else:
        a = [float(x) for x in a]
        b = [float(x) for x in b]
    n = len(a)
    if n == 0 or len(b) != n:
        return None
    point = mean(b) - mean(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int((1 - ci) / 2 * n_boot)]
    hi = diffs[int((1 + ci) / 2 * n_boot) - 1]
    return {"diff": point, "low": lo, "high": hi, "n": n, "crosses_zero": lo <= 0 <= hi, "method": "bootstrap"}


def paired_diff_ci(flags_a, flags_b, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Optional[dict]:
    """Paired bootstrap CI for the difference (B - A) of two RATES on the same items."""
    return _paired(flags_a, flags_b, n_boot, ci, seed, binary=True)


def paired_mean_diff_ci(values_a, values_b, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Optional[dict]:
    """Paired bootstrap CI for the difference (B - A) of two MEANS of continuous
    values (e.g. latency, cost) on the same items."""
    return _paired(values_a, values_b, n_boot, ci, seed, binary=False)


# --- confidence block for a whole eval (aggregate + per-slice + per-difficulty) ---

_METRICS = [
    ("faithfulness_rate", "all", "faithful"),
    ("answer_success_rate", "answerable", "answer_success"),
    ("false_refusal_rate", "answerable", "false_refusal"),
    ("hallucination_rate", "unanswerable", "hallucination"),
    ("correct_refusal_rate", "unanswerable", "correct_refusal"),
]


def _group_cis(items: List[dict], ci: float) -> dict:
    ans = [it for it in items if it["is_answerable"]]
    un = [it for it in items if not it["is_answerable"]]
    subsets = {"all": items, "answerable": ans, "unanswerable": un}
    out = {}
    for name, subset_key, flag in _METRICS:
        subset = subsets[subset_key]
        out[name] = wilson_ci([it[flag] for it in subset], ci) if subset else None
    return out


def compute_confidence(items: List[dict], ci: float = 0.95) -> dict:
    slices = sorted({it["slice"] for it in items if it.get("slice")})
    diffs = sorted({it["difficulty"] for it in items if it.get("difficulty")})
    return {
        "method": "wilson",
        "ci": ci,
        "aggregate": _group_cis(items, ci),
        "by_slice": {s: _group_cis([it for it in items if it["slice"] == s], ci) for s in slices},
        "by_difficulty": {d: _group_cis([it for it in items if it["difficulty"] == d], ci) for d in diffs},
    }