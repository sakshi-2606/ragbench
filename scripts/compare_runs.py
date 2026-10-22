"""
Compare two eval runs (config A vs config B) on the SAME dataset, with a paired
bootstrap CI on each metric difference (B - A).

    python scripts/compare_runs.py results/eval_openai.json results/eval_bge.json

Paired: both runs scored the same items, so we resample items and compute both
runs' metric on each resample. If a difference's CI crosses zero, the two
configs are not distinguishable on that metric from this data.
"""

import json
import sys

from ragbench.eval.bootstrap import paired_diff_ci, paired_mean_diff_ci


def load(path):
    with open(path) as f:
        return json.load(f)


def by_id(report):
    return {it["id"]: it for it in report["items"]}


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: compare_runs.py <A.json> <B.json>")
    A, B = load(sys.argv[1]), load(sys.argv[2])

    if A.get("dataset_hash") != B.get("dataset_hash"):
        print("WARNING: runs are on different dataset versions — pairing may be invalid.\n")

    ia, ib = by_id(A), by_id(B)
    ids = [i for i in ia if i in ib]
    answerable = [i for i in ids if ia[i]["is_answerable"]]
    unanswerable = [i for i in ids if not ia[i]["is_answerable"]]

    def lat(rep_item):
        return rep_item["record"]["metrics"]["latency"]["total_ms"]

    def cost(rep_item):
        return rep_item["record"]["metrics"]["total_cost_usd"]

    # (label, subset ids, value fn, is_rate)
    specs = [
        ("faithfulness",    ids,          lambda it: it["faithful"],        True),
        ("answer_success",  answerable,   lambda it: it["answer_success"],  True),
        ("false_refusal",   answerable,   lambda it: it["false_refusal"],   True),
        ("hallucination",   unanswerable, lambda it: it["hallucination"],   True),
        ("correct_refusal", unanswerable, lambda it: it["correct_refusal"], True),
        ("latency_ms",      ids,          lat,                              False),
        ("cost_usd",        ids,          cost,                             False),
    ]

    print(f"A = {sys.argv[1]}  (config {A.get('config_version')})")
    print(f"B = {sys.argv[2]}  (config {B.get('config_version')})")
    print(f"dataset {A.get('dataset_version')} / {A.get('dataset_hash')}   paired on {len(ids)} items\n")
    print(f"{'metric':16} {'A':>10} {'B':>10} {'diff (B-A)':>12}   {'95% CI':>18}   verdict")
    print("-" * 82)

    for label, subset, fn, is_rate in specs:
        a_vals = [fn(ia[i]) for i in subset]
        b_vals = [fn(ib[i]) for i in subset]
        res = paired_diff_ci(a_vals, b_vals) if is_rate else paired_mean_diff_ci(a_vals, b_vals)
        if res is None:
            continue
        if is_rate:
            a_pt = sum(1 for v in a_vals if v) / len(a_vals)
            b_pt = sum(1 for v in b_vals if v) / len(b_vals)
            a_s, b_s, d_s = f"{a_pt:.2f}", f"{b_pt:.2f}", f"{res['diff']:+.2f}"
            ci_s = f"[{res['low']:+.2f}, {res['high']:+.2f}]"
        else:
            a_pt = sum(a_vals) / len(a_vals)
            b_pt = sum(b_vals) / len(b_vals)
            a_s, b_s, d_s = f"{a_pt:.4g}", f"{b_pt:.4g}", f"{res['diff']:+.4g}"
            ci_s = f"[{res['low']:+.4g}, {res['high']:+.4g}]"
        verdict = "no sig. difference" if res["crosses_zero"] else "**significant**"
        print(f"{label:16} {a_s:>10} {b_s:>10} {d_s:>12}   {ci_s:>18}   {verdict}")

    print("\n('no sig. difference' = the 95% CI of the difference includes 0.)")


if __name__ == "__main__":
    main()