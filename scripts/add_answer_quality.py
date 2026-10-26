"""Batch (WS1 answer-quality) — score correctness + relevance on your latest run
and write an 'answer_quality' block into it (so the dashboard can show it without
a full re-run).

Run from repo root:  python scripts/add_answer_quality.py

Makes LLM judge calls: 2 per ANSWERABLE item (correctness + relevance), so it
needs OPENAI_API_KEY. Unanswerable items are skipped — they have no gold answer
to match. ~80 calls on gpt-4.1-mini for a 50-item set; costs cents.

Reporting choice (matches what we agreed): the headline rates are STRICT —
`correct_rate` counts only "correct", `relevant_rate` counts only "relevant".
The partial / incorrect fractions are kept alongside so nothing is hidden.
"""

import glob
import json
import os
from collections import Counter

from ragbench.eval.judge import judge_answer_quality, ANSWER_QUALITY_PROMPT_VERSION

CORRECTNESS_BUCKETS = ["correct", "partially_correct", "incorrect"]
RELEVANCE_BUCKETS = ["relevant", "partial", "irrelevant"]


def latest_eval_json():
    files = sorted(
        (f for f in glob.glob("results/eval_*.json") if "summary" not in os.path.basename(f)),
        key=os.path.getmtime, reverse=True,
    )
    if not files:
        raise SystemExit("No results/eval_*.json found.")
    return files[0]


path = latest_eval_json()
with open(path) as f:
    report = json.load(f)

# judge every answerable item (unanswerable items have no gold answer to match)
per_item = {}  # id -> {"correctness": ..., "relevance": ...}
print("judging answerable items...")
for it in report["items"]:
    if not it.get("is_answerable") or not it.get("gold_answer"):
        continue
    v, _ = judge_answer_quality(it["question"], it["record"]["answer"], it["gold_answer"])
    per_item[it["id"]] = {"correctness": v["correctness"], "relevance": v["relevance"]}
    print(f"  {it['id']:8s}  {v['correctness']:18s} {v['relevance']}")


def summ(ids):
    rows = [per_item[i] for i in ids if i in per_item]
    if not rows:
        return None
    n = len(rows)
    c = Counter(r["correctness"] for r in rows)
    r = Counter(r["relevance"] for r in rows)
    return {
        "n": n,
        "correctness": {b: c.get(b, 0) / n for b in CORRECTNESS_BUCKETS},
        "relevance": {b: r.get(b, 0) / n for b in RELEVANCE_BUCKETS},
        "correct_rate": c.get("correct", 0) / n,     # strict: partial excluded
        "relevant_rate": r.get("relevant", 0) / n,   # strict: partial excluded
    }


slices = sorted({it["slice"] for it in report["items"] if it.get("slice")})
answer_quality = {
    "prompt_version": ANSWER_QUALITY_PROMPT_VERSION,
    "overall": summ([it["id"] for it in report["items"]]),
    "by_slice": {s: summ([it["id"] for it in report["items"] if it.get("slice") == s]) for s in slices},
}
report["answer_quality"] = answer_quality
with open(path, "w") as f:
    json.dump(report, f, indent=2)

o = answer_quality["overall"]
print(f"\nfile: {path}")
print(f"\n=== overall (n={o['n']}) ===")
print(f"  correct            {o['correctness']['correct']:.2f}   <- strict correctness rate")
print(f"  partially_correct  {o['correctness']['partially_correct']:.2f}")
print(f"  incorrect          {o['correctness']['incorrect']:.2f}")
print(f"  relevant           {o['relevance']['relevant']:.2f}   <- strict relevance rate")
print(f"  partial            {o['relevance']['partial']:.2f}")
print(f"  irrelevant         {o['relevance']['irrelevant']:.2f}")
print("\n=== by slice  (correct/partial/incorrect  |  relevant/partial/irrelevant) ===")
for s, m in answer_quality["by_slice"].items():
    if m:
        c, r = m["correctness"], m["relevance"]
        print(f"  {s:20s} {c['correct']:.2f}/{c['partially_correct']:.2f}/{c['incorrect']:.2f}"
              f"   |  {r['relevant']:.2f}/{r['partial']:.2f}/{r['irrelevant']:.2f}   (n={m['n']})")
print(f"\nwrote 'answer_quality' block into {path}")