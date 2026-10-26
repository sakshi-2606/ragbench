"""Batch (WS1 context) — compute context precision/recall on your latest run
and write a 'context' block into it (so the dashboard can show it without a re-run).
Run from repo root:  python scripts/add_context.py
Offline — joins the latest results json with the dataset's relevance labels.
"""

import glob
import json
import os
from statistics import mean

DATASET = "benchmarks/evalset_v2.jsonl"


def latest_eval_json():
    files = sorted(
        (f for f in glob.glob("results/eval_*.json") if "summary" not in os.path.basename(f)),
        key=os.path.getmtime, reverse=True,
    )
    if not files:
        raise SystemExit("No results/eval_*.json found.")
    return files[0]


# relevance labels from the dataset
labels = {}
with open(DATASET) as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            labels[row["id"]] = set(row.get("relevant_chunk_ids", []))

path = latest_eval_json()
with open(path) as f:
    report = json.load(f)

per_item = {}  # id -> (precision, recall) or None
for it in report["items"]:
    rel = labels.get(it["id"], set())
    if not rel:
        per_item[it["id"]] = None
        continue
    ctx = it["record"].get("reranked_chunks") or it["record"].get("retrieved_chunks") or []
    ctx_ids = [c["payload"].get("chunk_id") for c in ctx]
    if not ctx_ids:
        per_item[it["id"]] = None
        continue
    hits = sum(1 for cid in ctx_ids if cid in rel)
    per_item[it["id"]] = (hits / len(ctx_ids), hits / len(rel))


def summ(ids):
    vals = [per_item[i] for i in ids if per_item.get(i) is not None]
    if not vals:
        return None
    return {"n": len(vals), "precision": mean(v[0] for v in vals), "recall": mean(v[1] for v in vals)}


slices = sorted({it["slice"] for it in report["items"] if it.get("slice")})
context = {
    "overall": summ([it["id"] for it in report["items"]]),
    "by_slice": {s: summ([it["id"] for it in report["items"] if it.get("slice") == s]) for s in slices},
}
report["context"] = context
with open(path, "w") as f:
    json.dump(report, f, indent=2)

o = context["overall"]
print("file:", path)
print(f"context precision (overall): {o['precision']:.2f}")
print(f"context recall    (overall): {o['recall']:.2f}   (n={o['n']})")
print("\n=== by slice (context recall / precision) ===")
for s, m in context["by_slice"].items():
    if m:
        print(f"  {s:20s} recall {m['recall']:.2f}   precision {m['precision']:.2f}   (n={m['n']})")
print(f"\nwrote 'context' block into {path}")