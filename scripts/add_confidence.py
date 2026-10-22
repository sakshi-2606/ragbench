"""Batch 1 (bootstrap CI) — compute CIs on your latest eval run and show them.
Run from repo root:  python scripts/add_confidence.py
Offline — no Qdrant/API. Reads the latest results/eval_*.json, prints each
rate with its 95% CI, and writes a 'confidence' block back into that file
(so the dashboard can show the intervals in the next batch).
"""

import glob
import json
import os

from ragbench.eval.bootstrap import compute_confidence


def latest_eval_json():
    files = sorted(f for f in glob.glob("results/eval_*.json") if "summary" not in os.path.basename(f))
    if not files:
        raise SystemExit("No results/eval_*.json found — run an eval first.")
    return files[-1]


def show(title, block):
    print(f"\n=== {title} ===")
    for metric, ci in block.items():
        if ci is None:
            continue
        print(f"  {metric:22s} {ci['point']:.2f}  [{ci['low']:.2f}, {ci['high']:.2f}]   (n={ci['n']})")


def main():
    path = latest_eval_json()
    with open(path) as f:
        report = json.load(f)

    conf = compute_confidence(report["items"], n_boot=1000, ci=0.95, seed=42)
    report["confidence"] = conf

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print("file:", path)
    print(f"({conf['n_boot']} resamples, {int(conf['ci']*100)}% CI)")
    show("aggregate", conf["aggregate"])
    for s, block in conf["by_slice"].items():
        show(f"slice: {s}", block)

    print(f"\nwrote 'confidence' block into {path}")


if __name__ == "__main__":
    main()