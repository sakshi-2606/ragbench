"""
Build (freeze) the manifest for an eval set.

Reads the dataset, asserts every item is tagged, and writes a sidecar
<dataset>.manifest.json capturing the version, content hash, item count, and
per-slice / per-difficulty distribution. Re-run this whenever you intentionally
change the dataset (and bump VERSION for a real new version).

Run from repo root:
    python scripts/build_manifest.py
"""

import datetime
import json
from collections import Counter

from ragbench.eval.dataset import _raw_items, content_hash, manifest_path_for

DATASET = "benchmarks/evalset_v1.jsonl"
VERSION = "v1"


def main():
    raw = _raw_items(DATASET)

    untagged = [r["id"] for r in raw if not r.get("slice") or not r.get("difficulty")]
    if untagged:
        raise SystemExit(f"Refusing to build manifest — untagged items: {untagged}")

    manifest = {
        "version": VERSION,
        "dataset": DATASET,
        "created": datetime.date.today().isoformat(),
        "n_items": len(raw),
        "content_hash": content_hash(raw),
        "n_answerable": sum(1 for r in raw if r.get("gold_answer") is not None),
        "n_unanswerable": sum(1 for r in raw if r.get("gold_answer") is None),
        "by_slice": dict(Counter(r["slice"] for r in raw)),
        "by_difficulty": dict(Counter(r["difficulty"] for r in raw)),
    }

    out = manifest_path_for(DATASET)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote", out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()