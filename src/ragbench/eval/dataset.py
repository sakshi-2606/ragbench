"""
Eval dataset loading + integrity.

A frozen eval set is identified two ways:
  - a human version label ("v1"), stored in a sidecar manifest, and
  - a content hash computed from the items themselves.

load_evalset() parses + validates the items, recomputes the content hash, and
(if a manifest sits beside the file) checks that the hash still matches. A
mismatch means the file was edited without rebuilding the manifest — the loader
refuses, so a version label can't silently drift.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Tuple

from ragbench.bench.schema import BenchItem


def _raw_items(path: str) -> List[dict]:
    """Read raw dict rows, skipping blank lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def content_hash(raw_items: List[dict]) -> str:
    """Order-independent fingerprint of the item set.

    Items are sorted by id and serialized canonically (sorted keys), so the
    hash depends only on the content, not on line order or key order.
    """
    norm = sorted(raw_items, key=lambda r: r["id"])
    blob = json.dumps(norm, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def manifest_path_for(dataset_path: str) -> str:
    return dataset_path.replace(".jsonl", ".manifest.json")


def load_evalset(path: str) -> Tuple[List[BenchItem], dict]:
    """Load + validate an eval set. Returns (items, meta) where meta has
    'hash', 'version' (None if no manifest), and 'n_items'."""
    raw = _raw_items(path)
    items = [BenchItem(**r) for r in raw]

    # a frozen set must be fully tagged
    untagged = [it.id for it in items if it.slice is None or it.difficulty is None]
    if untagged:
        raise ValueError(f"Untagged items in {path}: {untagged}")

    file_hash = content_hash(raw)
    version = None

    mpath = manifest_path_for(path)
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
        version = manifest.get("version")
        if manifest.get("content_hash") != file_hash:
            raise ValueError(
                f"Dataset hash mismatch for {path}: file={file_hash} "
                f"manifest={manifest.get('content_hash')}. The dataset changed "
                "without a manifest rebuild — rebuild the manifest (and bump the "
                "version if this was an intentional change)."
            )

    return items, {"hash": file_hash, "version": version, "n_items": len(items)}