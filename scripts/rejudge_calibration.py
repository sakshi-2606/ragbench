"""
Re-judge the SAME calibration rows with the current judge prompt and compare to
the previous judge, both scored against your (unchanged) human labels.

This is the calibration loop: change the judge prompt -> re-run here -> see
whether sensitivity / kappa improve, and which rows flipped. It does NOT rebuild
the set or need re-labeling, so it's a clean apples-to-apples comparison.

Run from repo root (OPENAI_API_KEY set; Qdrant not needed):
    python scripts/rejudge_calibration.py
"""

import csv
import glob
import json
import os

from ragbench.eval.judge import judge_faithfulness, JUDGE_PROMPT_VERSION

CAL_DIR = "calibration"
VALID = {"faithful", "unfaithful"}
MODEL = "gpt-4.1-mini"


def cohen_kappa(a, b):
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return po, pe, (float("nan") if pe == 1 else (po - pe) / (1 - pe))


def metrics(human, verdict):
    ids = sorted(human)
    h = [human[i] for i in ids]
    v = [verdict[i] for i in ids]
    tp = sum(1 for x, y in zip(h, v) if x == "unfaithful" and y == "unfaithful")
    fn = sum(1 for x, y in zip(h, v) if x == "unfaithful" and y == "faithful")
    fp = sum(1 for x, y in zip(h, v) if x == "faithful" and y == "unfaithful")
    tn = sum(1 for x, y in zip(h, v) if x == "faithful" and y == "faithful")
    _, _, k = cohen_kappa(h, v)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return {"kappa": k, "sensitivity": sens, "specificity": spec,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def fmt(m):
    s = "n/a" if m["sensitivity"] is None else f"{m['sensitivity']:.2f}"
    sp = "n/a" if m["specificity"] is None else f"{m['specificity']:.2f}"
    c = m["confusion"]
    return (f"kappa={m['kappa']:.2f}  sensitivity={s}  specificity={sp}  "
            f"[tn={c['tn']} fp={c['fp']} fn={c['fn']} tp={c['tp']}]")


def main():
    csv_path = sorted(glob.glob(os.path.join(CAL_DIR, "blind_labels_*.csv")))[-1]
    key_path = csv_path.replace("blind_labels_", "key_").replace(".csv", ".json")
    with open(key_path) as f:
        key = json.load(f)

    human, rows = {}, {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rid = row["row_id"]
            lab = (row.get("human_label") or "").strip().lower()
            if lab not in VALID:
                raise SystemExit(f"Row {rid} not labeled.")
            human[rid] = lab
            rows[rid] = {"question": row["question"],
                         "context": [c for c in row["context"].split("\n\n") if c.strip()],
                         "answer": row["answer"]}

    old_verdict = {rid: ("faithful" if key[rid]["judge_faithful"] else "unfaithful") for rid in human}

    print(f"re-judging {len(human)} rows with judge prompt {JUDGE_PROMPT_VERSION} ...")
    new_verdict = {}
    for rid in sorted(human):
        v, _ = judge_faithfulness(rows[rid]["question"], rows[rid]["answer"], rows[rid]["context"], model=MODEL)
        new_verdict[rid] = "faithful" if v.get("faithful") else "unfaithful"

    old_m = metrics(human, old_verdict)
    new_m = metrics(human, new_verdict)

    print("\n=== against your human labels ===")
    print("OLD judge :", fmt(old_m))
    print("NEW judge :", fmt(new_m))

    flipped = [rid for rid in sorted(human) if old_verdict[rid] != new_verdict[rid]]
    print(f"\nrows whose verdict changed ({len(flipped)}):")
    for rid in flipped:
        agree = "now matches you" if new_verdict[rid] == human[rid] else "now disagrees with you"
        print(f"  {rid} ({key[rid]['item_id']}, {key[rid]['source']}): "
              f"{old_verdict[rid]} -> {new_verdict[rid]}  | you={human[rid]}  [{agree}]")


if __name__ == "__main__":
    main()