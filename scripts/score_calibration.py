"""
Score the judge against your human labels.

Reads the labeled blind CSV + its key, lines up human labels vs the judge's
verdicts on the same rows, and reports:
  - raw agreement and Cohen's kappa (chance-corrected agreement), computed by hand
  - the 2x2 confusion matrix (human x judge)
  - sensitivity: of the answers YOU called unfaithful, how many did the judge catch?
    (the direct measure of judge leniency)
  - as a secondary signal, how many deliberately-corrupted answers the judge caught

Run from repo root:
    python scripts/score_calibration.py
Offline — no Qdrant/API. Reads the latest calibration/blind_labels_*.csv.
"""

import csv
import glob
import json
import os

CAL_DIR = "calibration"
VALID = {"faithful", "unfaithful"}


def latest_labeled_csv():
    files = sorted(glob.glob(os.path.join(CAL_DIR, "blind_labels_*.csv")))
    if not files:
        raise SystemExit("No calibration/blind_labels_*.csv found.")
    return files[-1]


def kappa_label(k):
    if k != k:  # nan
        return "undefined (degenerate — no variance)"
    if k < 0.0:
        return "poor (worse than chance)"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def cohen_kappa(a, b):
    """a, b: equal-length lists of labels. Returns (po, pe, kappa)."""
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    kappa = float("nan") if pe == 1 else (po - pe) / (1 - pe)
    return po, pe, kappa


def main():
    csv_path = latest_labeled_csv()
    key_path = csv_path.replace("blind_labels_", "key_").replace(".csv", ".json")
    if not os.path.exists(key_path):
        raise SystemExit(f"Key not found next to CSV: {key_path}")

    with open(key_path) as f:
        key = json.load(f)

    human, judge, construction, bad_rows, missing = {}, {}, {}, [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rid = row["row_id"]
            lab = (row.get("human_label") or "").strip().lower()
            if lab not in VALID:
                (missing if lab == "" else bad_rows).append(rid)
                continue
            human[rid] = lab
            judge[rid] = "faithful" if key[rid]["judge_faithful"] else "unfaithful"
            construction[rid] = key[rid]["construction_label"]

    if missing or bad_rows:
        raise SystemExit(f"Fix labels first — blank: {missing}  invalid: {bad_rows}")

    ids = sorted(human)
    h = [human[i] for i in ids]
    j = [judge[i] for i in ids]
    c = [construction[i] for i in ids]
    n = len(ids)

    # confusion: human (rows) x judge (cols), positive class = "unfaithful"
    tp = sum(1 for x, y in zip(h, j) if x == "unfaithful" and y == "unfaithful")
    fn = sum(1 for x, y in zip(h, j) if x == "unfaithful" and y == "faithful")
    fp = sum(1 for x, y in zip(h, j) if x == "faithful" and y == "unfaithful")
    tn = sum(1 for x, y in zip(h, j) if x == "faithful" and y == "faithful")

    po, pe, kappa = cohen_kappa(h, j)
    sens = tp / (tp + fn) if (tp + fn) else None     # judge catches human-unfaithful
    spec = tn / (tn + fp) if (tn + fp) else None
    prec = tp / (tp + fp) if (tp + fp) else None

    # secondary: did the judge catch the deliberately-corrupted negatives?
    constr_neg = [i for i in ids if construction[i] == "unfaithful"]
    judge_caught = sum(1 for i in constr_neg if judge[i] == "unfaithful")

    # label-quality sanity: human vs construction
    hc_agree = sum(1 for i in ids if human[i] == construction[i]) / n

    print(f"labeled CSV   : {csv_path}")
    print(f"n items       : {n}")
    print(f"\nhuman labels  : faithful {h.count('faithful')} / unfaithful {h.count('unfaithful')}")
    print(f"judge verdicts: faithful {j.count('faithful')} / unfaithful {j.count('unfaithful')}")

    print("\n===== human vs judge =====")
    print(f"raw agreement : {po:.2f}")
    print(f"chance (pe)   : {pe:.2f}")
    print(f"Cohen's kappa : {kappa:.2f}  [{kappa_label(kappa)}]")
    print("\nconfusion (rows=human, cols=judge):")
    print(f"                judge:faithful   judge:unfaithful")
    print(f"  human:faithful      {tn:3d}              {fp:3d}")
    print(f"  human:unfaithful    {fn:3d}              {tp:3d}")
    print(f"\nsensitivity (judge catches your 'unfaithful') : {sens:.2f}" if sens is not None else "\nsensitivity: n/a")
    print(f"specificity (judge agrees on your 'faithful')  : {spec:.2f}" if spec is not None else "specificity: n/a")
    if prec is not None:
        print(f"precision   (judge 'unfaithful' that you agree) : {prec:.2f}")

    print("\n===== secondary signals =====")
    print(f"deliberately-corrupted negatives caught by judge: {judge_caught}/{len(constr_neg)}")
    print(f"your labels vs construction (sanity)            : {hc_agree:.2f} agreement")

    disagree = [i for i in ids if human[i] != judge[i]]
    print(f"\nhuman/judge disagreements ({len(disagree)}): {disagree}")

    # persist result
    out = {
        "csv": csv_path,
        "n": n,
        "raw_agreement": po,
        "chance_agreement": pe,
        "cohen_kappa": None if kappa != kappa else kappa,
        "kappa_interpretation": kappa_label(kappa),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "corrupted_caught": f"{judge_caught}/{len(constr_neg)}",
        "disagreements": disagree,
    }
    res_path = key_path.replace("key_", "calibration_result_")
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved result  : {res_path}")


if __name__ == "__main__":
    main()