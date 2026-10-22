"""
Build a balanced, blind calibration set for judging the judge.

Why balanced: the judge rates almost everything "faithful", so a label set of
only real answers has no negatives — and Cohen's kappa needs variance to mean
anything. So we mix real answers (faithful side) with deliberately corrupted
answers (unfaithful side), run the judge on all of them, and hand YOU a blind
sheet to label. Batch 2 then compares your labels to the judge's verdicts.

Outputs (in calibration/):
  - blind_labels_<ts>.csv : question, context, answer, blank human_label  (you fill this)
  - key_<ts>.json         : per-row source, construction label, and judge verdict (hidden from you)

Run from repo root (Qdrant not needed; OPENAI_API_KEY set):
    python scripts/build_calibration_set.py
"""

import csv
import glob
import json
import os
import random
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from ragbench.eval.judge import judge_faithfulness

load_dotenv()

N_REAL = 15          # faithful side: keep the real answer
N_INJECT = 12        # unfaithful: inject one unsupported claim
N_FABRICATE = 3      # unfaithful: fabricate an answer to an unanswerable question
SEED = 42
JUDGE_MODEL = "gpt-4.1-mini"
CORRUPT_MODEL = "gpt-4.1-mini"
OUT_DIR = "calibration"


def latest_eval_json():
    files = sorted(f for f in glob.glob("results/eval_*.json") if "summary" not in os.path.basename(f))
    if not files:
        raise SystemExit("No results/eval_*.json found — run an eval first.")
    return files[-1]


def context_of(item):
    """The chunks the answer was grounded in (reranked if present, else dense)."""
    rec = item["record"]
    chunks = rec.get("reranked_chunks") or rec.get("retrieved_chunks") or []
    return [c["text"] for c in chunks]


def corrupt_answer(client, question, answer, context_chunks, mode):
    context = "\n".join(f"[chunk_{i}] {c}" for i, c in enumerate(context_chunks))
    if mode == "inject":
        sys = (
            "You rewrite an answer so it introduces EXACTLY ONE factual claim that "
            "is NOT supported by the provided context, while keeping the rest accurate "
            "and natural. Do not flag the change. Return only the rewritten answer."
        )
    else:  # fabricate
        sys = (
            "The context does NOT contain the answer to the question. Write a short, "
            "confident, specific answer anyway (it will be factually fabricated). "
            "Do not hedge or refuse. Return only the answer."
        )
    resp = client.chat.completions.create(
        model=CORRUPT_MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps({"question": question, "context": context, "answer": answer})},
        ],
    )
    return resp.choices[0].message.content.strip()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    src = latest_eval_json()
    with open(src) as f:
        report = json.load(f)
    items = report["items"]

    answerable = [it for it in items if it["is_answerable"] and not it["refused"]]
    unanswerable = [it for it in items if not it["is_answerable"]]

    random.seed(SEED)
    random.shuffle(answerable)
    random.shuffle(unanswerable)

    real_items = answerable[:N_REAL]
    inject_items = answerable[N_REAL:N_REAL + N_INJECT]
    fabricate_items = unanswerable[:N_FABRICATE]

    rows = []   # each: dict with row_id, question, context, answer + key fields

    def add_row(item, answer, source, construction_label, judge_faithful):
        rows.append({
            "question": item["question"],
            "context": "\n\n".join(context_of(item)),
            "answer": answer,
            "_item_id": item["id"],
            "_slice": item.get("slice"),
            "_source": source,
            "_construction_label": construction_label,   # faithful / unfaithful
            "_judge_faithful": judge_faithful,
        })

    # faithful side — reuse the judge verdict already in the report
    for it in real_items:
        add_row(it, it["record"]["answer"], "real", "faithful", bool(it["faithful"]))

    # unfaithful side — corrupt, then run the judge on the corrupted answer
    for it in inject_items:
        bad = corrupt_answer(client, it["question"], it["record"]["answer"], context_of(it), "inject")
        verdict, _ = judge_faithfulness(it["question"], bad, context_of(it), model=JUDGE_MODEL)
        add_row(it, bad, "inject", "unfaithful", bool(verdict.get("faithful")))

    for it in fabricate_items:
        bad = corrupt_answer(client, it["question"], it["record"]["answer"], context_of(it), "fabricate")
        verdict, _ = judge_faithfulness(it["question"], bad, context_of(it), model=JUDGE_MODEL)
        add_row(it, bad, "fabricate", "unfaithful", bool(verdict.get("faithful")))

    random.shuffle(rows)
    for i, r in enumerate(rows):
        r["row_id"] = f"c{i:02d}"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"blind_labels_{ts}.csv")
    key_path = os.path.join(OUT_DIR, f"key_{ts}.json")

    # blind CSV — nothing that reveals the answer
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "question", "context", "answer", "human_label"])
        for r in rows:
            w.writerow([r["row_id"], r["question"], r["context"], r["answer"], ""])

    # key — hidden truth + judge verdict
    key = {
        r["row_id"]: {
            "item_id": r["_item_id"],
            "slice": r["_slice"],
            "source": r["_source"],
            "construction_label": r["_construction_label"],
            "judge_faithful": r["_judge_faithful"],
        }
        for r in rows
    }
    with open(key_path, "w") as f:
        json.dump(key, f, indent=2)

    n_faith = sum(1 for r in rows if r["_construction_label"] == "faithful")
    print(f"source eval     : {src}")
    print(f"rows            : {len(rows)}  ({n_faith} faithful / {len(rows) - n_faith} unfaithful by construction)")
    print(f"blind CSV       : {csv_path}")
    print(f"key (hidden)    : {key_path}")
    print("\nNext: open the CSV, fill the human_label column with 'faithful' or 'unfaithful' for every row,")
    print("reading ONLY the question/context/answer. Do not open the key file while labeling.")


if __name__ == "__main__":
    main()