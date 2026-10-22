"""
Judge-bias experiment: concise vs verbose, with position-bias control.

For a sample of real answers we build a VERBOSE variant — longer and more
elaborate but with the SAME facts (no new claims). Each concise-vs-verbose pair
is judged in both slot orders. Two biases fall out:

  - Position bias  = how often the verdict FLIPS when we swap the order.
  - Verbosity bias = among the consistent verdicts, how often the (equally
                     faithful, just longer) verbose answer wins.

Run from repo root (OPENAI_API_KEY set; reads latest results/eval_*.json):
    python scripts/run_pairwise_bias.py
"""

import glob
import json
import os
import random
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI

from ragbench.eval.pairwise import compare_both_orders, PAIRWISE_PROMPT_VERSION

load_dotenv()

N = 30
SEED = 42
GEN_MODEL = "gpt-4.1-mini"

VERBOSE_SYSTEM = """Rewrite the answer to be noticeably LONGER and more elaborate:
add framing, restate and expand the existing points, add transitions and hedges.
DO NOT add any new factual claim that was not in the original answer — keep every
fact identical, just say it at greater length. Return only the rewritten answer."""


def latest_eval_json():
    files = sorted(f for f in glob.glob("results/eval_*.json") if "summary" not in os.path.basename(f))
    if not files:
        raise SystemExit("No results/eval_*.json found — run an eval first.")
    return files[-1]


def context_of(item):
    rec = item["record"]
    chunks = rec.get("reranked_chunks") or rec.get("retrieved_chunks") or []
    return [c["text"] for c in chunks]


def make_verbose(client, answer):
    resp = client.chat.completions.create(
        model=GEN_MODEL, temperature=0.5,
        messages=[{"role": "system", "content": VERBOSE_SYSTEM},
                  {"role": "user", "content": answer}],
    )
    return resp.choices[0].message.content.strip()


def main():
    src = latest_eval_json()
    with open(src) as f:
        items = json.load(f)["items"]

    pool = [it for it in items if it["is_answerable"] and not it["refused"] and it["record"]["answer"].strip()]
    random.seed(SEED)
    random.shuffle(pool)
    sample = pool[:N]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    rows = []
    print(f"running {len(sample)} concise-vs-verbose comparisons (both orders each) ...")
    for it in sample:
        concise = it["record"]["answer"]
        verbose = make_verbose(client, concise)
        res = compare_both_orders(it["question"], context_of(it), concise, verbose, "concise", "verbose")
        res.update({"id": it["id"], "concise_len": len(concise), "verbose_len": len(verbose)})
        rows.append(res)

    n = len(rows)
    inconsistent = [r for r in rows if not r["consistent"]]
    consistent = [r for r in rows if r["consistent"]]
    verbose_wins = [r for r in consistent if r["winner"] == "verbose"]
    concise_wins = [r for r in consistent if r["winner"] == "concise"]
    ties = [r for r in consistent if r["winner"] == "tie"]
    decisive = len(verbose_wins) + len(concise_wins)

    print("\n===== position bias =====")
    print(f"pairs                 : {n}")
    print(f"verdict flipped on swap: {len(inconsistent)}/{n}  = {len(inconsistent)/n:.0%}   <- position bias")

    print("\n===== verbosity bias (consistent verdicts only) =====")
    print(f"consistent verdicts   : {len(consistent)}")
    print(f"  verbose won         : {len(verbose_wins)}")
    print(f"  concise won         : {len(concise_wins)}")
    print(f"  tie                 : {len(ties)}")
    if decisive:
        print(f"verbose win-rate (decisive): {len(verbose_wins)/decisive:.0%}   "
              f"(>50% = judge favors the longer, equally-faithful answer)")

    print("\n===== length =====")
    print(f"avg concise chars: {mean(r['concise_len'] for r in rows):.0f}")
    print(f"avg verbose chars: {mean(r['verbose_len'] for r in rows):.0f}")

    out = {
        "n": n, "pairwise_prompt_version": PAIRWISE_PROMPT_VERSION,
        "position_flip_rate": len(inconsistent) / n,
        "consistent": len(consistent), "verbose_wins": len(verbose_wins),
        "concise_wins": len(concise_wins), "ties": len(ties),
        "verbose_win_rate_decisive": (len(verbose_wins) / decisive) if decisive else None,
        "rows": rows,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/pairwise_bias.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results/pairwise_bias.json")


if __name__ == "__main__":
    main()