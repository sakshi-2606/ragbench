"""
RAGBench eval dashboard (Streamlit).

Read-only viewer: it loads a saved results/eval_*.json (written by run_eval) and
renders the scorecard, stage-latency chart, dataset split, and per-item table.
It never calls the pipeline or the API — it only reads files.

Run from the repo root:
    streamlit run dashboard/app.py
"""

import glob
import json
import os

import pandas as pd
import streamlit as st

RESULTS_DIR = os.getenv("RAGBENCH_RESULTS", "results")

st.set_page_config(page_title="RAGBench Eval", layout="wide")
st.title("RAGBench — Eval Dashboard")
st.caption("Read-only view of a saved eval run")

# ---- pick a run ----
files = sorted(
    (f for f in glob.glob(os.path.join(RESULTS_DIR, "eval_*.json")) if "summary" not in os.path.basename(f)),
    reverse=True,
)
if not files:
    st.warning(f"No eval_*.json files found in {RESULTS_DIR}/. Run an eval first (POST /eval or run_eval).")
    st.stop()

choice = st.selectbox("Run", files, format_func=os.path.basename)
with open(choice) as f:
    report = json.load(f)

agg = report["aggregates"]
items = report["items"]

# ---- run metadata ----
m = st.columns(4)
m[0].markdown(f"**Dataset**  \n`{report['dataset']}`")
m[1].markdown(f"**Config version**  \n`{report['config_version']}`")
m[2].markdown(f"**Timestamp**  \n`{report['timestamp'][:19]}`")
m[3].markdown(f"**Items**  \n`{agg['n_items']}`")

st.divider()

# ---- scorecard ----
st.subheader("Scorecard")
r1 = st.columns(4)
r1[0].metric("Faithfulness", f"{agg['faithfulness_rate'] * 100:.1f}%")
r1[1].metric("Answer success", f"{agg['answer_success_rate'] * 100:.1f}%")
r1[2].metric("Correct refusal", f"{agg['correct_refusal_rate'] * 100:.1f}%")
r1[3].metric("False refusal", f"{agg['false_refusal_rate'] * 100:.1f}%")

r2 = st.columns(4)
avg_cost = agg["total_cost_usd"] / agg["n_items"] if agg["n_items"] else 0
r2[0].metric("Hallucination", f"{agg['hallucination_rate'] * 100:.1f}%")
r2[1].metric("Avg latency", f"{agg['avg_total_latency_ms']:.0f} ms")
r2[2].metric("Total cost", f"${agg['total_cost_usd']:.5f}")
r2[3].metric("Avg cost / query", f"${avg_cost:.6f}")

st.divider()

# ---- stage latency + dataset split ----
left, right = st.columns([1.5, 1])

with left:
    st.subheader("Avg latency by stage (ms)")
    stage_keys = ["embed_query_ms", "retrieve_ms", "rerank_ms", "answer_ms", "judge_ms"]
    labels = ["embed", "retrieve", "rerank", "answer", "judge"]
    n = len(items) or 1
    avg_stage = {
        lbl: sum(it["record"]["metrics"]["latency"].get(k, 0) for it in items) / n
        for lbl, k in zip(labels, stage_keys)
    }
    st.bar_chart(pd.DataFrame({"avg_ms": avg_stage}))

with right:
    st.subheader("Dataset split")
    st.metric("Answerable", agg["n_answerable"])
    st.metric("Unanswerable", agg["n_unanswerable"])

st.divider()

# ---- per-item table ----
st.subheader("Per-item results")


def outcome(it):
    if it["is_answerable"]:
        if it["false_refusal"]:
            return "false refusal"
        return "success" if it["answer_success"] else "fail"
    if it["hallucination"]:
        return "hallucination"
    return "correct refusal" if it["correct_refusal"] else "-"


rows = [
    {
        "id": it["id"],
        "question": it["question"],
        "type": "answerable" if it["is_answerable"] else "unanswerable",
        "faithful": it["faithful"],
        "refused": it["refused"],
        "outcome": outcome(it),
        "latency_ms": it["record"]["metrics"]["latency"]["total_ms"],
        "cost_usd": round(it["record"]["metrics"]["total_cost_usd"], 6),
    }
    for it in items
]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)