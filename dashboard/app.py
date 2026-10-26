"""
RAGBench eval dashboard (Streamlit).

Read-only viewer over a saved results/eval_*.json: scorecard (with Wilson CIs),
stage latency, dataset split, per-slice/difficulty breakdowns, retrieval ranking
(precision@k / recall@k / MRR / NDCG@k), and the per-item table.

Run from the repo root:
    streamlit run dashboard/app.py
"""

import glob
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

RESULTS_DIR = os.getenv("RAGBENCH_RESULTS", "results")

st.set_page_config(page_title="RAGBench Eval", layout="wide")
st.title("RAGBench — Eval Dashboard")
st.caption("Read-only view of a saved eval run")

files = sorted(
    (f for f in glob.glob(os.path.join(RESULTS_DIR, "eval_*.json")) if "summary" not in os.path.basename(f)),
    key=os.path.getmtime, reverse=True,
)
if not files:
    st.warning(f"No eval_*.json files found in {RESULTS_DIR}/. Run an eval first.")
    st.stop()

choice = st.selectbox("Run", files, format_func=os.path.basename)
with open(choice) as f:
    report = json.load(f)

agg = report["aggregates"]
items = report["items"]
conf = report.get("confidence", {})
agg_ci = conf.get("aggregate", {})
slice_ci = conf.get("by_slice", {})


def ci_caption(metric):
    c = agg_ci.get(metric)
    return f"95% CI {c['low'] * 100:.0f}–{c['high'] * 100:.0f}%" if c else ""


# ---- run metadata ----
m = st.columns(4)
m[0].markdown(f"**Dataset**  \n`{report['dataset']}`")
m[1].markdown(f"**Dataset version**  \n`{report.get('dataset_version') or '—'}`  ·  `{(report.get('dataset_hash') or '')[:12]}`")
m[2].markdown(f"**Config version**  \n`{report['config_version']}`")
m[3].markdown(f"**Items**  \n`{agg['n_items']}`")

st.divider()

# ---- scorecard ----
st.subheader("Scorecard")
if conf:
    st.caption("Intervals are 95% Wilson confidence intervals.")
r1 = st.columns(4)
r1[0].metric("Faithfulness", f"{agg['faithfulness_rate'] * 100:.1f}%"); r1[0].caption(ci_caption("faithfulness_rate"))
r1[1].metric("Answer success", f"{agg['answer_success_rate'] * 100:.1f}%"); r1[1].caption(ci_caption("answer_success_rate"))
r1[2].metric("Correct refusal", f"{agg['correct_refusal_rate'] * 100:.1f}%"); r1[2].caption(ci_caption("correct_refusal_rate"))
r1[3].metric("False refusal", f"{agg['false_refusal_rate'] * 100:.1f}%"); r1[3].caption(ci_caption("false_refusal_rate"))
r2 = st.columns(4)
avg_cost = agg["total_cost_usd"] / agg["n_items"] if agg["n_items"] else 0
r2[0].metric("Hallucination", f"{agg['hallucination_rate'] * 100:.1f}%"); r2[0].caption(ci_caption("hallucination_rate"))
r2[1].metric("Avg latency", f"{agg['avg_total_latency_ms']:.0f} ms")
r2[2].metric("Total cost", f"${agg['total_cost_usd']:.5f}")
r2[3].metric("Avg cost / query", f"${avg_cost:.6f}")

st.divider()

# ---- stage latency + split ----
left, right = st.columns([1.5, 1])
with left:
    st.subheader("Avg latency by stage (ms)")
    stage_keys = ["embed_query_ms", "retrieve_ms", "rerank_ms", "answer_ms", "judge_ms"]
    labels = ["embed", "retrieve", "rerank", "answer", "judge"]
    n = len(items) or 1
    stage_df = pd.DataFrame(
        {"stage": labels,
         "avg_ms": [sum(it["record"]["metrics"]["latency"].get(k, 0) for it in items) / n for k in stage_keys]}
    )
    chart = (alt.Chart(stage_df).mark_bar()
             .encode(x=alt.X("stage", sort=labels, title=None), y=alt.Y("avg_ms", title="avg ms"), tooltip=["stage", "avg_ms"])
             .properties(height=300))
    st.altair_chart(chart, use_container_width=True)
with right:
    st.subheader("Dataset split")
    st.metric("Answerable", agg["n_answerable"])
    st.metric("Unanswerable", agg["n_unanswerable"])

st.divider()


# ---- answer-quality breakdowns ----
def breakdown_df(groups: dict) -> pd.DataFrame:
    rows = []
    for name, mx in groups.items():
        rows.append({
            "group": name, "n": mx["n"], "faithfulness": mx["faithfulness_rate"],
            "answer_success": mx.get("answer_success_rate"), "false_refusal": mx.get("false_refusal_rate"),
            "hallucination": mx.get("hallucination_rate"), "correct_refusal": mx.get("correct_refusal_rate"),
            "avg_latency_ms": round(mx["avg_total_latency_ms"]),
        })
    return pd.DataFrame(rows)


RATE_COLS = ["faithfulness", "answer_success", "false_refusal", "hallucination", "correct_refusal"]
rate_fmt = {c: st.column_config.NumberColumn(format="%.2f") for c in RATE_COLS}
by_slice = report.get("by_slice", {})
by_difficulty = report.get("by_difficulty", {})

if by_slice:
    st.subheader("Answer success by slice")
    slice_df = breakdown_df(by_slice)
    ans = slice_df.dropna(subset=["answer_success"]).sort_values("answer_success", ascending=False).copy()
    if not ans.empty:
        ans["ci_low"] = ans["group"].map(lambda g: (slice_ci.get(g, {}).get("answer_success_rate") or {}).get("low"))
        ans["ci_high"] = ans["group"].map(lambda g: (slice_ci.get(g, {}).get("answer_success_rate") or {}).get("high"))
        order = list(ans["group"])
        x = alt.X("group", sort=order, title=None)
        base = alt.Chart(ans)
        bars = base.mark_bar(opacity=0.85).encode(
            x=x, y=alt.Y("answer_success:Q", title="answer success", scale=alt.Scale(domain=[0, 1])),
            tooltip=["group", "answer_success", "n", "ci_low", "ci_high"])
        layers = [bars]
        if ans["ci_low"].notna().any():
            whisk = base.mark_rule(strokeWidth=2, color="#8a93a3").encode(x=x, y="ci_low:Q", y2="ci_high:Q")
            layers.append(whisk)
        st.altair_chart(alt.layer(*layers).properties(height=280), use_container_width=True)
    st.subheader("Breakdown by slice")
    st.dataframe(slice_df, use_container_width=True, hide_index=True, column_config=rate_fmt)
    st.subheader("Breakdown by difficulty")
    diff_order = {"easy": 0, "medium": 1, "hard": 2}
    diff_df = breakdown_df(by_difficulty).sort_values("group", key=lambda s: s.map(diff_order))
    st.dataframe(diff_df, use_container_width=True, hide_index=True, column_config=rate_fmt)

# ---- retrieval ranking ----
ret = report.get("retrieval", {})
ov = ret.get("overall")
if ov:
    st.divider()
    st.subheader("Retrieval ranking (dense top-10)")
    rr = st.columns(4)
    rr[0].metric("MRR", f"{ov['mrr']:.2f}"); rr[0].caption(f"over {ov['n']} labeled items")
    rr[1].metric("Recall@10", f"{ov['recall']['10']:.2f}")
    rr[2].metric("NDCG@10", f"{ov['ndcg']['10']:.2f}")
    rr[3].metric("Precision@3", f"{ov['precision']['3']:.2f}")

    kdf = pd.DataFrame(
        [{"k": int(k), "precision": ov["precision"][k], "recall": ov["recall"][k], "ndcg": ov["ndcg"][k]}
         for k in ov["precision"]]
    )
    num_fmt = {c: st.column_config.NumberColumn(format="%.2f") for c in ["precision", "recall", "ndcg"]}
    st.dataframe(kdf, use_container_width=True, hide_index=True, column_config=num_fmt)

    # context precision/recall on the reranked context the LLM actually sees
    cx = report.get("context", {}).get("overall")
    if cx:
        cc = st.columns(4)
        cc[0].metric("Context precision", f"{cx['precision']:.2f}"); cc[0].caption("relevant share of the reranked context")
        cc[1].metric("Context recall", f"{cx['recall']:.2f}"); cc[1].caption("relevant chunks kept after rerank")

    cbs = report.get("context", {}).get("by_slice", {})
    bs = ret.get("by_slice", {})
    rows = [{"slice": s, "mrr": mx["mrr"], "recall@10": mx["recall"]["10"], "ndcg@10": mx["ndcg"]["10"],
             "ctx_recall": (cbs.get(s) or {}).get("recall"), "n": mx["n"]}
            for s, mx in bs.items() if mx]
    if rows:
        sdf = pd.DataFrame(rows).sort_values("mrr")
        st.caption("MRR by slice — lower = the first relevant chunk ranks worse (watch adversarial).")
        bar = (alt.Chart(sdf).mark_bar()
               .encode(x=alt.X("slice", sort=list(sdf["slice"]), title=None),
                       y=alt.Y("mrr", title="MRR", scale=alt.Scale(domain=[0, 1])),
                       tooltip=["slice", "mrr", "recall@10", "ndcg@10", "n"])
               .properties(height=240))
        st.altair_chart(bar, use_container_width=True)
        rank_fmt = {c: st.column_config.NumberColumn(format="%.2f") for c in ["mrr", "recall@10", "ndcg@10", "ctx_recall"]}
        st.dataframe(sdf, use_container_width=True, hide_index=True, column_config=rank_fmt)

# ---- answer quality (correctness + relevance) ----
aq = report.get("answer_quality", {})
aq_ov = aq.get("overall")
if aq_ov:
    st.divider()
    st.subheader("Answer quality (answerable items)")
    cc = aq_ov["correctness"]
    rr = aq_ov["relevance"]
    a = st.columns(4)
    a[0].metric("Correct (strict)", f"{aq_ov['correct_rate'] * 100:.0f}%")
    a[0].caption(f"partial {cc['partially_correct'] * 100:.0f}%  ·  incorrect {cc['incorrect'] * 100:.0f}%")
    a[1].metric("Relevant (strict)", f"{aq_ov['relevant_rate'] * 100:.0f}%")
    a[1].caption(f"partial {rr['partial'] * 100:.0f}%  ·  irrelevant {rr['irrelevant'] * 100:.0f}%")
    a[2].metric("Judged items", aq_ov["n"])
    a[2].caption(f"judge prompt {aq.get('prompt_version', '?')}")

    aq_bs = aq.get("by_slice", {})
    rows = [{"slice": s,
             "correct": m["correctness"]["correct"],
             "partial": m["correctness"]["partially_correct"],
             "incorrect": m["correctness"]["incorrect"],
             "relevant": m["relevance"]["relevant"],
             "irrelevant": m["relevance"]["irrelevant"],
             "n": m["n"]}
            for s, m in aq_bs.items() if m]
    if rows:
        adf = pd.DataFrame(rows).sort_values("correct")
        comp = adf.melt(id_vars=["slice", "n"], value_vars=["correct", "partial", "incorrect"],
                        var_name="bucket", value_name="frac")
        order = ["correct", "partial", "incorrect"]
        colors = alt.Scale(domain=order, range=["#3b9a57", "#d6a93b", "#c25b56"])
        st.caption("Correctness composition by slice — green correct, amber partial, red incorrect (watch adversarial).")
        bar = (alt.Chart(comp).mark_bar()
               .encode(x=alt.X("slice", sort=list(adf["slice"]), title=None),
                       y=alt.Y("frac", title="share", scale=alt.Scale(domain=[0, 1])),
                       color=alt.Color("bucket", sort=order, scale=colors,
                                       legend=alt.Legend(title=None, orient="top")),
                       order=alt.Order("bucket", sort="ascending"),
                       tooltip=["slice", "bucket", "frac", "n"])
               .properties(height=260))
        st.altair_chart(bar, use_container_width=True)
        aq_fmt = {c: st.column_config.NumberColumn(format="%.2f")
                  for c in ["correct", "partial", "incorrect", "relevant", "irrelevant"]}
        st.dataframe(adf, use_container_width=True, hide_index=True, column_config=aq_fmt)

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
        "id": it["id"], "slice": it.get("slice"), "difficulty": it.get("difficulty"),
        "question": it["question"], "faithful": it["faithful"], "refused": it["refused"],
        "outcome": outcome(it), "latency_ms": it["record"]["metrics"]["latency"]["total_ms"],
        "cost_usd": round(it["record"]["metrics"]["total_cost_usd"], 6),
    }
    for it in items
]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)