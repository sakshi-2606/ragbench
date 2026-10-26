"""
Run the eval under a given config and save it to a named file.

    python scripts/run_eval_cli.py configs/query.yaml      results/eval_openai.json
    python scripts/run_eval_cli.py configs/query_bge.yaml   results/eval_bge.json

Named output (not timestamped) so the compare tool can find the two arms.
Qdrant up + the config's collection indexed; OPENAI_API_KEY set.
"""

import sys

from ragbench.utils.config import RunConfig
from ragbench.eval.run_eval import run_eval

DATASET = "benchmarks/evalset_v2.jsonl"


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_eval_cli.py <config.yaml> <out.json>")
    cfg_path, out_path = sys.argv[1], sys.argv[2]

    cfg = RunConfig.from_yaml(cfg_path)
    report = run_eval(DATASET, cfg, persist_dir=None)

    with open(out_path, "w") as f:
        f.write(report.model_dump_json(indent=2))

    a = report.aggregates
    ov = report.retrieval.get("overall") if report.retrieval else None
    cit = report.citations.get("overall") if report.citations else None
    print(f"saved {out_path}")
    print(f"  config_version={cfg.version}  collection={cfg.retrieval.collection}  embed={cfg.retrieval.embed_model}")
    print(f"  faithfulness={a.faithfulness_rate:.2f}  answer_success={a.answer_success_rate:.2f}  "
          f"false_refusal={a.false_refusal_rate:.2f}  avg_latency={a.avg_total_latency_ms:.0f}ms")
    if ov:
        print(f"  retrieval: mrr={ov['mrr']:.2f}  recall@10={ov['recall']['10']:.2f}  ndcg@10={ov['ndcg']['10']:.2f}")
    if cit:
        fmt = lambda x: f"{x:.2f}" if x is not None else "—"
        print(f"  citations: validity={fmt(cit['validity'])}  precision={fmt(cit['precision'])}  "
              f"recall={fmt(cit['recall'])}  (cited {cit['n_cited']}/{cit['n']})")


if __name__ == "__main__":
    main()