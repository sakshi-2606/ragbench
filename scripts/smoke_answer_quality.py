"""Batch 1 (WS1 answer-quality) — eyeball the correctness/relevance buckets on a
few hand-built cases BEFORE wiring the judge into the eval.

Run from the repo root:
    python scripts/smoke_answer_quality.py

Needs OPENAI_API_KEY in the environment (same key the faithfulness judge uses).
The four cases are designed to hit each interesting corner once.
"""

from ragbench.eval.judge import judge_answer_quality, ANSWER_QUALITY_PROMPT_VERSION

CASES = [
    # (what we expect to see, question, gold, answer)
    (
        "correct + relevant",
        "How do I roll back a Helm release?",
        "Use `helm rollback <release> <revision>` to revert to a previous revision.",
        "Run `helm rollback myapp 2` to roll the release back to revision 2.",
    ),
    (
        "incorrect + relevant  (faithful-looking but wrong)",
        "How do I roll back a Helm release?",
        "Use `helm rollback <release> <revision>` to revert to a previous revision.",
        "Helm releases cannot be rolled back; you must uninstall and reinstall the chart.",
    ),
    (
        "partial relevance  (on-topic but evasive)",
        "How do I roll back a Helm release?",
        "Use `helm rollback <release> <revision>` to revert to a previous revision.",
        "Helm keeps a revision history for every release, stored as secrets in the cluster.",
    ),
    (
        "partially_correct  (multi-hop: one of two facts)",
        "Which chart version does the prod release use, and who last upgraded it?",
        "Prod runs chart version 1.4.2, last upgraded by the platform team.",
        "The prod release is on chart version 1.4.2.",
    ),
]

print(f"answer-quality judge prompt version: {ANSWER_QUALITY_PROMPT_VERSION}\n")
for expected, q, gold, ans in CASES:
    v, usage = judge_answer_quality(q, ans, gold)
    print(f"[expect: {expected}]")
    print(f"  correctness = {v['correctness']:18s} ({v['correctness_rationale']})")
    print(f"  relevance   = {v['relevance']:18s} ({v['relevance_rationale']})")
    print()
