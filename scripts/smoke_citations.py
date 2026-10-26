"""Batch 1 (WS1 citations) — confirm the answer now cites STABLE chunk ids before
we wire ids through the pipeline.

Run from the repo root:
    python scripts/smoke_citations.py

Needs OPENAI_API_KEY. Builds a tiny labelled context by hand (no retrieval) and
checks that the answer cites the id of the chunk that actually holds the answer,
and cites nothing that isn't in the context.
"""

import re

from ragbench.generation.answer import generate_answer, ANSWER_PROMPT_VERSION

CITE_RE = re.compile(r"\[([A-Za-z0-9_]+)\]")

ctx_ids = ["k8s_001", "helm_002", "net_003"]
ctx_txt = [
    "A Kubernetes Deployment manages a ReplicaSet; scale it with "
    "`kubectl scale deployment <name> --replicas=N`.",
    "Helm packages apps as charts; `helm rollback <release> <revision>` reverts a release.",
    "A NetworkPolicy restricts pod-to-pod traffic using label selectors.",
]

CASES = [
    ("expect cite k8s_001", "How do I scale a Kubernetes Deployment?"),
    ("expect cite helm_002", "How do I roll back a Helm release?"),
    ("expect refusal, no citations", "What is the default log retention period?"),
]

print(f"answer prompt version: {ANSWER_PROMPT_VERSION}\n")
for expected, q in CASES:
    ans, _ = generate_answer(q, ctx_txt, context_ids=ctx_ids)
    cited = sorted(set(CITE_RE.findall(ans)))
    valid = [c for c in cited if c in ctx_ids]
    invalid = [c for c in cited if c not in ctx_ids]
    print(f"[{expected}]")
    print(f"  Q: {q}")
    print(f"  A: {ans}")
    print(f"  cited={cited}  valid={valid}  invalid(fabricated)={invalid}")
    print()