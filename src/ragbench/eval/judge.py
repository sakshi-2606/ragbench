import os
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Bump this when the judge prompt changes, so results trace to a judge version.
JUDGE_PROMPT_VERSION = "v2"

# v2 — tightened after calibration showed the judge accepted plausible-but-
# unsupported additions (benefit claims, extra real-world examples). It now
# enumerates claims and checks each against the context, and is told explicitly
# to ignore outside knowledge.
JUDGE_SYSTEM = """You are a strict RAG faithfulness judge. Judge ONLY whether every
claim in the ANSWER is directly supported by the provided CONTEXT — NOT whether the
answer is true in the real world.

Procedure:
1. Break the ANSWER into its individual factual claims.
2. For EACH claim, check whether it is explicitly supported by the CONTEXT.
3. A claim that is plausible, generically true, or something you know from outside
   knowledge but that is NOT stated in the CONTEXT makes the answer UNFAITHFUL. This
   includes added benefits (e.g. "ensures high availability"), extra examples (e.g.
   "or a GCS bucket"), and any specific detail not present in the CONTEXT.
4. Ignore your own prior knowledge entirely. The only source of truth is the CONTEXT.

If ALL claims are supported, faithful=true. If ANY claim is unsupported, faithful=false.
If unclear, faithful=false.

Return ONLY valid JSON:
faithful (boolean),
confidence (0-1),
rationale (string, <= 25 words; name the unsupported claim if any).
"""


def judge_faithfulness(
    question: str,
    answer: str,
    context_chunks: List[str],
    model: str = "gpt-4.1-mini",
):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = "\n\n".join([f"[chunk_{i}] {c}" for i, c in enumerate(context_chunks)])

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps({
                    "question": question,
                    "context": context,
                    "answer": answer
                })
            },
        ],
    )

    verdict = json.loads(resp.choices[0].message.content)
    usage = resp.usage.model_dump() if resp.usage else {}
    return verdict, usage


# --------------------------------------------------------------------------- #
# Answer-quality judge (WS1): correctness vs gold, relevance vs question.
#
# Two SEPARATE LLM calls on purpose. The relevance judge is never given the
# gold answer, so "relevant" cannot quietly collapse into "correct" — the
# blind-to-gold rule is enforced at the API boundary, not just requested in
# the prompt. Both are scored on small discrete buckets (not a free-floating
# 0-1 float), which LLM judges report far more reliably.
# --------------------------------------------------------------------------- #
ANSWER_QUALITY_PROMPT_VERSION = "v2"

# v2 — smoke test showed the relevance judge marking a wrong-but-responsive answer
# "irrelevant" (correctness leaking into relevance). The relevance prompt now states
# that relevance is about RESPONSIVENESS, and a direct-but-wrong answer is relevant.

CORRECTNESS_SYSTEM = """You are a strict answer-correctness judge. You are given a
QUESTION, a GOLD reference answer, and a candidate ANSWER. Decide how well the ANSWER
agrees with the GOLD on the key facts the question asks for.

The GOLD is the only source of truth. Do NOT use outside knowledge, and do NOT reward an
answer merely for being plausible — reward it only for agreeing with the GOLD.

Choose exactly one label:
- "correct": agrees with the GOLD on all the key facts. Different wording, or extra
  detail that does not contradict the GOLD, is still correct.
- "partially_correct": gets the main point but misses a required part that the GOLD
  includes, OR is right about one part while contradicting the GOLD on another. (If the
  GOLD states two facts and the ANSWER gives only one, that is partially_correct.)
- "incorrect": disagrees with the GOLD, or fails to give the answer the GOLD gives.

Return ONLY valid JSON:
correctness (one of "correct", "partially_correct", "incorrect"),
confidence (0-1),
rationale (string, <= 25 words).
"""

RELEVANCE_SYSTEM = """You are an answer-relevance judge. You are given a QUESTION and a
candidate ANSWER. Decide whether the ANSWER engages with and responds to what the
QUESTION asks.

This is about RESPONSIVENESS, not truth. You are NOT given a reference answer and you
must NOT judge whether the answer is correct. A direct response to the question counts
as relevant EVEN IF it is factually wrong.

Choose exactly one label:
- "relevant": gives a direct response to the question asked. For a "how do I X?"
  question this means it tells you how to X, or asserts X cannot be done and what to do
  instead. A wrong but direct answer is still relevant.
- "partial": discusses the right topic but does not actually respond to the question
  (for example, gives background about the topic instead of answering what was asked).
- "irrelevant": about a different topic, or answers a different question.

Return ONLY valid JSON:
relevance (one of "relevant", "partial", "irrelevant"),
confidence (0-1),
rationale (string, <= 25 words).
"""


def _judge_call(system: str, payload: dict, model: str) -> Tuple[dict, dict]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )
    verdict = json.loads(resp.choices[0].message.content)
    usage = resp.usage.model_dump() if resp.usage else {}
    return verdict, usage


def judge_answer_quality(
    question: str,
    answer: str,
    gold_answer: str,
    model: str = "gpt-4.1-mini",
):
    """Two independent judgments on an ANSWERABLE item:

      correctness (answer vs gold)              -> correct / partially_correct / incorrect
      relevance   (answer vs question, no gold) -> relevant / partial / irrelevant

    Returns (verdict, usage). `usage` sums the token counts of the two calls.
    """
    c_verdict, c_usage = _judge_call(
        CORRECTNESS_SYSTEM,
        {"question": question, "gold": gold_answer, "answer": answer},
        model,
    )
    r_verdict, r_usage = _judge_call(
        RELEVANCE_SYSTEM,
        {"question": question, "answer": answer},
        model,
    )

    def _sum(k):
        return (c_usage.get(k, 0) or 0) + (r_usage.get(k, 0) or 0)

    verdict = {
        "correctness": str(c_verdict.get("correctness", "")).lower().strip(),
        "correctness_confidence": c_verdict.get("confidence"),
        "correctness_rationale": c_verdict.get("rationale"),
        "relevance": str(r_verdict.get("relevance", "")).lower().strip(),
        "relevance_confidence": r_verdict.get("confidence"),
        "relevance_rationale": r_verdict.get("rationale"),
    }
    usage = {
        "prompt_tokens": _sum("prompt_tokens"),
        "completion_tokens": _sum("completion_tokens"),
        "total_tokens": _sum("total_tokens"),
    }
    return verdict, usage