"""
Pairwise judging with a position-bias control.

A pairwise judge is shown two answers (A and B) to the same question and picks
which is more faithful to the context. Naive pairwise judging is corrupted by
position bias — the judge tends to favor a slot (usually A) regardless of
content. The control: judge every pair in BOTH orders and only trust a result
when it's consistent across the swap. A verdict that flips when you swap the
order is position bias, measured.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PAIRWISE_PROMPT_VERSION = "v1"

PAIRWISE_SYSTEM = """You compare two answers (A and B) to the same question and
judge ONLY which is more FAITHFUL to the provided CONTEXT — i.e. which has fewer
claims that are not supported by the context.

Ignore length, style, and fluency: a longer or more elaborate answer is NOT
better. Judge only grounding in the context. If the two are equally faithful,
answer "tie".

Return ONLY valid JSON: {"winner": "A" | "B" | "tie", "reason": "<=20 words"}.
"""


def pairwise_judge(question, context_chunks, answer_a, answer_b, model="gpt-4.1-mini"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = "\n\n".join(f"[chunk_{i}] {c}" for i, c in enumerate(context_chunks))
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PAIRWISE_SYSTEM},
            {"role": "user", "content": json.dumps(
                {"question": question, "context": context, "answer_A": answer_a, "answer_B": answer_b})},
        ],
    )
    return json.loads(resp.choices[0].message.content).get("winner", "tie")


def compare_both_orders(question, context_chunks, left, right, left_name, right_name, model="gpt-4.1-mini"):
    """Judge (left vs right) in BOTH slot orders, mapping each verdict back to the
    content name. Returns the two content-verdicts, whether they agree, and the
    position-consistent winner ('inconsistent' when the judge flipped with order)."""
    # order 1: A = left, B = right
    w1 = pairwise_judge(question, context_chunks, left, right, model)
    v1 = {"A": left_name, "B": right_name, "tie": "tie"}[w1]
    # order 2: A = right, B = left  (swap the slots)
    w2 = pairwise_judge(question, context_chunks, right, left, model)
    v2 = {"A": right_name, "B": left_name, "tie": "tie"}[w2]

    consistent = (v1 == v2)
    return {"order1": v1, "order2": v2, "consistent": consistent,
            "winner": v1 if consistent else "inconsistent"}