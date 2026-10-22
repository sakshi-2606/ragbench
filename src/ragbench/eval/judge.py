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