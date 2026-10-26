import os
import json
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Bump when the answer prompt changes, so results trace to an answer-prompt version.
# v1 — labelled context positionally ([chunk_0]) and cited positions.
# v2 — labels context with stable corpus chunk ids ([k8s_001]) and cites those ids,
#      so citations can be scored against the dataset's relevant_chunk_ids.
ANSWER_PROMPT_VERSION = "v2"

# Original prompt — kept as the fallback when no chunk ids are supplied, so any
# other caller behaves exactly as before.
SYSTEM = """You answer ONLY using the provided CONTEXT.
If and only if the answer is not explicitly supported by the context,
respond exactly with:

"I don't know based on the provided context."

Do not use prior knowledge.
Do not infer beyond the context.
Keep answers concise.
Cite sources as [chunk_i].
"""

# v2 prompt — identical rules, except citations use the stable source id shown on
# each chunk. The refusal sentence is byte-for-byte the same so refusal detection
# and the correct-refusal metric are unaffected.
SYSTEM_CITED = """You answer ONLY using the provided CONTEXT.
If and only if the answer is not explicitly supported by the context,
respond exactly with:

"I don't know based on the provided context."

Do not use prior knowledge.
Do not infer beyond the context.
Keep answers concise.

Each CONTEXT chunk is labelled with a source id in square brackets, e.g. [k8s_001].
When you use a fact from a chunk, cite that chunk's id inline in square brackets,
exactly as shown (e.g. ...scale the deployment [k8s_001]). Cite every chunk you rely
on, and never cite an id that is not shown in the CONTEXT.
"""


def generate_answer(
    question: str,
    context_chunks: List[str],
    model: str = "gpt-4.1-mini",
    context_ids: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # When ids are supplied, label each chunk with its stable source id and ask
    # for id-based citations. Otherwise fall back to the original positional form.
    if context_ids:
        system = SYSTEM_CITED
        context = "\n\n".join(
            f"[{cid}] {c}" for cid, c in zip(context_ids, context_chunks)
        )
    else:
        system = SYSTEM
        context = "\n\n".join(f"[chunk_{i}] {c}" for i, c in enumerate(context_chunks))

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"question": question, "context": context})},
        ],
    )

    answer = resp.choices[0].message.content
    usage = resp.usage.model_dump() if resp.usage else {}
    return answer, usage