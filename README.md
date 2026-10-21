# RAGBench
An experimental framework for benchmarking Retrieval-Augmented Generation (RAG) systems across retrieval quality, reranking impact, faithfulness, hallucination, safety gating, latency, and cost.

## Objective
RAGBench isolates where RAG systems fail:
 - Is retrieval missing relevant context?
 - Is reranking helping or adding unnecessary latency?
 - Do reasoning models hallucinate more?
 - How do threshold gates affect coverage vs safety?

## Architecture
Dense Retrieval (Qdrant + embeddings)
 → Optional Cross-Encoder Reranking (BGE reranker)
 → LLM Answer Generation
 → LLM Judge (Faithfulness + Safety Evaluation)

## Key Experiments
1. Dense vs Dense+Rerank Ablation
    Measured recall@k, latency, and hallucination under:
      - text-embedding-3-large (OpenAI, 3072-dim)
      - bge-small-en-v1.5 (local)

    Finding:
      Reranking added ~2–3x latency without recall improvement in small corpora, indicating a dense-retrieval bottleneck.

2. Threshold Gating (Safety vs Utility)
    Explored tau_dense and tau_rerank thresholds.
    Observed:
      - Higher thresholds reduce hallucination
      - But increase false refusal rate
      - Borderline queries get gated → lower coverage

3. LLM Comparison
    Benchmarked:
      - GPT-4.1-mini
      - GPT-4o
      - GPT-5.1
    
    Finding:
      - Faithfulness remained high
      - Reasoning models showed higher false refusal under strict gating

## Metrics
 - Recall@k
 - Faithfulness rate
 - Hallucination rate
 - Answer success rate
 - False refusal rate
 - Avg retrieval latency
 - Avg rerank latency
 - Cost per query

## Key Insight
RAG systems are often recall-limited, not ranking-limited.
If dense retrieval fails to surface the correct chunk, reranking cannot fix the problem — it only increases latency.

## Future Work
 - Large-scale corpus (1000+ chunks) to stress retrieval models
 - Hybrid retrieval (BM25 + dense)
 - Adaptive threshold gating
 - Multi-judge agreement for hallucination robustness
 - Context window scaling experiments

## Experimental Findings

    Empirical Observations
      - Reranking does not improve recall if dense retrieval fails to retrieve relevant chunks within top-k.
      - Reranking significantly increases latency (400–600ms per query).
      - Local embeddings provide large latency gains but may degrade recall at scale.
      - Strict gating thresholds reduce hallucination but increase false refusal.
      - Reasoning-heavy LLMs may over-refuse when context confidence is low.
