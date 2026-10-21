"""
Config object for a reproducible /query run.

One pydantic model, grouped by stage, loaded from a YAML file. It replaces the
scattered constants (COLLECTION, ANSWER_MODEL, JUDGE_MODEL) and the defaults
baked into run_rag's signature.

`version` is a short SHA-256 hash of the resolved config — it goes into every
QueryRecord as `config_version`, so any record can be traced back to the exact
settings that produced it. (Hydra's override/multirun machinery is deferred;
plain pydantic + YAML is all this slice needs.)
"""

from __future__ import annotations

import hashlib
import json

import yaml
from pydantic import BaseModel


class RetrievalConfig(BaseModel):
    collection: str
    embed_model: str = "text-embedding-3-small"
    embed_kind: str = "openai"          # "openai" or "local"
    dense_top_k: int = 10


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    top_n: int = 3


class AnswerConfig(BaseModel):
    model: str = "gpt-4.1-mini"


class JudgeConfig(BaseModel):
    model: str = "gpt-4.1-mini"


class RunConfig(BaseModel):
    seed: int = 42
    retrieval: RetrievalConfig
    rerank: RerankConfig = RerankConfig()
    answer: AnswerConfig = AnswerConfig()
    judge: JudgeConfig = JudgeConfig()

    @property
    def version(self) -> str:
        """Short, stable hash of the resolved config. Same settings -> same
        hash, so it identifies a configuration across runs."""
        blob = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)