"""
Thin FastAPI wrapper over the pipeline.

  GET  /health  -> liveness + which config is loaded
  POST /query   -> run one question, return the full QueryRecord
  POST /eval    -> run a dataset, return the full EvalReport (and persist files)

All real work lives in run_query() / run_eval(). Config loaded once at startup.

Run from the repo root:
    uvicorn ragbench.api.service:app --port 8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from ragbench.utils.config import RunConfig
from ragbench.utils.run_schema import QueryRecord
from ragbench.utils.eval_schema import EvalReport
from ragbench.pipeline.run_query import run_query
from ragbench.eval.run_eval import run_eval


CONFIG_PATH = os.getenv("RAGBENCH_CONFIG", "configs/query.yaml")

app = FastAPI(title="RAGBench Query Service", version="0.4.0")
config = RunConfig.from_yaml(CONFIG_PATH)


class QueryRequest(BaseModel):
    question: str


class EvalRequest(BaseModel):
    dataset_path: str = "benchmarks/evalset_v2.jsonl"   # frozen, tagged, relevance-labeled set
    persist: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "config_version": config.version}


@app.post("/query", response_model=QueryRecord)
def query(req: QueryRequest) -> QueryRecord:
    return run_query(req.question, config)


@app.post("/eval", response_model=EvalReport)
def eval_dataset(req: EvalRequest) -> EvalReport:
    """Run the whole dataset and return the full eval report.
    Synchronous — the request stays open until the run finishes (~2-3 min)."""
    persist_dir = "results" if req.persist else None
    return run_eval(req.dataset_path, config, persist_dir=persist_dir)