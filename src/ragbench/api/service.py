"""
Thin FastAPI wrapper over run_query.

POST /query takes a question, runs the full pipeline under the loaded config,
and returns the complete QueryRecord as the HTTP response. All the real work
lives in run_query() — this file only does HTTP plumbing.

Run from the repo root (so the relative config path resolves):
    uvicorn ragbench.api.service:app --port 8000
Interactive docs (schema included): http://localhost:8000/docs
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from ragbench.utils.config import RunConfig
from ragbench.utils.run_schema import QueryRecord
from ragbench.pipeline.run_query import run_query


# Config is loaded ONCE at startup. Point RAGBENCH_CONFIG at a different YAML to
# run the service with other settings; defaults to configs/query.yaml.
CONFIG_PATH = os.getenv("RAGBENCH_CONFIG", "configs/query.yaml")

app = FastAPI(title="RAGBench Query Service", version="0.1.0")
config = RunConfig.from_yaml(CONFIG_PATH)


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    """Liveness check — also reports which config the service is running."""
    return {"status": "ok", "config_version": config.version}


@app.post("/query", response_model=QueryRecord)
def query(req: QueryRequest) -> QueryRecord:
    """Run one query end to end and return the full structured record."""
    return run_query(req.question, config)