"""
FastAPI server for ClinicalTrialEnv.
Exposes the OpenEnv HTTP interface:
  POST /reset   → Observation
  POST /step    → {observation, reward, done, info}
  GET  /state   → state dict
  GET  /health  → {"status": "ok"}
  GET  /tasks   → list of available tasks with descriptions
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional

from env.environment import ClinicalTrialEnv
from env.models import Action, DeviationReport

app = FastAPI(
    title="ClinicalTrialEnv",
    description="OpenEnv-compliant environment for clinical trial protocol deviation detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment registry (one env per task_id per session)
# ---------------------------------------------------------------------------
_envs: dict[str, ClinicalTrialEnv] = {}


def _get_env(task_id: str) -> ClinicalTrialEnv:
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"No active env for task '{task_id}'. Call /reset first.")
    return _envs[task_id]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: int = 42


class StepRequest(BaseModel):
    task_id: str = "task1"
    action: dict   # will be parsed into Action


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "ClinicalTrialEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "task1",
                "name": "Single Deviation Detection",
                "difficulty": "easy",
                "description": (
                    "3 patients, 1 protocol deviation (missed visit). "
                    "A hint is provided on the first observation."
                ),
                "n_patients": 3,
                "n_deviations": 1,
            },
            {
                "task_id": "task2",
                "name": "Multi-Patient Audit",
                "difficulty": "medium",
                "description": (
                    "5 patients, 4 deviations across different categories "
                    "(wrong dose, missing lab, late SAE, out-of-window visit). No hints."
                ),
                "n_patients": 5,
                "n_deviations": 4,
            },
            {
                "task_id": "task3",
                "name": "Complex Trial Audit with Decoy",
                "difficulty": "hard",
                "description": (
                    "8 patients, 7 deviations including critical eligibility violations, "
                    "overdose, and cascading effects. One patient has borderline labs "
                    "designed to trigger false positives. No hints."
                ),
                "n_patients": 8,
                "n_deviations": 7,
            },
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest):
    env = ClinicalTrialEnv(task_id=request.task_id, seed=request.seed)
    obs = env.reset()
    _envs[request.task_id] = env
    return obs.model_dump()


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    env = _get_env(request.task_id)
    try:
        action = Action(**request.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action format: {e}")
    obs, reward, done, info = env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def state(task_id: str = Query("task1")):
    env = _get_env(task_id)
    return env.state()


@app.post("/close")
def close(task_id: str = Query("task1")):
    if task_id in _envs:
        _envs[task_id].close()
        del _envs[task_id]
    return {"closed": task_id}
