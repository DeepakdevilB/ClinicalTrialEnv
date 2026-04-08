"""
ClinicalTrialEnv — Main environment class.

Implements the OpenEnv interface:
  reset()  → Observation
  step()   → (Observation, reward, done, info)
  state()  → dict

Three tasks:
  task1: easy   — 3 patients, 1 deviation, with hint
  task2: medium — 5 patients, 4 deviations, no hint
  task3: hard   — 8 patients, 7 deviations + decoy, no hint

Agent action: submit a list of DeviationReport objects.
Reward: incremental weighted F1 per step; -penalty for false positives.
Max steps: 5 (agent can refine its reports iteratively).
"""

from typing import Any
from env.models import Action, Observation, ProtocolRules
from env.grader import grade_action, final_score
from env.data_generator import (
    build_task1_scenario,
    build_task2_scenario,
    build_task3_scenario,
    _default_protocol,
)


TASK_HINTS = {
    "task1": (
        "Hint: one patient missed a scheduled visit. Check the visit_schedule "
        "for completed=False entries."
    ),
    "task2": None,
    "task3": None,
}

MAX_STEPS = 5


class ClinicalTrialEnv:
    """
    OpenEnv-compliant environment for clinical trial protocol deviation detection.
    """

    def __init__(self, task_id: str = "task1", seed: int = 42):
        if task_id not in ("task1", "task2", "task3"):
            raise ValueError(f"Unknown task_id: {task_id}. Choose from task1, task2, task3.")
        self.task_id = task_id
        self.seed = seed
        self._patients = None
        self._ground_truth = None
        self._protocol = None
        self._step_count = 0
        self._done = False
        self._found_keys: set[str] = set()
        self._action_history: list[str] = []
        self._current_reward: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._protocol = _default_protocol()
        self._step_count = 0
        self._done = False
        self._found_keys = set()
        self._action_history = []
        self._current_reward = 0.0

        if self.task_id == "task1":
            self._patients, self._ground_truth = build_task1_scenario(self.seed)
        elif self.task_id == "task2":
            self._patients, self._ground_truth = build_task2_scenario(self.seed)
        else:
            self._patients, self._ground_truth = build_task3_scenario(self.seed)

        return self._build_observation()

    def step(self, action: Action | dict) -> tuple[Observation, float, bool, dict]:
        """
        Process the agent's action and return (observation, reward, done, info).

        action: Action model or dict (auto-converted)
        reward: float in [-1, 1] — incremental this step
        done: True if agent submitted or max steps reached
        info: StepInfo serialized to dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        if isinstance(action, dict):
            action = Action(**action)

        self._step_count += 1

        reward, info, self._found_keys = grade_action(
            action, self._ground_truth, self._found_keys
        )
        self._current_reward = reward

        # Summarise action for history
        n_reports = len(action.reports)
        summary = (
            f"Step {self._step_count}: submitted {n_reports} report(s) — "
            f"F1={info.f1:.2f}, TP={info.true_positives}, FP={info.false_positives}"
        )
        self._action_history.append(summary)

        # Episode ends when agent submits or max steps reached
        if action.submit or self._step_count >= MAX_STEPS:
            self._done = True
            # Final score overrides last step reward
            reward = final_score(self._ground_truth, self._found_keys)
            self._current_reward = reward

        obs = self._build_observation()
        return obs, reward, self._done, info.model_dump()

    def state(self) -> dict[str, Any]:
        """Return the full current environment state."""
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "step": self._step_count,
            "done": self._done,
            "found_deviation_keys": list(self._found_keys),
            "ground_truth_count": len(self._ground_truth) if self._ground_truth else 0,
            "patient_count": len(self._patients) if self._patients else 0,
            "last_reward": self._current_reward,
            "action_history": self._action_history,
        }

    def close(self):
        """Cleanup (no-op for this environment)."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        hint = TASK_HINTS.get(self.task_id) if self._step_count == 0 else None
        return Observation(
            task_id=self.task_id,
            step=self._step_count,
            protocol=self._protocol,
            patients=self._patients,
            action_history=self._action_history,
            hint=hint,
        )
