"""
Typed models for the ClinicalTrialEnv environment.
All models use Pydantic v2 for validation and serialization.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class PatientRecord(BaseModel):
    """A single patient's data as seen in one observation."""
    patient_id: str
    visit_schedule: list[dict[str, Any]]   # [{visit: str, expected_day: int, actual_day: Optional[int], ...}]
    adverse_events: list[dict[str, Any]]   # [{event: str, grade: int, reported_day: int, sae_flag: bool}]
    lab_results: list[dict[str, Any]]      # [{test: str, value: float, unit: str, day: int, flag: str}]
    dosing_records: list[dict[str, Any]]   # [{day: int, dose_mg: float, administered: bool}]
    inclusion_criteria: dict[str, Any]     # {criterion_id: {description: str, met: bool}}
    exclusion_criteria: dict[str, Any]     # {criterion_id: {description: str, violated: bool}}


class ProtocolRules(BaseModel):
    """The trial protocol rules the agent must check against."""
    trial_id: str
    visit_window_days: int = Field(description="Acceptable deviation from scheduled visit day")
    dose_per_visit_mg: float
    max_dose_mg_per_day: float
    required_labs: list[str]               # lab tests required at each visit
    sae_reporting_window_hours: int = 24   # SAE must be reported within N hours
    washout_period_days: int = 7           # days between dose and eligibility screen


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    step: int
    protocol: ProtocolRules
    patients: list[PatientRecord]
    action_history: list[str] = Field(default_factory=list)
    hint: Optional[str] = None             # Non-null only in easy task


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class DeviationReport(BaseModel):
    """One reported deviation."""
    patient_id: str
    deviation_type: str   # e.g. "missed_visit", "wrong_dose", "late_sae_report", etc.
    details: str          # free-text explanation
    severity: str         # "minor" | "major" | "critical"


class Action(BaseModel):
    """
    The agent submits a list of deviations it detected.
    An empty list signals 'no deviations found'.
    Calling submit=True finalises the episode.
    """
    reports: list[DeviationReport] = Field(default_factory=list)
    submit: bool = False
    reasoning: Optional[str] = None   # chain-of-thought (not graded, logged only)


# ---------------------------------------------------------------------------
# Reward / Info
# ---------------------------------------------------------------------------

class StepInfo(BaseModel):
    """Returned in the info dict from step()."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    penalty: float = 0.0
    message: str = ""
