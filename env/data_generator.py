"""
Synthetic clinical trial data generator.
All randomness is seeded for reproducibility.
Generates PatientRecord objects with known protocol deviations injected.
"""

import random
from typing import Optional
from env.models import PatientRecord, ProtocolRules


# ---------------------------------------------------------------------------
# Deviation type catalogue
# ---------------------------------------------------------------------------

DEVIATION_TYPES = {
    "missed_visit": "Patient did not attend a scheduled protocol visit",
    "visit_out_of_window": "Patient visit occurred outside the allowed visit window",
    "wrong_dose": "Patient received incorrect dose per protocol",
    "missing_lab": "Required lab test not performed at scheduled visit",
    "late_sae_report": "Serious adverse event not reported within required window",
    "inclusion_violation": "Patient did not meet inclusion criteria at screening",
    "exclusion_violation": "Patient violated an exclusion criterion at enrollment",
    "overdose": "Patient received dose exceeding maximum allowed daily dose",
    "washout_violation": "Insufficient washout period before study drug administration",
    "undocumented_ae": "Adverse event grade change not documented in source records",
}

LAB_TESTS = ["CBC", "BMP", "LFT", "COAG", "URINE_DIPSTICK"]
VISIT_NAMES = ["Screening", "Baseline", "Week 2", "Week 4", "Week 8", "Week 12", "End of Study"]
AE_NAMES = ["Nausea", "Fatigue", "Headache", "Dizziness", "Rash", "Dyspnea", "Palpitations"]


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _build_clean_patient(patient_id: str, protocol: ProtocolRules, rng: random.Random) -> PatientRecord:
    """Generate a patient with zero deviations."""
    visit_schedule = []
    for i, name in enumerate(VISIT_NAMES[:5]):
        expected_day = i * 14
        # Actual day within window
        jitter = rng.randint(-protocol.visit_window_days, protocol.visit_window_days)
        visit_schedule.append({
            "visit": name,
            "expected_day": expected_day,
            "actual_day": expected_day + jitter,
            "completed": True,
        })

    adverse_events = []
    if rng.random() < 0.4:
        grade = rng.randint(1, 2)
        event_day = rng.randint(5, 40)
        adverse_events.append({
            "event": rng.choice(AE_NAMES),
            "grade": grade,
            "reported_day": event_day,
            "reported_hours_after_onset": rng.randint(1, 20),  # within 24h for grade<3
            "sae_flag": False,
        })

    lab_results = []
    for visit_idx, v in enumerate(visit_schedule):
        for test in protocol.required_labs:
            lab_results.append({
                "test": test,
                "value": round(rng.uniform(0.8, 1.2) * 100, 1),
                "unit": "U/L",
                "day": v["actual_day"],
                "visit": v["visit"],
                "flag": "NORMAL",
            })

    dosing_records = []
    for v in visit_schedule:
        dosing_records.append({
            "day": v["actual_day"],
            "dose_mg": protocol.dose_per_visit_mg,
            "administered": True,
        })

    inclusion_criteria = {
        "INC001": {"description": "Age 18-75", "met": True},
        "INC002": {"description": "ECOG performance status 0-2", "met": True},
        "INC003": {"description": "Adequate organ function", "met": True},
    }
    exclusion_criteria = {
        "EXC001": {"description": "Prior treatment with study drug class", "violated": False},
        "EXC002": {"description": "Active systemic infection", "violated": False},
        "EXC003": {"description": "Pregnancy or breastfeeding", "violated": False},
    }

    return PatientRecord(
        patient_id=patient_id,
        visit_schedule=visit_schedule,
        adverse_events=adverse_events,
        lab_results=lab_results,
        dosing_records=dosing_records,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
    )


# ---------------------------------------------------------------------------
# Deviation injectors (each returns the modified patient + a ground-truth tag)
# ---------------------------------------------------------------------------

def inject_missed_visit(patient: PatientRecord, visit_index: int = 2) -> dict:
    """Remove actual_day and mark a visit as not completed."""
    v = patient.visit_schedule[visit_index]
    v["completed"] = False
    v["actual_day"] = None

    # Also remove labs for that visit
    visit_name = v["visit"]
    patient.lab_results = [
        lr for lr in patient.lab_results if lr.get("visit") != visit_name
    ]

    return {
        "patient_id": patient.patient_id,
        "deviation_type": "missed_visit",
        "details": f"Visit '{v['visit']}' (expected day {v['expected_day']}) was not completed.",
        "severity": "major",
    }


def inject_visit_out_of_window(patient: PatientRecord, protocol: ProtocolRules, visit_index: int = 3) -> dict:
    """Push a visit far outside the allowed window."""
    v = patient.visit_schedule[visit_index]
    shift = protocol.visit_window_days + rng_shift(8, 15)
    v["actual_day"] = v["expected_day"] + shift

    # Update labs day to match
    for lr in patient.lab_results:
        if lr.get("visit") == v["visit"]:
            lr["day"] = v["actual_day"]

    return {
        "patient_id": patient.patient_id,
        "deviation_type": "visit_out_of_window",
        "details": (
            f"Visit '{v['visit']}' occurred on day {v['actual_day']}, "
            f"{shift} days after expected day {v['expected_day']}. "
            f"Allowed window: ±{protocol.visit_window_days} days."
        ),
        "severity": "minor",
    }


def inject_wrong_dose(patient: PatientRecord, protocol: ProtocolRules, visit_index: int = 1) -> dict:
    """Record an incorrect dose."""
    d = patient.dosing_records[visit_index]
    wrong_dose = protocol.dose_per_visit_mg * 1.5
    d["dose_mg"] = wrong_dose
    return {
        "patient_id": patient.patient_id,
        "deviation_type": "wrong_dose",
        "details": (
            f"Dose administered on day {d['day']} was {wrong_dose} mg; "
            f"protocol specifies {protocol.dose_per_visit_mg} mg."
        ),
        "severity": "major",
    }


def inject_missing_lab(patient: PatientRecord, protocol: ProtocolRules, visit_name: str = "Week 4") -> dict:
    """Remove one required lab test from a visit."""
    missing_test = protocol.required_labs[0]
    before = len(patient.lab_results)
    patient.lab_results = [
        lr for lr in patient.lab_results
        if not (lr.get("visit") == visit_name and lr["test"] == missing_test)
    ]
    return {
        "patient_id": patient.patient_id,
        "deviation_type": "missing_lab",
        "details": f"Required lab '{missing_test}' was not collected at visit '{visit_name}'.",
        "severity": "minor",
    }


def inject_late_sae(patient: PatientRecord) -> dict:
    """Add a grade-3 SAE that was reported > 24h after onset."""
    patient.adverse_events.append({
        "event": "Dyspnea",
        "grade": 3,
        "reported_day": 30,
        "reported_hours_after_onset": 72,   # 3x the required window
        "sae_flag": True,
    })
    return {
        "patient_id": patient.patient_id,
        "deviation_type": "late_sae_report",
        "details": (
            "Grade 3 SAE (Dyspnea) was reported 72 hours after onset; "
            "protocol requires reporting within 24 hours."
        ),
        "severity": "critical",
    }


def inject_inclusion_violation(patient: PatientRecord) -> dict:
    """Mark an inclusion criterion as not met."""
    patient.inclusion_criteria["INC002"]["met"] = False
    return {
        "patient_id": patient.patient_id,
        "deviation_type": "inclusion_violation",
        "details": "Patient did not meet INC002: ECOG performance status 0-2 at screening.",
        "severity": "critical",
    }


def inject_exclusion_violation(patient: PatientRecord) -> dict:
    """Mark an exclusion criterion as violated."""
    patient.exclusion_criteria["EXC001"]["violated"] = True
    return {
        "patient_id": patient.patient_id,
        "deviation_type": "exclusion_violation",
        "details": "Patient violated EXC001: had prior treatment with study drug class.",
        "severity": "critical",
    }


def inject_overdose(patient: PatientRecord, protocol: ProtocolRules) -> dict:
    """Inject two doses on the same day, exceeding max daily dose."""
    day = patient.dosing_records[2]["day"]
    patient.dosing_records.append({
        "day": day,
        "dose_mg": protocol.dose_per_visit_mg,
        "administered": True,
    })
    total = protocol.dose_per_visit_mg * 2
    return {
        "patient_id": patient.patient_id,
        "deviation_type": "overdose",
        "details": (
            f"Patient received {total} mg on day {day} (two doses); "
            f"maximum allowed daily dose is {protocol.max_dose_mg_per_day} mg."
        ),
        "severity": "critical",
    }


def rng_shift(lo: int, hi: int) -> int:
    """Small helper to avoid passing rng everywhere for simple shifts."""
    return random.randint(lo, hi)


# ---------------------------------------------------------------------------
# Public API: build scenario for each task
# ---------------------------------------------------------------------------

def build_task1_scenario(seed: int = 42) -> tuple[list[PatientRecord], list[dict]]:
    """
    Easy: 3 patients, exactly 1 obvious deviation (missed visit, with hint).
    """
    rng = random.Random(seed)
    protocol = _default_protocol()
    patients = [_build_clean_patient(f"PT-{101+i}", protocol, rng) for i in range(3)]
    ground_truth = [inject_missed_visit(patients[1], visit_index=2)]
    return patients, ground_truth


def build_task2_scenario(seed: int = 42) -> tuple[list[PatientRecord], list[dict]]:
    """
    Medium: 5 patients, 4 deviations spread across different patients and types.
    No hints.
    """
    rng = random.Random(seed)
    protocol = _default_protocol()
    patients = [_build_clean_patient(f"PT-{201+i}", protocol, rng) for i in range(5)]
    ground_truth = [
        inject_wrong_dose(patients[0], protocol, visit_index=1),
        inject_missing_lab(patients[1], protocol, visit_name="Week 4"),
        inject_late_sae(patients[2]),
        inject_visit_out_of_window(patients[4], protocol, visit_index=3),
    ]
    return patients, ground_truth


def build_task3_scenario(seed: int = 42) -> tuple[list[PatientRecord], list[dict]]:
    """
    Hard: 8 patients, 7 deviations including critical eligibility violations,
    cascading effects, and one patient that looks suspicious but is clean.
    """
    rng = random.Random(seed)
    protocol = _default_protocol()
    patients = [_build_clean_patient(f"PT-{301+i}", protocol, rng) for i in range(8)]
    ground_truth = [
        inject_inclusion_violation(patients[0]),
        inject_exclusion_violation(patients[1]),
        inject_overdose(patients[2], protocol),
        inject_late_sae(patients[3]),
        inject_wrong_dose(patients[4], protocol, visit_index=2),
        inject_missing_lab(patients[5], protocol, visit_name="Week 8"),
        inject_visit_out_of_window(patients[7], protocol, visit_index=4),
        # patients[6] is intentionally clean but has borderline labs (decoy)
    ]
    # Borderline decoy on patient 6 — labs slightly elevated but within range
    for lr in patients[6].lab_results:
        if lr["test"] == "LFT":
            lr["value"] = 118.0   # elevated but not flagged
            lr["flag"] = "HIGH_BORDERLINE"
    return patients, ground_truth


def _default_protocol() -> ProtocolRules:
    return ProtocolRules(
        trial_id="TRIAL-CTE-2024-001",
        visit_window_days=3,
        dose_per_visit_mg=100.0,
        max_dose_mg_per_day=150.0,
        required_labs=["CBC", "LFT", "BMP"],
        sae_reporting_window_hours=24,
        washout_period_days=7,
    )
