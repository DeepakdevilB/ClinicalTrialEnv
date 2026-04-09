"""
Reward grader for ClinicalTrialEnv.

Scoring philosophy:
- Each correctly identified deviation earns partial credit.
- Each false positive (hallucinated deviation) incurs a penalty.
- Critical deviations (SAE, eligibility) are weighted 2x.
- Final score = weighted F1 in [0.0, 1.0].
- Step rewards are incremental: each new true positive in a step earns +delta.
"""

from env.models import Action, StepInfo

SEVERITY_WEIGHTS = {
    "critical": 2.0,
    "major": 1.5,
    "minor": 1.0,
}

FALSE_POSITIVE_PENALTY = 0.15   # per false positive report

# Validator requires scores strictly inside (0, 1) — never 0.0 or 1.0
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp(value: float) -> float:
    """Clamp a score to the open interval (0.001, 0.999)."""
    return round(max(_SCORE_MIN, min(_SCORE_MAX, value)), 4)


def grade_action(
    action: Action,
    ground_truth: list[dict],
    previous_found: set[str],   # deviation keys already found in prior steps
) -> tuple[float, StepInfo, set[str]]:
    """
    Compare agent's reports against ground truth.

    Returns:
        step_reward: float in [-1, 1] for this step only
        info: StepInfo with precision/recall/F1 and explanation
        updated_found: set of deviation keys found so far
    """
    # Build a canonical key for each ground truth deviation
    gt_keys = {
        _deviation_key(d): d for d in ground_truth
    }

    # Track which ground truth deviations were matched
    matched_gt_keys: set[str] = set(previous_found)  # cumulative
    new_tp = 0
    new_fp = 0

    for report in action.reports:
        key = _deviation_key_from_report(report)
        if key in gt_keys and key not in matched_gt_keys:
            matched_gt_keys.add(key)
            new_tp += 1
        elif key not in gt_keys:
            new_fp += 1

    # Calculate weighted metrics across entire episode so far
    tp_weighted = sum(
        SEVERITY_WEIGHTS.get(gt_keys[k]["severity"], 1.0)
        for k in matched_gt_keys if k in gt_keys
    )
    total_gt_weight = sum(
        SEVERITY_WEIGHTS.get(d["severity"], 1.0) for d in ground_truth
    )
    total_reported = len(action.reports)
    fp_penalty = new_fp * FALSE_POSITIVE_PENALTY

    precision = tp_weighted / max(tp_weighted + new_fp, 1e-9)
    recall = tp_weighted / max(total_gt_weight, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    f1 = _clamp(f1 - fp_penalty)

    # Step-level reward: incremental gain from new true positives minus penalty
    new_tp_weight = sum(
        SEVERITY_WEIGHTS.get(gt_keys[k]["severity"], 1.0)
        for k in matched_gt_keys
        if k not in previous_found and k in gt_keys
    )
    step_reward = (new_tp_weight / max(total_gt_weight, 1e-9)) - fp_penalty
    step_reward = _clamp(step_reward)

    info = StepInfo(
        true_positives=len(matched_gt_keys),
        false_positives=new_fp,
        false_negatives=len(gt_keys) - len(matched_gt_keys),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        penalty=round(fp_penalty, 4),
        message=_build_message(new_tp, new_fp, len(gt_keys) - len(matched_gt_keys)),
    )

    return round(step_reward, 4), info, matched_gt_keys


def final_score(ground_truth: list[dict], found_keys: set[str]) -> float:
    """
    Compute final episode score (0.0 – 1.0) based on deviations found.
    """
    gt_keys = {_deviation_key(d) for d in ground_truth}
    total_weight = sum(SEVERITY_WEIGHTS.get(d["severity"], 1.0) for d in ground_truth)
    found_weight = sum(
        SEVERITY_WEIGHTS.get(d["severity"], 1.0)
        for d in ground_truth
        if _deviation_key(d) in found_keys
    )
    return _clamp(found_weight / max(total_weight, 1e-9))


def _deviation_key(d: dict) -> str:
    return f"{d['patient_id']}::{d['deviation_type']}"


def _deviation_key_from_report(report) -> str:
    return f"{report.patient_id}::{report.deviation_type}"


def _build_message(new_tp: int, new_fp: int, remaining: int) -> str:
    parts = []
    if new_tp:
        parts.append(f"+{new_tp} correct deviation(s) identified")
    if new_fp:
        parts.append(f"{new_fp} false positive(s) — penalty applied")
    if remaining:
        parts.append(f"{remaining} deviation(s) still undetected")
    return "; ".join(parts) if parts else "No change this step"
