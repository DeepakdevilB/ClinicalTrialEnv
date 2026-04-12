"""
Pre-submission validation test suite for ClinicalTrialEnv.

Run with:  python tests/test_env.py
All tests must pass before submitting.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import traceback

PASS = "✅"
FAIL = "❌"
results = []

def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"{PASS} {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"{FAIL} {name}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. Model imports
# ---------------------------------------------------------------------------
def test_model_imports():
    from env.models import (
        Observation, Action, DeviationReport,
        PatientRecord, ProtocolRules, StepInfo
    )
    # Ensure they're real Pydantic models
    assert hasattr(Observation, "model_fields")
    assert hasattr(Action, "model_fields")

check("Pydantic model imports", test_model_imports)


# ---------------------------------------------------------------------------
# 2. Data generator — all 3 tasks produce correct patient/deviation counts
# ---------------------------------------------------------------------------
def test_data_generator():
    from env.data_generator import (
        build_task1_scenario, build_task2_scenario, build_task3_scenario
    )
    p1, gt1 = build_task1_scenario(42)
    assert len(p1) == 3, f"Task1: expected 3 patients, got {len(p1)}"
    assert len(gt1) == 1, f"Task1: expected 1 deviation, got {len(gt1)}"

    p2, gt2 = build_task2_scenario(42)
    assert len(p2) == 5, f"Task2: expected 5 patients, got {len(p2)}"
    assert len(gt2) == 4, f"Task2: expected 4 deviations, got {len(gt2)}"

    p3, gt3 = build_task3_scenario(42)
    assert len(p3) == 8, f"Task3: expected 8 patients, got {len(p3)}"
    assert len(gt3) == 7, f"Task3: expected 7 deviations, got {len(gt3)}"

check("Data generator — patient and deviation counts", test_data_generator)


# ---------------------------------------------------------------------------
# 3. Determinism — same seed, same output
# ---------------------------------------------------------------------------
def test_determinism():
    from env.data_generator import build_task2_scenario
    p_a, gt_a = build_task2_scenario(99)
    p_b, gt_b = build_task2_scenario(99)
    ids_a = [p.patient_id for p in p_a]
    ids_b = [p.patient_id for p in p_b]
    assert ids_a == ids_b, "Patient IDs differ across same-seed calls"
    dev_a = [d["deviation_type"] for d in gt_a]
    dev_b = [d["deviation_type"] for d in gt_b]
    assert dev_a == dev_b, "Deviation types differ across same-seed calls"

check("Determinism — same seed produces same data", test_determinism)


# ---------------------------------------------------------------------------
# 4. Environment reset
# ---------------------------------------------------------------------------
def test_reset():
    from env.environment import ClinicalTrialEnv
    for task_id in ["task1", "task2", "task3"]:
        env = ClinicalTrialEnv(task_id=task_id, seed=42)
        obs = env.reset()
        assert obs.task_id == task_id
        assert obs.step == 0
        assert obs.patients is not None
        assert obs.protocol is not None
        assert len(obs.action_history) == 0

check("Environment reset() — all 3 tasks", test_reset)


# ---------------------------------------------------------------------------
# 5. State returns expected keys
# ---------------------------------------------------------------------------
def test_state():
    from env.environment import ClinicalTrialEnv
    env = ClinicalTrialEnv("task1", seed=42)
    env.reset()
    s = env.state()
    required_keys = {"task_id", "seed", "step", "done", "found_deviation_keys",
                     "ground_truth_count", "patient_count", "last_reward", "action_history"}
    missing = required_keys - s.keys()
    assert not missing, f"Missing state keys: {missing}"

check("state() returns all required keys", test_state)


# ---------------------------------------------------------------------------
# 6. Perfect score on task1 (correct answer = 1.0)
# ---------------------------------------------------------------------------
def test_perfect_task1():
    from env.environment import ClinicalTrialEnv
    from env.models import Action, DeviationReport
    env = ClinicalTrialEnv("task1", seed=42)
    env.reset()
    action = Action(
        reports=[DeviationReport(
            patient_id="PT-102",
            deviation_type="missed_visit",
            details="Week 4 visit not completed",
            severity="major"
        )],
        submit=True
    )
    _, reward, done, info = env.step(action)
    assert done, "Episode should be done after submit=True"
    assert reward >= 0.8, f"Perfect answer should score high, got {reward}"
    assert info["false_positives"] == 0

check("Perfect score on task1 > 0.8", test_perfect_task1)


# ---------------------------------------------------------------------------
# 7. Zero score for empty submission
# ---------------------------------------------------------------------------
def test_zero_empty():
    from env.environment import ClinicalTrialEnv
    from env.models import Action
    env = ClinicalTrialEnv("task1", seed=42)
    env.reset()
    action = Action(reports=[], submit=True)
    _, reward, done, _ = env.step(action)
    assert done
    assert reward < 0.25, f"Empty submission should score low, got {reward}"

check("Empty submission scores low", test_zero_empty)


# ---------------------------------------------------------------------------
# 8. False positive penalty
# ---------------------------------------------------------------------------
def test_false_positive_penalty():
    from env.environment import ClinicalTrialEnv
    from env.models import Action, DeviationReport
    env = ClinicalTrialEnv("task1", seed=42)
    env.reset()
    # Only false positives — should be penalized below 0
    action = Action(
        reports=[
            DeviationReport(patient_id="PT-101", deviation_type="wrong_dose",
                            details="Wrong dose", severity="major"),
            DeviationReport(patient_id="PT-103", deviation_type="overdose",
                            details="Overdose", severity="critical"),
        ],
        submit=True
    )
    _, reward, done, info = env.step(action)
    assert done
    assert reward < 0.3, f"Pure false positives should score low, got {reward}"
    assert info["false_positives"] == 2

check("False positive penalty applied correctly", test_false_positive_penalty)


# ---------------------------------------------------------------------------
# 9. Reward in [0, 1] range for all tasks
# ---------------------------------------------------------------------------
def test_reward_range():
    from env.environment import ClinicalTrialEnv
    from env.models import Action, DeviationReport
    from env.data_generator import (
        build_task1_scenario, build_task2_scenario, build_task3_scenario
    )
    builders = {
        "task1": build_task1_scenario,
        "task2": build_task2_scenario,
        "task3": build_task3_scenario,
    }
    for task_id, builder in builders.items():
        _, gt = builder(42)
        env = ClinicalTrialEnv(task_id, seed=42)
        env.reset()
        # Submit all correct answers
        reports = [
            DeviationReport(
                patient_id=d["patient_id"],
                deviation_type=d["deviation_type"],
                details=d["details"],
                severity=d["severity"]
            )
            for d in gt
        ]
        action = Action(reports=reports, submit=True)
        _, reward, done, _ = env.step(action)
        assert 0.0 < reward < 1.0, f"{task_id}: reward {reward} out of (0,1)"
        assert reward >= 0.8, f"{task_id}: perfect answer should score high, got {reward}"

check("Reward in (0.0, 1.0) range for all tasks", test_reward_range)


# ---------------------------------------------------------------------------
# 10. Multi-step iterative improvement
# ---------------------------------------------------------------------------
def test_multi_step():
    from env.environment import ClinicalTrialEnv
    from env.models import Action, DeviationReport
    from env.data_generator import build_task2_scenario

    _, gt = build_task2_scenario(42)
    env = ClinicalTrialEnv("task2", seed=42)
    env.reset()

    rewards = []
    # Step 1: find 1 deviation
    action1 = Action(reports=[DeviationReport(**gt[0])], submit=False)
    _, r1, done1, _ = env.step(action1)
    rewards.append(r1)
    assert not done1

    # Step 2: find another deviation
    action2 = Action(reports=[DeviationReport(**gt[0]), DeviationReport(**gt[1])], submit=False)
    _, r2, done2, _ = env.step(action2)
    rewards.append(r2)

    # Step 3: submit all
    action3 = Action(reports=[DeviationReport(**d) for d in gt], submit=True)
    _, r3, done3, _ = env.step(action3)
    rewards.append(r3)

    assert done3, "Should be done after submit=True"
    assert r3 >= 0.8, f"Perfect final answer should score high, got {r3}"
    assert r1 > 0, f"First correct deviation should give positive reward, got {r1}"

check("Multi-step iterative improvement works", test_multi_step)


# ---------------------------------------------------------------------------
# 11. Max steps forces done
# ---------------------------------------------------------------------------
def test_max_steps():
    from env.environment import ClinicalTrialEnv
    from env.models import Action
    env = ClinicalTrialEnv("task1", seed=42)
    env.reset()
    done = False
    for _ in range(5):
        _, _, done, _ = env.step(Action(reports=[], submit=False))
    assert done, "Episode should be done after MAX_STEPS=5 steps"

check("Max steps (5) forces episode end", test_max_steps)


# ---------------------------------------------------------------------------
# 12. Observation serializes to JSON
# ---------------------------------------------------------------------------
def test_observation_json():
    from env.environment import ClinicalTrialEnv
    env = ClinicalTrialEnv("task3", seed=42)
    obs = env.reset()
    dumped = obs.model_dump()
    serialized = json.dumps(dumped)
    assert len(serialized) > 100
    reparsed = json.loads(serialized)
    assert reparsed["task_id"] == "task3"

check("Observation serializes/deserializes as JSON", test_observation_json)


# ---------------------------------------------------------------------------
# 13. Hint present on step 0 for task1 only
# ---------------------------------------------------------------------------
def test_hint_logic():
    from env.environment import ClinicalTrialEnv
    from env.models import Action

    # Task1 step 0 → hint present
    env = ClinicalTrialEnv("task1", seed=42)
    obs = env.reset()
    assert obs.hint is not None, "task1 step 0 should have a hint"

    # Task1 step 1 → no hint
    obs2, _, _, _ = env.step(Action(reports=[], submit=False))
    assert obs2.hint is None, "task1 step 1 should not repeat hint"

    # Task2 step 0 → no hint
    env2 = ClinicalTrialEnv("task2", seed=42)
    obs3 = env2.reset()
    assert obs3.hint is None, "task2 should have no hint"

check("Hint logic — task1 step 0 only", test_hint_logic)


# ---------------------------------------------------------------------------
# 14. Action history accumulates across steps
# ---------------------------------------------------------------------------
def test_action_history():
    from env.environment import ClinicalTrialEnv
    from env.models import Action
    env = ClinicalTrialEnv("task2", seed=42)
    env.reset()
    env.step(Action(reports=[], submit=False))
    obs2, _, _, _ = env.step(Action(reports=[], submit=False))
    assert len(obs2.action_history) == 2, f"Expected 2 history entries, got {len(obs2.action_history)}"

check("Action history accumulates across steps", test_action_history)


# ---------------------------------------------------------------------------
# 15. Task3 decoy patient (PT-306) has no ground truth deviation
# ---------------------------------------------------------------------------
def test_task3_decoy():
    from env.data_generator import build_task3_scenario
    _, gt = build_task3_scenario(42)
    gt_patients = {d["patient_id"] for d in gt}
    # PT-307 is index 6 (patients[6]) — intentionally clean decoy with borderline labs
    assert "PT-307" not in gt_patients, "PT-307 is the decoy — should have no ground truth deviation"
    # PT-308 (index 7) should have the out-of-window visit deviation
    assert "PT-308" in gt_patients, "PT-308 should have a deviation (out-of-window visit)"

check("Task3 decoy patient (PT-306) has no ground truth deviation", test_task3_decoy)


# ---------------------------------------------------------------------------
# 16. openenv.yaml is valid YAML with required fields
# ---------------------------------------------------------------------------
def test_yaml():
    import yaml
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "openenv.yaml")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    required = {"name", "version", "tasks", "observation_space", "action_space", "reward"}
    missing = required - data.keys()
    assert not missing, f"openenv.yaml missing fields: {missing}"
    assert len(data["tasks"]) >= 3, "Need at least 3 tasks"
    for task in data["tasks"]:
        assert "id" in task
        assert "difficulty" in task

check("openenv.yaml is valid with all required fields", test_yaml)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*52)
passed = sum(1 for r, _ in results if r == PASS)
failed = sum(1 for r, _ in results if r == FAIL)
print(f"Results: {passed}/{len(results)} passed")
if failed:
    print(f"\n{FAIL} {failed} test(s) failed — fix before submitting!")
    sys.exit(1)
else:
    print(f"\n{PASS} All tests passed — ready to submit!")
    sys.exit(0)
