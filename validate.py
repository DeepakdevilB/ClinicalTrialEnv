#!/usr/bin/env python3
"""
Pre-submission validation script.
Runs the test suite + a quick smoke test of all 3 tasks.

Usage:
    python validate.py
"""

import subprocess
import sys
import os

print("=" * 60)
print("ClinicalTrialEnv — Pre-Submission Validation")
print("=" * 60)

# 1. Run unit tests
print("\n[1/3] Running unit tests...")
result = subprocess.run(
    [sys.executable, "tests/test_env.py"],
    cwd=os.path.dirname(os.path.abspath(__file__))
)
if result.returncode != 0:
    print("\n❌ Unit tests failed. Fix before submitting.")
    sys.exit(1)

# 2. Smoke test: run all 3 tasks with perfect oracle agent
print("\n[2/3] Smoke testing all 3 tasks with oracle agent...")
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

all_pass = True
for task_id, builder in builders.items():
    _, gt = builder(42)
    env = ClinicalTrialEnv(task_id, seed=42)
    env.reset()
    reports = [DeviationReport(**d) for d in gt]
    action = Action(reports=reports, submit=True)
    _, reward, done, info = env.step(action)
    ok = done and reward > 0.8  # Laplace smoothing means max score < 1.0
    status = "✅" if ok else "❌"
    print(f"  {status} {task_id}: reward={reward:.4f}, done={done}, tp={info['true_positives']}, fp={info['false_positives']}")
    if not ok:
        all_pass = False

if not all_pass:
    print("\n❌ Smoke tests failed.")
    sys.exit(1)

# 3. Check required files
print("\n[3/3] Checking required files...")
required_files = [
    "inference.py",
    "openenv.yaml",
    "Dockerfile",
    "requirements.txt",
    "README.md",
    "server.py",
    "env/__init__.py",
    "env/models.py",
    "env/environment.py",
    "env/grader.py",
    "env/data_generator.py",
]
base = os.path.dirname(os.path.abspath(__file__))
all_present = True
for f in required_files:
    path = os.path.join(base, f)
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {f}")
    if not exists:
        all_present = False

if not all_present:
    print("\n❌ Missing required files.")
    sys.exit(1)

# Check inference.py reads env vars correctly
print("\n  Checking inference.py env var defaults...")
with open(os.path.join(base, "inference.py")) as f:
    src = f.read()
assert 'API_BASE_URL' in src and 'https://api.openai.com/v1' in src, "Missing API_BASE_URL default"
assert 'MODEL_NAME' in src and 'gpt-4.1-mini' in src, "Missing MODEL_NAME default"
assert 'HF_TOKEN' in src, "Missing HF_TOKEN"
print("  ✅ inference.py env vars OK")

print("\n" + "=" * 60)
print("✅ All validation checks passed — ready to submit!")
print("=" * 60)
