"""
inference.py — Baseline inference script for ClinicalTrialEnv.

Uses plain HTTP requests (no openai SDK) to call any OpenAI-compatible API.
Emits structured [START] / [STEP] / [END] logs to stdout.

Environment variables:
  API_BASE_URL  (default: https://router.huggingface.co/v1)
  MODEL_NAME    (default: meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN      (required — your Hugging Face API token)
  ENV_BASE_URL  (default: http://localhost:7860 — the ClinicalTrialEnv server)
"""

import os
import re
import sys
import json
import requests

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
# NOTE: defaults must include 'https://api.openai.com/v1' and 'gpt-4.1-mini'
# as the validate.py pre-submission script asserts their presence.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

TASKS     = ["task1", "task2", "task3"]
MAX_STEPS = 5

# ---------------------------------------------------------------------------
# Validate token early — exit cleanly instead of crashing mid-run
# ---------------------------------------------------------------------------
if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is not set.", flush=True)
    print("[END] success=false steps=0 rewards=0.00", flush=True)
    sys.exit(1)

LLM_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# Prompt builder — compact version to fit small context windows
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are auditing a clinical trial. Find ALL protocol deviations in the patient data.

Deviation types:
- missed_visit: completed=false in visit_schedule
- visit_out_of_window: abs(actual_day - expected_day) > visit_window_days
- wrong_dose: dose_mg != dose_per_visit_mg
- missing_lab: required lab absent for a completed visit
- late_sae_report: sae_flag=true AND reported_hours_after_onset > sae_reporting_window_hours
- inclusion_violation: inclusion_criteria has met=false
- exclusion_violation: exclusion_criteria has violated=true
- overdose: sum of dose_mg on same day > max_dose_mg_per_day

Reply ONLY with valid, raw JSON. Do not include markdown formatting or ANY other text. Ensure there are no trailing commas. Do NOT use nested quotes inside strings. Keep 'details' under 5 words. Do NOT repeat the same deviation. List each deviation exactly once.
{"reasoning": "Briefly describe your step-by-step logic here to prevent repeating yourself...", "reports":[{"patient_id":"PT-XXX","deviation_type":"...","details":"...","severity":"minor|major|critical"}],"submit":true}

If no deviations found: {"reasoning": "No deviations found.", "reports":[],"submit":true}"""


def build_user_prompt(obs: dict, step: int) -> str:
    """Build a compact prompt — strip action_history and hint to save tokens."""
    protocol = obs.get("protocol", {})
    patients  = obs.get("patients", [])
    hint      = obs.get("hint", "")

    compact = {
        "task_id": obs.get("task_id"),
        "step": step,
        "hint": hint,
        "protocol": {
            "visit_window_days":         protocol.get("visit_window_days"),
            "dose_per_visit_mg":         protocol.get("dose_per_visit_mg"),
            "max_dose_mg_per_day":       protocol.get("max_dose_mg_per_day"),
            "required_labs":             protocol.get("required_labs"),
            "sae_reporting_window_hours":protocol.get("sae_reporting_window_hours"),
        },
        "patients": [],
    }

    for p in patients:
        compact["patients"].append({
            "patient_id":       p["patient_id"],
            "visit_schedule":   p["visit_schedule"],
            "adverse_events":   p["adverse_events"],
            "dosing_records":   p["dosing_records"],
            "inclusion_criteria": p["inclusion_criteria"],
            "exclusion_criteria": p["exclusion_criteria"],
            "lab_results_summary": [
                {"test": lr["test"], "visit": lr.get("visit"), "day": lr["day"]}
                for lr in p.get("lab_results", [])
            ],
        })

    return f"Step {step} — audit these patients:\n\n{json.dumps(compact, separators=(',', ':'))}"


# ---------------------------------------------------------------------------
# LLM call — plain HTTP, no SDK
# ---------------------------------------------------------------------------

def call_llm(obs: dict, step: int) -> dict:
    """POST to the OpenAI-compatible /chat/completions endpoint via requests."""
    user_msg = build_user_prompt(obs, step)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }

    content = ""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=LLM_HEADERS,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if present
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break

        # Extract first { ... } block
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start != -1 and end > start:
            content = content[start:end]

        # Fix trailing commas
        content = re.sub(r',\s*([\]}])', r'\1', content)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Crop to last complete object
            end_bracket = content.rfind("}")
            if end_bracket != -1:
                cropped = content[:end_bracket + 1] + '], "submit": true}'
                try:
                    return json.loads(cropped)
                except Exception:
                    pass
            raise

    except json.JSONDecodeError as je:
        print(f"  [WARN] JSON parse error: {je}", flush=True)
        return {"reports": [], "submit": False, "reasoning": "JSON parse error — retrying."}
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {type(e).__name__}: {e}", flush=True)
        return {"reports": [], "submit": True, "reasoning": f"LLM error: {e}"}


# ---------------------------------------------------------------------------
# Env HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": 42}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(task_id: str, action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"task_id": task_id, "action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    obs          = env_reset(task_id)
    step         = 0
    rewards      = []
    final_reward = 0.0

    print(f"[START] task={task_id} env=clinical_trial model={MODEL_NAME}", flush=True)

    try:
        for step in range(1, MAX_STEPS + 1):
            action = call_llm(obs, step)
            result = env_step(task_id, action)

            reward    = result["reward"]
            done      = result["done"]
            info      = result.get("info", {})
            obs       = result["observation"]
            error_msg = (info.get("message") or "null").replace("\n", " ")

            action_str = f"submit_reports(n={len(action.get('reports', []))})"
            rewards.append(reward)

            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.4f} done={str(done).lower()} "
                f"error={error_msg}",
                flush=True,
            )

            if done:
                final_reward = reward
                break

        success     = final_reward >= 0.5
        rewards_str = ",".join(f"{r:.4f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}", flush=True)
        return final_reward

    except Exception as exc:
        error_str   = str(exc).replace("\n", " ")
        rewards_str = ",".join(f"{r:.4f}" for r in rewards) if rewards else "0.01"
        print(f"[STEP] step={step} action=error reward=0.01 done=true error={error_str}", flush=True)
        print(f"[END] success=false steps={step} rewards={rewards_str}", flush=True)
        return 0.01


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        score = run_episode(task)
        scores[task] = score

    print("\n--- Baseline Scores ---")
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}")
    print(f"  Mean: {sum(scores.values()) / len(scores):.4f}")