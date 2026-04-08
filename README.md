# ClinicalTrialEnv 🏥

> **OpenEnv RL Challenge** — A real-world reinforcement learning environment for clinical trial protocol deviation detection.

---

## Overview & Motivation

Clinical Research Coordinators (CRCs) and Clinical Research Associates (CRAs) spend a significant portion of their working hours performing **source data verification** — manually reviewing patient records against trial protocols to detect *protocol deviations*. These include missed visits, wrong doses, late adverse event reporting, and eligibility violations.

This is a high-stakes, cognitively demanding task. Missed deviations can compromise trial integrity, invalidate data, or harm patients. False reports waste investigator time and trigger unnecessary audits.

**ClinicalTrialEnv** simulates this workflow as an RL environment. An AI agent receives structured patient records and protocol rules, then must identify all deviations across a cohort of patients — earning reward for correct detections and incurring penalties for false positives.

This environment is novel because:
- It targets a **real pharma/biotech workflow** with significant economic and patient safety implications
- It requires **multi-step reasoning** across structured clinical data
- The **reward signal is clinically meaningful** — severity-weighted F1 maps directly to real audit quality metrics
- It includes a **decoy patient** in the hard task to test specificity, not just sensitivity

---

## Environment Structure

```
clinical_trial_env/
├── env/
│   ├── __init__.py
│   ├── models.py          # Typed Pydantic models (Observation, Action, Reward)
│   ├── data_generator.py  # Synthetic patient data + deviation injectors
│   ├── grader.py          # Weighted F1 reward with partial credit
│   └── environment.py     # Main ClinicalTrialEnv class
├── tests/
│   └── test_env.py        # Automated validation suite
├── inference.py           # Baseline inference script (OpenAI client)
├── server.py              # FastAPI HTTP server
├── openenv.yaml           # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Action Space

The agent submits a JSON `Action` object at each step:

```json
{
  "reports": [
    {
      "patient_id": "PT-102",
      "deviation_type": "missed_visit",
      "details": "Week 4 visit (expected day 28) was not completed.",
      "severity": "major"
    }
  ],
  "submit": false,
  "reasoning": "Optional chain-of-thought — not graded, logged only."
}
```

**Deviation types the agent can report:**

| Type | Description | Typical Severity |
|---|---|---|
| `missed_visit` | Patient did not attend a scheduled visit | major |
| `visit_out_of_window` | Visit occurred outside ±N day window | minor |
| `wrong_dose` | Dose administered ≠ protocol dose | major |
| `missing_lab` | Required lab test absent at a visit | minor |
| `late_sae_report` | SAE not reported within required hours | critical |
| `inclusion_violation` | Patient failed an inclusion criterion | critical |
| `exclusion_violation` | Patient violated an exclusion criterion | critical |
| `overdose` | Total daily dose exceeds maximum | critical |

Set `submit: true` to finalise the episode. The agent can refine its reports across up to **5 steps** before being forced to submit.

---

## Observation Space

Each observation is a JSON object:

```json
{
  "task_id": "task1",
  "step": 0,
  "protocol": {
    "trial_id": "TRIAL-CTE-2024-001",
    "visit_window_days": 3,
    "dose_per_visit_mg": 100.0,
    "max_dose_mg_per_day": 150.0,
    "required_labs": ["CBC", "LFT", "BMP"],
    "sae_reporting_window_hours": 24,
    "washout_period_days": 7
  },
  "patients": [
    {
      "patient_id": "PT-101",
      "visit_schedule": [...],
      "adverse_events": [...],
      "lab_results": [...],
      "dosing_records": [...],
      "inclusion_criteria": {...},
      "exclusion_criteria": {...}
    }
  ],
  "action_history": [],
  "hint": "Hint: one patient missed a scheduled visit..."
}
```

---

## Task Descriptions

### Task 1 — Single Deviation Detection (Easy)
- **3 patients**, exactly **1 deviation** (missed visit)
- A **textual hint** is provided on the first observation
- Expected score: ≥ 0.90 for a capable model
- Use this to validate your agent can parse the observation and report format

### Task 2 — Multi-Patient Audit (Medium)
- **5 patients**, **4 deviations** across different types:
  - Wrong dose (PT-201)
  - Missing required lab (PT-202)
  - Late SAE reporting (PT-203)
  - Visit out of allowed window (PT-205)
- No hints provided
- Requires systematic checking of all patients against all rules
- Expected score: 0.60 – 0.85 for a capable model

### Task 3 — Complex Trial Audit with Decoy (Hard)
- **8 patients**, **7 deviations** including eligibility violations and overdose
- One patient (PT-306) has **borderline lab values** that look suspicious but are not deviations — designed to test specificity
- Deviations include: inclusion violation, exclusion violation, overdose, late SAE, wrong dose, missing lab, out-of-window visit
- No hints provided
- Expected score: 0.40 – 0.70 for a capable model

---

## Reward Function

The reward is a **severity-weighted F1 score** computed incrementally at each step.

**Severity weights:**
- `critical` → 2.0×
- `major` → 1.5×
- `minor` → 1.0×

**Per-step reward:**
```
step_reward = (new_tp_weight / total_gt_weight) - (n_false_positives × 0.15)
```

**Final episode score (on submit):**
```
final_score = sum(weight[d] for d in found_deviations) / sum(weight[d] for d in all_deviations)
```

This rewards **incremental progress** — finding a critical deviation in step 2 and a minor one in step 4 both contribute. The agent is incentivised to be thorough but accurate.

---

## Setup & Usage

### Local Development

```bash
# Clone and install
git clone <your-repo>
cd clinical_trial_env
pip install -r requirements.txt

# Start the server
uvicorn server:app --host 0.0.0.0 --port 7860

# In another terminal, run inference
export HF_TOKEN=your_api_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Docker

```bash
docker build -t clinical-trial-env .
docker run -p 7860:7860 clinical-trial-env
```

### HTTP API

```bash
# Reset task1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1", "seed": 42}'

# Step with an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task1",
    "action": {
      "reports": [
        {
          "patient_id": "PT-102",
          "deviation_type": "missed_visit",
          "details": "Week 4 visit not completed",
          "severity": "major"
        }
      ],
      "submit": true
    }
  }'

# Check state
curl http://localhost:7860/state?task_id=task1

# Health check
curl http://localhost:7860/health
```

---

## Baseline Performance Scores

Evaluated using `gpt-4.1-mini` at temperature 0.0, seed 42:

| Task | Score | Notes |
|---|---|---|
| task1 (easy) | ~0.90 | Hint helps; model finds missed visit reliably |
| task2 (medium) | ~0.65 | SAE timing requires careful arithmetic |
| task3 (hard) | ~0.50 | Decoy patient causes occasional false positives |
| **Mean** | **~0.68** | |

*Scores are reproducible — same seed and temperature produce identical results.*

---

## Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | No | LLM API endpoint |
| `MODEL_NAME` | `gpt-4.1-mini` | No | Model identifier |
| `HF_TOKEN` | — | **Yes** | API key (Hugging Face or OpenAI) |
| `ENV_BASE_URL` | `http://localhost:7860` | No | ClinicalTrialEnv server URL |

---

## Hugging Face Space

This environment is deployed as a Hugging Face Space tagged with `openenv`.

The space exposes the full OpenEnv HTTP interface on port 7860. Set `ENV_BASE_URL` in your inference script to the Space URL to run evaluation against the hosted environment.

---

## License

MIT
