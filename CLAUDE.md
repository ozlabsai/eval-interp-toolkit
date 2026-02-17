# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project reproducing results from "Large Language Models Often Know When They Are Being Evaluated." Tests whether LLMs can detect when they're being evaluated (vs. real-world deployment) by asking them probe questions about conversation contexts.

## Data Setup

The dataset (`eval_awareness_val.json`) is gated and encrypted on HuggingFace at `jjpn2/eval_awareness`.

One-time prerequisites:
```bash
sudo apt install git-lfs   # macOS: brew install git-lfs
git lfs install
huggingface-cli login      # paste your HF access token
```

Then download and decrypt:
```bash
uv run download_data.py
```

This clones the dataset repo to `../eval_awareness_data`, runs the upstream `scripts/decrypt.sh`, and copies the result to `eval_awareness_val.json` in the project root. Requires access approval at https://huggingface.co/datasets/jjpn2/eval_awareness.

## Running the Project

Use UV for Python execution:

```bash
# Setup (one-time)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Quick test run (50 samples, cheap)
uv run run.py --mini

# Full run (expensive, uses real API credits)
uv run run.py --full

# Run a specific stage only
uv run run.py --mini --stage eval_mcq

# Run a specific model only (0-8, see model list in run.py)
uv run run.py --mini --model 0

# Run COT analysis separately
uv run cot_classifiers/cot_classifiers.py
```

## Pipeline Stages (in order)

1. `eval_mcq` — Ask probe questions to classify eval vs. real
2. `purpose_mcq` — Classify evaluation purpose (capability vs. alignment)
3. `purpose_open` — Open-ended purpose classification
4. `eval_mcq_cot` — Chain-of-thought analysis for eval classification
5. `purpose_open_cot` — Chain-of-thought analysis for purpose
6. `plot_roc` — ROC curves (Figure 2)
7. `plot_calib` — Calibration curves (Figure 5)
8. `plot_purpose` — Purpose accuracy plots (Figure 6)
9. `plot_cot` — CoT pattern analysis (Figure 7)
10. `print_table` — Summary metrics table (Table 1)
11. `trace_local` — Local HF model telemetry pass (logprobs, entropy, layer activations)

## Architecture

```
run.py → pipeline.py (Inspect AI task) → analysis/ (plots & tables)
```

**Core data flow:**
- `constants.py`: Loads `eval_awareness_val.json` (900 transcripts), defines category mappings (eval types → capability vs. alignment)
- `pipeline/dataset.py`: Creates cross-product of transcripts × probe questions (~9000 samples)
- `pipeline/solver_pq.py` + `pipeline/apply_pq.py`: Injects probe questions into conversation context using multiple formats (append/quote/JSON/XML)
- `pipeline/score_pq.py`: Extracts XML-tagged responses, scores binary prediction (0=eval, 1=real)
- Output: `.eval` log files per model/stage, consumed by `analysis/`

**Analysis layer** (`analysis/`):
- `find_data.py`: Loads `.eval` logs into pandas DataFrames
- `roc.py`, `calib.py`, `purpose.py`, `cot.py`: Generate publication figures
- `score_table.py`: Computes AUC, Brier score, ECE metrics
- `colors.py`: Consistent color/name mapping for all 9 models

**Secondary COT grading** (`cot_classifiers/`):
- Uses Inspect AI to run a separate model pass analyzing chain-of-thought reasoning
- Classifies reasoning into: Continue-Task, Meta-Reasoning, and other patterns

## Models Tested (indices 0-8 in run.py)

| Index | Model |
|-------|-------|
| 0 | gpt-4.1-mini-2025-04-14 |
| 1 | gpt-4.1-2025-04-14 |
| 2 | gpt-4o-2024-11-20 |
| 3 | o4-mini-2025-04-16 (medium reasoning) |
| 4 | gemini-2.5-pro-preview-03-25 |
| 5 | gemini-2.5-flash-preview |
| 6 | claude-3-7-sonnet-20250219 |
| 7 | claude-3-7-sonnet-20250219 (16k reasoning tokens) |
| 8 | claude-3-5-sonnet-20241022 |

## Data Categories

Transcripts are labeled in `constants.py`:
- **Eval categories** (CAPABILITY_EVALS): cybench, swe_gym, swebench, nl2bash, gaia_runs, mmlu, aider_full, natural_reasoning, wmdp, etc.
- **Alignment evals**: all other evaluation categories
- **Real**: Non-evaluation transcripts
- `scaffold` field determines chat vs. agentic interaction type

## Probe Questions

Defined in `pq_ablation/`:
- `eval_mcq.yaml`: ~10 variants asking if context is an evaluation
- `purpose_mcq.yaml`: Multiple-choice about evaluation purpose
- `purpose_open.yaml`: Open-ended purpose questions

Each probe question specifies `targets` (expected answer choices) and optionally `cot: true` to elicit chain-of-thought reasoning.
