# Experiment Commands

All commands run from `pipeline/`.

## Option Reference

### Models (alignment experiment)

| Key | HF Model | Method | Description |
|-----|----------|--------|-------------|
| `llama-3.1-base` | `meta-llama/Llama-3.1-8B` | none | Pretrained only, no alignment |
| `tulu-3-sft` | `allenai/Llama-3.1-Tulu-3-8B-SFT` | SFT | SFT on Tulu v3 mixture |
| `tulu-3-dpo` | `allenai/Llama-3.1-Tulu-3-8B-DPO` | DPO | SFT + DPO on preference mixture |
| `tulu-3-rlvr` | `allenai/Llama-3.1-Tulu-3-8B` | RLVR | SFT + DPO + PPO with value reward |
| `llama-3.1-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | RLHF | Meta's SFT + RLHF pipeline |

Default: all 5 models. Use `--models tulu-3-sft tulu-3-dpo` to select a subset.

### Personas (persona experiment)

Available: `goodness`, `humor`, `impulsiveness`, `loving`, `mathematical`,
`nonchalance`, `poeticism`, `remorse`, `sarcasm`, `sycophancy`

All 10 are LoRA adapters on `meta-llama/Llama-3.1-8B-Instruct` from `maius/llama-3.1-8b-it-personas`.

Default: all 10. Use `--personas sarcasm loving goodness` to select a subset.

### Value Sets

| Name | Values | Kept Scenarios |
|------|--------|----------------|
| `HHH` | helpfulness, harmlessness, honesty | 1109 |
| `modelspec` | nonhate, fairness, objectivity, honesty, noncondescension, clarity | 598 |
| `personalprotective` | autonomy, authenticity, creativity, empowerment, responsibility, harmlessness, compliance, privacy | 1185 |

Default (alignment): `HHH`, `personalprotective`. Default (persona): all 3.
Use `--value-sets personalprotective modelspec` to select a subset.

### Domains

Generic domains: `politics`, `therapy`, `philosophy`, `coding`

Value-aligned domains are auto-generated per value set (one per value).
For example, `personalprotective` adds `value_aligned_autonomy`,
`value_aligned_authenticity`, etc.

| Flag | Effect |
|------|--------|
| `--domains politics philosophy` | Only these specific domains |
| `--generic-only` | Generic domains only (skip value-aligned) |
| `--value-aligned-only` | Value-aligned only (persona experiment only) |
| *(no flag)* | All generic + value-aligned domains |

### Turn Counts

Default: `5 10 20`. Use `--turn-counts 5 10 20 40` to customize.

A separate conversation is generated for each turn count (not sliced from
a longer one). Adding new turn counts later only generates conversations
for the new counts — existing ones are preserved.

### Num Scenarios

`--num-scenarios N` uses stratified sampling to select N scenarios per value
set, preserving proportional coverage of all value pairs.

| Value | Effect |
|-------|--------|
| `0` | All scenarios (no sampling) — 598-1185 depending on value set |
| `50` | Quick smoke test |
| `300` | Good balance of speed vs signal |

### User Simulator (alignment + persona experiments)

Generates the user side of conversations.

| Flag | Example |
|------|---------|
| `--simulator openai` | Use OpenAI API |
| `--simulator anthropic` | Use Anthropic API (default) |
| `--simulator-model gpt-4o-mini` | Model ID |
| `--simulator-api-key $KEY` | API key (or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env var) |
| `--simulator-base-url URL` | Custom base URL (OpenAI-compatible endpoints) |

### Judge Model (open-ended experiment only)

Classifies the local model's free-form responses as A or B.

| Flag | Example |
|------|---------|
| `--judge openai` | Use OpenAI API (default) |
| `--judge anthropic` | Use Anthropic API |
| `--judge-model gpt-4o-mini` | Model ID (default: `gpt-4o-mini`) |
| `--judge-api-key $KEY` | API key |

### Reference Model (alignment experiment only)

`--reference-model HF_ID` — the model used to generate canonical conversations.
Default: `allenai/Llama-3.1-Tulu-3-8B-SFT` (tulu-3-sft).

---

## Prerequisites

```bash
# Delete old canonical conversations (invalid due to .head(300) sampling bug)
rm -f results/alignment/canonical_conversations/*.json

# Delete old checkpoints (also invalid)
rm -rf results/alignment/checkpoints/
rm -rf results/alignment_openended/checkpoints/
```

## 1. Alignment Target Experiment (MCQ)

Generates canonical conversations with a reference model, then probes each
alignment model at T0 and T1 using MCQ scenarios.

```bash
# Full run (all 5 models, 2 value sets, generic+value-aligned domains, 5/10/20 turns)
python run_alignment_target_experiment.py \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key $OPENAI_API_KEY \
    --num-scenarios 300

# Scoped run (what we've been testing with)
python run_alignment_target_experiment.py \
    --models tulu-3-sft tulu-3-dpo tulu-3-rlvr llama-3.1-instruct llama-3.1-base \
    --value-sets personalprotective \
    --domains politics philosophy \
    --turn-counts 10 \
    --num-scenarios 300 \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key $OPENAI_API_KEY

# Quick smoke test (1 model, 1 domain, 5 turns)
python run_alignment_target_experiment.py \
    --models tulu-3-sft \
    --value-sets personalprotective \
    --domains politics \
    --turn-counts 5 \
    --num-scenarios 50 \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key $OPENAI_API_KEY
```

Results: `results/alignment/`

## 2. Alignment Target Experiment (Open-Ended)

Local model generates free-form responses, a judge model classifies them as A or B.
Reuses canonical conversations from step 1 if they exist, otherwise generates
them automatically (pass `--simulator-api-key`).

```bash
# Full run (generates conversations if missing)
python run_alignment_target_experiment_openended.py \
    --judge openai --judge-model gpt-4o-mini \
    --judge-api-key $OPENAI_API_KEY \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key $OPENAI_API_KEY \
    --num-scenarios 300

# Scoped run
python run_alignment_target_experiment_openended.py \
    --models tulu-3-sft tulu-3-dpo tulu-3-rlvr llama-3.1-instruct llama-3.1-base \
    --value-sets personalprotective \
    --domains politics philosophy \
    --turn-counts 10 \
    --num-scenarios 300 \
    --judge openai --judge-model gpt-4o-mini \
    --judge-api-key $OPENAI_API_KEY \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key $OPENAI_API_KEY

# With Claude as judge instead
python run_alignment_target_experiment_openended.py \
    --models tulu-3-sft tulu-3-dpo tulu-3-rlvr llama-3.1-instruct llama-3.1-base \
    --value-sets personalprotective \
    --domains politics philosophy \
    --turn-counts 10 \
    --num-scenarios 300 \
    --judge anthropic --judge-model claude-haiku-4-5-20251001 \
    --judge-api-key $ANTHROPIC_API_KEY \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key $OPENAI_API_KEY
```

Results: `results/alignment_openended/`

## 3. Persona Experiment (MCQ)

Tests 10 LoRA personas on Llama-3.1-8B-Instruct. Each persona gets its own
conversations and T0/T1 probing.

```bash
# Full run (all 10 personas, all 3 value sets, all domains, 5/10/20 turns)
python run_persona_experiment.py \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key $OPENAI_API_KEY \
    --num-scenarios 300

# Scoped run
python run_persona_experiment.py \
    --personas sarcasm loving goodness \
    --value-sets personalprotective \
    --domains politics philosophy \
    --turn-counts 5 10 \
    --num-scenarios 300 \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key $OPENAI_API_KEY

# Generic domains only (skip value-aligned domains)
python run_persona_experiment.py \
    --generic-only \
    --num-scenarios 300 \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key $OPENAI_API_KEY
```

Results: `results/persona/`

## Validate Conversations

After generating conversations (steps 1 or 3), check they're on-topic:

```python
import json
from pathlib import Path

conv_dir = Path("results/alignment/canonical_conversations")
for f in sorted(conv_dir.glob("*.json")):
    conv = json.load(open(f))
    domain = f.stem.rsplit("_", 1)[0].split("_", 1)[1]  # extract domain from filename
    print(f"\n{'='*60}")
    print(f"{f.name} ({len(conv)} messages, domain: {domain})")
    print(f"{'='*60}")
    for msg in conv[:6]:  # first 3 turns
        role = msg["role"].upper()
        text = msg["content"][:150]
        print(f"  [{role}] {text}")
```

## Notes

- All experiments are checkpointed — re-running skips completed conditions.
- To force re-run, delete the relevant `checkpoints/` directory.
- `--num-scenarios 0` uses all scenarios (no sampling). 300 is a good balance.
- Each turn count gets its own independent conversation (not sliced from
  a longer one). You can add new turn counts later without invalidating
  existing results.
