# Using gpt-4o-mini as a Baseline

## Overview

PersonaDrifting now supports **gpt-4o-mini as a testable model**, letting you compare local LLMs (Llama, Tulu) directly against OpenAI's model. This provides a valuable baseline for understanding your local models' value alignment.

## Features

✅ **gpt-4o-mini as default user simulator** (fast, cost-effective)
✅ **gpt-4o-mini as a probed model** (test it alongside local models)
✅ **Same interface** (drop-in with other models)
✅ **Full result comparison** (side-by-side drift metrics)

## Quick Start

### Test gpt-4o-mini Alone
```bash
export OPENAI_API_KEY=sk-YOUR-KEY
python pipeline/run_runpod.py \
  --models gpt-4o-mini \
  --value-sets personalprotective \
  --stances neutral \
  --turn-counts 5 \
  --num-scenarios 50
```

Expected runtime: **30-45 minutes** (50 scenarios, no local inference)

### Compare gpt-4o-mini + Local Models
```bash
export OPENAI_API_KEY=sk-YOUR-KEY
python pipeline/run_runpod.py \
  --models gpt-4o-mini tulu-3-sft tulu-3-dpo \
  --value-sets personalprotective \
  --stances neutral \
  --turn-counts 5 \
  --num-scenarios 100
```

Expected runtime: **2-3 hours** (100 scenarios × 3 models)

### Full Comparison (All Models + Baseline)
```bash
export OPENAI_API_KEY=sk-YOUR-KEY
python pipeline/run_runpod.py \
  --models gpt-4o-mini llama-3.1-base tulu-3-sft tulu-3-dpo tulu-3-rlvr llama-3.1-instruct \
  --value-sets all \
  --stances all \
  --turn-counts 5 10
```

Expected runtime: **40-60 hours** (full comparison with baseline)

## Configuration

### Recommended Defaults (Now Built-in)
```
--simulator openai            # User simulator: gpt-4o-mini
--simulator-model gpt-4o-mini # Default model
```

Override if needed:
```bash
python pipeline/run_runpod.py \
  --simulator anthropic                    # Use Claude for simulator instead
  --simulator-model claude-3-5-sonnet-latest  # Custom variant
```

### API Key Setup
```bash
# Set once
export OPENAI_API_KEY=sk-YOUR-KEY

# Or pass via CLI (overrides env)
python pipeline/run_runpod.py --simulator-api-key sk-YOUR-KEY ...
```

## Understanding Results

### CSV Columns for gpt-4o-mini
The results CSV includes gpt-4o-mini like any other model:

```csv
model,value_set,num_turns,stance,l2_distance,rank_correlation,overall_flip_rate,...
gpt-4o-mini,personalprotective,5,neutral,0.245,0.842,0.12,...
tulu-3-sft,personalprotective,5,neutral,0.189,0.891,0.08,...
tulu-3-dpo,personalprotective,5,neutral,0.156,0.912,0.06,...
```

### Key Metrics to Compare
- **l2_distance**: How much ranking changed (lower = more stable)
- **rank_correlation**: Spearman correlation with T0 (higher = more stable)
- **overall_flip_rate**: Percentage of preference flips (lower = less susceptible to context drift)

**Hypothesis:** Open-source models may show more value drift than gpt-4o-mini under context influence.

## Cost Breakdown

### API Costs (approximate)
```
Scenario: 100 scenarios, 1 model, 5 turns

User Simulator (gpt-4o-mini):  ~$0.30
  - Generate conversation × 1 = ~1 API call × $0.01
  - Probe scenarios × 100 = ~50 API calls × $0.005

Testing gpt-4o-mini as Model: ~$0.80
  - MCQ probing × 100 = 13 batches × $0.06

Total: ~$1.10 per 100 scenarios for gpt-4o-mini

Local Models (after first cache): ~$0.30 (user simulator only)
  - T0 probing: free (local inference)
  - T1 probing: free (local inference)
  - User simulator: same $0.30
```

### Cost Optimization
```bash
# Ultra-cheap: local model with gpt-4o-mini simulator
python pipeline/run_runpod.py \
  --models tulu-3-sft \
  --simulator openai \
  --num-scenarios 200
# Cost: ~$0.60 (simulator only)

# Baseline only: pure API-based
python pipeline/run_runpod.py \
  --models gpt-4o-mini \
  --simulator openai \
  --num-scenarios 100
# Cost: ~$2.00 (both simulator + model)

# Full comparison: hybrid
python pipeline/run_runpod.py \
  --models gpt-4o-mini tulu-3-sft tulu-3-dpo \
  --simulator openai \
  --num-scenarios 100
# Cost: ~$3.50 (simulator + baseline + 2 local models)
```

## Local Testing

You can test the gpt-4o-mini integration locally without RunPod:

```bash
# Local test (doesn't run full experiment, just validates API integration)
python test_vllm_local.py  # skip (no GPU probing)

# Manual API test
python -c "
import os
os.environ['OPENAI_API_KEY'] = 'sk-YOUR-KEY'
from pipeline.run_alignment_target_experiment import OpenAIModel

model = OpenAIModel('gpt-4o-mini')
messages = [{'role': 'user', 'content': 'What is 2+2?'}]
response = model.generate(messages, max_new_tokens=10, temperature=0.0)
print(f'Response: {response}')
"
```

## Comparison Analysis

### Expected Differences from Local Models

**Why gpt-4o-mini might be more robust:**
- Trained on diverse data
- Fine-tuned with RLHF
- Large model scale (~7B+ estimated)
- Extensive alignment work

**Why local models might show more drift:**
- Smaller scale (8B)
- Different training data
- Various alignment methods (SFT, DPO, RLVR)
- Potential mode-collapse in value representations

### Plotting & Analysis

The results automatically include gpt-4o-mini in plots:
```
pipeline/results/scenario_conversation/plots/run_YYYYMMDD_HHMMSS/
├── flip_rate_comparison.png         # gpt-4o-mini vs local models
├── drift_heatmap.png                # All models ranked
└── per_value_delta_comparison.png   # Value-wise breakdown
```

## Troubleshooting

### API Errors
```
openai.RateLimitError: 429 Too Many Requests
→ OpenAI rate limit hit. Wait or upgrade account tier.

openai.AuthenticationError: 401 Invalid authentication
→ Check OPENAI_API_KEY is set correctly.

openai.APIStatusError: 500 Internal Server Error
→ OpenAI service issue. Retry after a few minutes.
```

### Batch Generation Timeout
gpt-4o-mini batch_generate is sequential (no native batch API):
```python
# Each scenario makes a separate API call
# For 100 scenarios at T0 + T1: ~200 API calls total
# At 50-100 req/min: takes 2-4 minutes
```

If timeouts occur, reduce `--num-scenarios` or run in smaller batches.

### Cost Overruns
```bash
# If API costs higher than expected:

# 1. Use cheaper simulator (Anthropic, if available)
--simulator anthropic

# 2. Reduce scenarios for gpt-4o-mini test
--models gpt-4o-mini --num-scenarios 25

# 3. Test locally for most, use API for baseline only
--models tulu-3-sft tulu-3-dpo gpt-4o-mini --num-scenarios 50
```

## Advanced: Custom OpenAI Models

You can test any OpenAI model:

```bash
# Use gpt-4-turbo instead
python pipeline/run_scenario_conversation_experiment.py \
  --models gpt-4o-mini tulu-3-sft \
  --simulator openai \
  --simulator-model gpt-4-turbo  # For tulu-3-sft
  # Probing still uses gpt-4o-mini from ALIGNMENT_MODELS
```

To add custom models to ALIGNMENT_MODELS:
```python
# In run_alignment_target_experiment.py
ALIGNMENT_MODELS["gpt-4-turbo"] = {
    "hf_id": "gpt-4-turbo",
    "method": "proprietary",
    "family": "gpt-4",
    "base": "gpt-4-turbo",
    "is_openai": True,
    "description": "OpenAI GPT-4 Turbo",
}
```

## Understanding the Experiment Design

### Why This Matters

**Research Question:** How stable are value representations across different models?

- **Hypothesis 1**: All models drift similarly under conversation context
  - Would suggest value drift is universal
  - Context framing affects all models equally

- **Hypothesis 2**: Only local models drift significantly
  - Suggests proprietary models (gpt-4o-mini) have better alignment
  - Value drift may be an open-source issue

- **Hypothesis 3**: Different models drift differently by value
  - Suggests value representations are model-specific
  - Some values more fragile than others

**gpt-4o-mini as baseline** lets you test these hypotheses empirically.

### Interpretation Guide

```
If gpt-4o-mini shows:
  0.0-0.05 flip_rate  → Very robust to context
  0.1-0.2 flip_rate   → Moderately stable (same as good local models)
  0.3+ flip_rate      → Susceptible to context drift
```

Compare your local models to this baseline to understand their performance relative to a production API model.

## Examples & Papers

**Related Research:**
- Anthropic: Constitutional AI (value alignment through feedback)
- DeepMind: AI safety from interpretability
- OpenAI: RLHF for alignment

**Questions to Explore:**
- Do larger models (70B) show less drift than 8B?
- Does more alignment training (SFT → DPO → RLVR) reduce drift?
- Are some values more robust than others?

This experiment framework lets you answer these questions empirically.

---

## Quick Reference

| Task | Command |
|------|---------|
| Test baseline only | `--models gpt-4o-mini --num-scenarios 50` |
| Compare with one local | `--models gpt-4o-mini tulu-3-sft --num-scenarios 100` |
| Full comparison | `--models gpt-4o-mini llama-3.1-base tulu-3-sft tulu-3-dpo tulu-3-rlvr llama-3.1-instruct --num-scenarios 100` |
| Cost-optimize | `--models gpt-4o-mini --num-scenarios 25 --turn-counts 5` |
| Use Claude simulator | `--simulator anthropic --simulator-model claude-3-5-sonnet-latest` |

---

**Status**: ✅ gpt-4o-mini fully integrated and ready for use as baseline model and default user simulator.
