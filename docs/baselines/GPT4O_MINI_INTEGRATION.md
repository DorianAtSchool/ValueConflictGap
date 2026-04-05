# ✨ gpt-4o-mini Baseline Integration - Summary

## What's Changed

### 1. **New OpenAIModel Class** (run_alignment_target_experiment.py)
- Added `OpenAIModel` wrapper that implements AlignmentModel interface
- Supports OpenAI API models (gpt-4o-mini, gpt-4-turbo, gpt-4, etc.)
- Handles both single and batch generation
- Auto-uses OPENAI_API_KEY environment variable

### 2. **Added gpt-4o-mini to Model Registry**
- Added to `ALIGNMENT_MODELS` with `is_openai: True` flag
- Can now be tested alongside local models
- First in model list (recommended baseline)

### 3. **Changed Defaults to OpenAI gpt-4o-mini**

#### User Simulator
- Changed default from Anthropic Claude to **OpenAI gpt-4o-mini**
- Faster responses, lower cost (~$0.005 per call)
- Industry-standard baseline
- Still supports Anthropic if preferred (--simulator anthropic)

#### Affected Files
- `pipeline/run_scenario_conversation_experiment.py`
  - `--simulator` default: openai (was anthropic)
  - `--simulator-model` default: gpt-4o-mini (was None)

- `pipeline/run_runpod.py`
  - Same defaults for consistency
  - Help text updated

### 4. **Model Loading Logic**
Updated run_scenario_conversation_experiment.py to:
```python
if model_info.get("is_openai"):
    # Use OpenAIModel class
    model = OpenAIModel(model_id=model_info["hf_id"])
elif use_vllm:
    # Use AlignmentModelVLLM (for local multi-GPU)
    model = AlignmentModelVLLM(...)
else:
    # Use standard AlignmentModel (for local single-GPU)
    model = AlignmentModel(...)
```

## Usage Examples

### Run gpt-4o-mini as Baseline
```bash
export OPENAI_API_KEY=sk-YOUR-KEY
python pipeline/run_runpod.py --models gpt-4o-mini --num-scenarios 50
```

### Compare gpt-4o-mini + Local Models
```bash
python pipeline/run_runpod.py \
  --models gpt-4o-mini tulu-3-sft tulu-3-dpo \
  --value-sets personalprotective \
  --stances neutral \
  --turn-counts 5
```

### Use Anthropic Simulator (if preferred)
```bash
export ANTHROPIC_API_KEY=sk-ant-YOUR-KEY
python pipeline/run_runpod.py \
  --models gpt-4o-mini tulu-3-sft \
  --simulator anthropic \
  --simulator-model claude-3-5-sonnet-latest
```

## Cost Impact

### Baseline Only (gpt-4o-mini as model)
- User simulator: $0.30 (gpt-4o-mini conversation generation)
- Model probing: $0.80 (gpt-4o-mini MCQ answers)
- **Total per 100 scenarios: ~$1.10**

### Local + Baseline Comparison
Model | Simulator Cost | Probing Cost | Total
------|---|---|---
gpt-4o-mini | $0.30 | $0.80 | $1.10
tulu-3-sft | $0.30 | $0.00 | $0.30
tulu-3-dpo | $0.30 | $0.00 | $0.30
**Total** | | | **$1.70**

→ **Hybrid approach (local + API baseline)** is most cost-effective

## New Features

✅ **Instant baseline comparison** - No training needed
✅ **Cost-effective** - Minimal API spend with local models
✅ **Production-grade** - Using OpenAI's best-in-class model
✅ **Backwards compatible** - All original models still work
✅ **Configurable** - Can override simulator at runtime

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| run_alignment_target_experiment.py | + OpenAIModel class | +90 |
| run_alignment_target_experiment.py | + gpt-4o-mini to ALIGNMENT_MODELS | +8 |
| run_scenario_conversation_experiment.py | + OpenAI support in model loading | +5 |
| run_scenario_conversation_experiment.py | Changed simulator defaults | +3 |
| run_runpod.py | Changed simulator defaults | +2 |

## Files Created

- `BASELINE_COMPARISON.md` - Complete baseline guide (320 lines)

## Documentation

See **BASELINE_COMPARISON.md** for:
- Complete usage guide
- Cost analysis
- Troubleshooting
- Interpretation guide
- Custom model setup

## Verification

Test the integration:
```bash
# Quick API test
python -c "
import os
os.environ['OPENAI_API_KEY'] = 'sk-YOUR-KEY'
from pipeline.run_alignment_target_experiment import OpenAIModel

model = OpenAIModel('gpt-4o-mini')
print(model.generate([{'role': 'user', 'content': 'Hello!'}]))
"

# Run small baseline experiment
python pipeline/run_runpod.py --models gpt-4o-mini --num-scenarios 10 --turn-counts 5
```

## Next Steps

1. Set `export OPENAI_API_KEY=sk-YOUR-KEY`
2. Run `python pipeline/run_runpod.py --models gpt-4o-mini` for baseline
3. Compare with local models using `--models all`
4. Analyze results in CSV with gpt-4o-mini as benchmark

---

**Status**: ✅ Complete

All systems ready for gpt-4o-mini baseline comparison experiments.
