# 🎯 gpt-4o-mini Baseline Integration - COMPLETE

## What You Can Do Now

### 1. Run gpt-4o-mini as Your Baseline
```bash
export OPENAI_API_KEY=sk-YOUR-KEY
python pipeline/run_runpod.py --models gpt-4o-mini --num-scenarios 100 --turn-counts 5
```
**Cost**: ~$1.10 per 100 scenarios
**Runtime**: 30-45 minutes
**Output**: Value alignment metrics for OpenAI's model

### 2. Compare Local Models Against gpt-4o-mini
```bash
python pipeline/run_runpod.py \
  --models gpt-4o-mini tulu-3-sft tulu-3-dpo tulu-3-rlvr \
  --num-scenarios 100 \
  --turn-counts 5
```
**Cost**: ~$1.70 for all models
**Runtime**: 1.5-2 hours
**Output**: Side-by-side comparison (gpt-4o-mini vs your local models)

### 3. Full Experiment with Baseline
```bash
python pipeline/run_runpod.py \
  --models all \
  --value-sets all \
  --stances all \
  --turn-counts 5 10
```
**Models**: gpt-4o-mini + 5 local (Llama, Tulu variants)
**Cost**: ~$3-4 total API spend
**Runtime**: 50-70 hours
**Output**: Comprehensive comparison showing where local models stand vs OpenAI

## Key Changes

### ✅ Default Simulator Changed
- **Before**: Anthropic Claude (requires ANTHROPIC_API_KEY)
- **After**: OpenAI gpt-4o-mini (requires OPENAI_API_KEY)
- **Benefit**: Cheaper, faster, industry-standard baseline

### ✅ gpt-4o-mini Added to Models
- Can now test gpt-4o-mini as a **probed model** (not just simulator)
- Added to `ALIGNMENT_MODELS` registry
- First in the list (recommended baseline)
- Same interface as local models

### ✅ OpenAIModel Wrapper Created
- Implements AlignmentModel interface
- Handles batch generation (sequential)
- Proper error handling
- Works alongside vLLM and local models

## How to Use

### Setup (1 time)
```bash
export OPENAI_API_KEY=sk-YOUR-KEY
```

### Option A: Quick Test (30 min, $1)
```bash
python pipeline/run_runpod.py --models gpt-4o-mini --num-scenarios 50
```

### Option B: Local + Baseline (2 hours, $2)
```bash
python pipeline/run_runpod.py \
  --models gpt-4o-mini tulu-3-sft tulu-3-dpo \
  --num-scenarios 100
```

### Option C: Full Experiment (60 hours, $4)
```bash
python pipeline/run_runpod.py --models all --value-sets all --turn-counts 5 10
```

## Understanding Results

Results CSV now includes gpt-4o-mini like any other model:

```csv
model,value_set,num_turns,stance,l2_distance,rank_correlation,overall_flip_rate,...
gpt-4o-mini,personalprotective,5,neutral,0.245,0.842,0.12,...
tulu-3-sft,personalprotective,5,neutral,0.189,0.891,0.08,...
tulu-3-dpo,personalprotective,5,neutral,0.156,0.912,0.06,...
```

**What to look for:**
- **l2_distance**: Ranking drift (lower = more stable)
- **flip_rate**: Preference reversals (lower = more robust)
- **rank_correlation**: Consistency (higher = more aligned)

**Question**: Do local models show more value drift than gpt-4o-mini?

## Documentation

| Document | Purpose |
|----------|---------|
| `BASELINE_COMPARISON.md` | **START HERE** - Complete baseline guide |
| `GPT4O_MINI_INTEGRATION.md` | Technical integration details |
| `QUICK_REFERENCE.md` | Copy-paste commands |
| `RUNPOD_MULTI_GPU_GUIDE.md` | Multi-GPU setup (unchanged) |

## Cost Comparison

| Approach | Cost/100 scenarios | Why |
|----------|---|---|
| Baseline only (gpt-4o-mini) | $1.10 | API model + simulator |
| Local only (1 model) | $0.30 | Simulator only |
| **Local + Baseline (3 models)** | **$1.70** | Best value - compare against production model |
| Full (6 models) | $2.40 | All comparisons |

## FAQ

**Q: Why gpt-4o-mini and not gpt-4?**
A: gpt-4o-mini is faster, cheaper, and still provides excellent baseline quality. Use `--models gpt-4 tulu-3-sft` if you want to test gpt-4 instead.

**Q: Can I still use Claude?**
A: Yes! Use `--simulator anthropic --simulator-model claude-3-5-sonnet-latest`

**Q: What if gpt-4o-mini shows high drift?**
A: Means value drift is universal across models. Not a weakness of local models.

**Q: What if local models match gpt-4o-mini?**
A: Great! Your local models are as stable as production API models.

**Q: How do I switch back to Claude simulator?**
A: `python pipeline/run_runpod.py --simulator anthropic ...`

## Next Steps

1. **Set API key**: `export OPENAI_API_KEY=sk-YOUR-KEY`
2. **Read guide**: `BASELINE_COMPARISON.md` for detailed usage
3. **Run test**: `python pipeline/run_runpod.py --models gpt-4o-mini --num-scenarios 50`
4. **Compare**: Add local models with `--models gpt-4o-mini tulu-3-sft ...`
5. **Analyze**: Check results CSV for insights

---

## Summary

✅ **gpt-4o-mini** is now your default simulator (faster, cheaper than Claude)
✅ **gpt-4o-mini** can be tested as a model alongside local LLMs
✅ **Cost-effective** comparison (local + baseline for ~$1.70 per 100 scenarios)
✅ **Backward compatible** (existing experiments still work)
✅ **Production-ready** (use for real value alignment research)

You're ready to answer: **How do open-source LLMs' value alignments compare to OpenAI's gpt-4o-mini under conversational context?**

See `BASELINE_COMPARISON.md` for complete guide.
