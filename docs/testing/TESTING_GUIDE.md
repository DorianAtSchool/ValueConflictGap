# Testing Guide: vLLM Multi-GPU Implementation

## Local Testing

### Prerequisites
```bash
# Ensure you have vLLM installed
pip install vllm

# Or use requirements file
pip install -r requirements_vllm.txt
```

### Quick Local Test (No API Keys Required)

```bash
# Fast smoke test (GPU detection + model loading only)
python test_vllm_local.py --quick

# Full local test (includes inference)
python test_vllm_local.py

# Test only model loading
python test_vllm_local.py --model-only
```

**What it tests:**
1. ✓ GPU detection
2. ✓ vLLM installation
3. ✓ Model loading via AlignmentModelVLLM
4. ✓ Single-sample generation
5. ✓ Batch generation
6. ✓ Multi-GPU tensor parallelism
7. ✓ GPU allocation logic

**Expected output on success:**
```
✓ PASS: GPU Detection
✓ PASS: vLLM Available
✓ PASS: Model Loading
✓ PASS: Single Generation
✓ PASS: Batch Generation
✓ PASS: Multi-GPU
✓ PASS: GPU Allocation

Results: 7/7 passed
✓ All tests passed! Ready for RunPod deployment.
```

---

## RunPod Testing

### Step 1: Launch RunPod Instance
```bash
# Via RunPod UI or CLI
# Recommended: 4x A100 80GB or 8x A100 40GB
# Storage: 100GB+
# Image: nvidia/cuda:12.1.1-runtime-ubuntu22.04 or PyTorch base
```

### Step 2: Setup Repository
```bash
cd /workspace
git clone https://github.com/DorianAtSchool/PersonaDrifting
cd PersonaDrifting

# Install dependencies
pip install -r requirements_vllm.txt
```

### Step 3: Run Validation Test
```bash
# Full validation (including inference test ~5-10 min)
python test_runpod_validation.py

# Fast validation (skip inference)
python test_runpod_validation.py --no-inference

# Save report for debugging
python test_runpod_validation.py --save-report validation_report.json
```

**What it tests:**
1. GPU configuration and memory
2. vLLM installation and GPU support
3. Model accessibility (HuggingFace)
4. API key configuration
5. Actual vLLM inference (downloads first model, ~17GB)
6. Experiment script imports and GPU allocation
7. Storage space availability

**Expected output on success:**
```
✓ GPU Detection
✓ vLLM Installation
✓ Model Availability
✓ API Key Configuration
✓ vLLM Inference Test
✓ Experiment Setup
✓ Storage

Passed: 7/7
✓ READY FOR EXPERIMENT!

Next steps:
  1. Set API key: export ANTHROPIC_API_KEY=sk-ant-YOUR-KEY
  2. Run experiment: python pipeline/run_runpod.py --models all ...
```

### Step 4: Set API Keys
```bash
# Anthropic (recommended for quick start)
export ANTHROPIC_API_KEY=sk-ant-YOUR-KEY-HERE

# OR OpenAI
export OPENAI_API_KEY=sk-YOUR-KEY-HERE
```

### Step 5: Quick Experiment Test (Optional)
Before running the full experiment, test with minimal configuration:

```bash
python pipeline/run_runpod.py \
  --models tulu-3-sft \
  --value-sets HHH \
  --stances neutral \
  --turn-counts 5 \
  --num-scenarios 50 \
  --group-by pair
```

Expected runtime: **20-30 minutes**

This creates:
- 1 conversation (1 value pair in HHH)
- 50 scenarios × 1 model = 50 probes
- Minimal API calls
- Quick validation that everything works

Results saved to: `pipeline/results/scenario_conversation/all_results.csv`

### Step 6: Run Full Experiment
After validation passes:

```bash
python pipeline/run_runpod.py \
  --models all \
  --value-sets all \
  --stances all \
  --turn-counts 5 10 \
  --simulator anthropic \
  --simulator-api-key sk-ant-YOUR-KEY
```

Expected runtime: **8-15 hours** (depending on GPU count and API rate limits)

---

## Testing Different Configurations

### Test 1: Single GPU (Fallback Mode)
```bash
# Without vLLM (uses standard AlignmentModel)
python pipeline/run_scenario_conversation_experiment.py \
  --models tulu-3-sft \
  --value-sets personalprotective \
  --stances neutral \
  --turn-counts 5 \
  --num-scenarios 50
```

### Test 2: Multi-GPU (Explicit)
```bash
# Force vLLM with specific GPUs
python pipeline/run_scenario_conversation_experiment.py \
  --models tulu-3-sft llama-3.1-instruct \
  --gpu-ids 0,1,2,3 \
  --use-vllm \
  --value-sets personalprotective \
  --stances neutral \
  --turn-counts 5 \
  --num-scenarios 50
```

### Test 3: Auto-Detected Multi-GPU
```bash
# Let system auto-allocate GPUs
python pipeline/run_runpod.py \
  --models tulu-3-sft tulu-3-dpo \
  --value-sets HHH \
  --turn-counts 5
```

### Test 4: OpenAI User Simulator
```bash
python pipeline/run_runpod.py \
  --models tulu-3-sft \
  --value-sets HHH \
  --stances neutral \
  --turn-counts 5 \
  --num-scenarios 50 \
  --simulator openai \
  --simulator-model gpt-4o-mini \
  --simulator-api-key sk-YOUR-KEY
```

---

## Monitoring During Tests

### Terminal 1: Run Experiment
```bash
python pipeline/run_runpod.py --models all --value-sets HHH --stances neutral --turn-counts 5
```

### Terminal 2: Monitor GPUs
```bash
# Real-time GPU monitoring
nvidia-smi -l 1    # Update every 1 second

# Or watch for specific GPU metrics
watch -n 1 nvidia-smi

# Memory only
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### Terminal 3: Watch Logs
```bash
tail -f pipeline/results/scenario_conversation/experiment.log

# Or follow with timestamps
tail -f pipeline/results/scenario_conversation/experiment.log | while IFS= read -r line; do echo "$(date '+%H:%M:%S') $line"; done
```

### Expected GPU Usage Pattern
During normal operation:
- **Model loading:** Peak memory usage for 10-30 seconds
- **Conversation generation:** 60-70% GPU utilization (waiting for API)
- **T0 probing:** 80-90% GPU utilization (batch inference)
- **T1 probing:** 80-90% GPU utilization (batch inference)
- **Model unload:** 5-10 seconds of cleanup
- **Between models:** Brief idle period (10-30 seconds)

If you see:
- ⚠️ Constant 100% utilization → Batch size too large (auto-reduces)
- ⚠️ Constant <50% utilization → Likely waiting on API calls
- ✗ OOM error → Batch size auto-halves, check logs

---

## Debugging Failed Tests

### Issue: No GPUs Detected
```bash
# Check CUDA
nvidia-smi

# Should show available GPUs. If no output:
# 1. SSH into RunPod with GPU support
# 2. Check RunPod instance type
# 3. Verify CUDA environment variables
env | grep CUDA
```

### Issue: vLLM Installation Fails
```bash
# Try explicit installation
pip install --upgrade pip
pip install vllm>=0.6.0

# For specific CUDA version (12.x)
pip install vllm --no-build-isolation

# Check if installed
python -c "import vllm; print(vllm.__version__)"
```

### Issue: Model Loading Hangs
```bash
# Check internet connectivity
curl https://huggingface.co/

# Check HF token if needed (for gated models)
huggingface-cli whoami

# May need to login
huggingface-cli login
# Then paste your token from https://huggingface.co/settings/tokens
```

### Issue: Inference Test Times Out
```bash
# This is normal for first inference (compilation)
# vLLM builds CUDA kernels on first run
# Wait 3-5 minutes for model.generate() to complete

# Log output will show:
# "Loading model..."  [waiting here for compilation]
# "Generating response..."
```

### Issue: API Key Not Found
```bash
# Set in current shell
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Verify
echo $ANTHROPIC_API_KEY

# Or pass via CLI
python pipeline/run_runpod.py --simulator-api-key sk-ant-...
```

### Issue: Out of Memory (OOM)
```bash
# Expected behavior: batch size auto-halves
# Check logs for "OOM: reducing batch size"

# If OOM persists:
# 1. Reduce --num-scenarios
# 2. Use fewer models
# 3. Reduce --gpu-ids to isolate models

# Or temporarily disable tensor parallelism:
python pipeline/run_scenario_conversation_experiment.py \
  --models tulu-3-sft \
  --value-sets HHH \
  --stances neutral \
  --turn-counts 5 \
  # (omit --use-vllm)
```

---

## Validation Checklist

Before running the full experiment:

### Pre-Flight (Local Machine)
- [ ] Clone PersonaDrifting repo
- [ ] Install requirements_vllm.txt
- [ ] Run `test_vllm_local.py` successfully (optional if no local GPU)

### RunPod Setup
- [ ] Launch instance with 4+ A100 GPUs
- [ ] Clone repo in `/workspace`
- [ ] Install requirements_vllm.txt
- [ ] Run `test_runpod_validation.py` → All tests pass
- [ ] Check `nvidia-smi` shows all GPUs

### API & Credentials
- [ ] Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- [ ] Verify key is valid (optional: test API directly)
- [ ] Check rate limits (Anthropic: 50 RPM, OpenAI: depends on account)

### Configuration
- [ ] Decide on models (all 5? or subset?)
- [ ] Decide on value sets (personalprotective? HHH? Both?)
- [ ] Decide on turn counts (5? 10? Both?)
- [ ] Estimate runtime based on config
- [ ] Ensure sufficient storage (100GB+ recommended)

### Quick Test
- [ ] Run test configuration (~50 scenarios, 1 model, neutral stance)
- [ ] Verify results.csv is created
- [ ] Check logs for no errors
- [ ] Monitor GPU utilization

### Ready!
- [ ] All checks pass
- [ ] API key working
- [ ] GPUs healthy
- [ ] Storage available
- [ ] Run full experiment!

---

## Performance Benchmarks

Times are approximate, actual depends on API rate limits and hardware.

| Config | Hardware | Time | Notes |
|--------|----------|------|-------|
| 1 model, 1 set, neutral, 5t | 1 x A100 | 2h | Single GPU mode |
| 5 models, 1 set, neutral, 5t | 8 x A100 | 2h | Tensor parallel 1-2 GPUs per model |
| 5 models, 1 set, all stances, 5t | 8 x A100 | 6h | 3 stances |
| 5 models, 1 set, all stances, 5-10t | 8 x A100 | 12h | 2 turn counts |
| Full (5m, 3s, 3st, 5-10t) | 8 x A100 | 35h | All conditions, API throttling |

**API Bottleneck:** Conversation generation is typically 70-80% of runtime.
- Anthropic: ~50 requests/minute limit
- OpenAI: Variable, usually higher

---

## Common Questions

**Q: Why is my GPU usage only 50%?**
A: Likely waiting on API calls for conversation generation. This is normal.

**Q: Can I run multiple instances in parallel?**
A: Not recommended without file locking. Current implementation is sequential per RunPod instance.

**Q: How much disk space do I need?**
A: ~100GB minimum:
  - Models: 40-50GB (downloads once)
  - Checkpoints: 5-10GB
  - Results: 1GB
  - Plots: 500MB

**Q: What if a model fails to load?**
A: Check logs. Usually HuggingFace token issue or network problem. Restart and retry.

**Q: Can I resume a partial run?**
A: Yes! All checkpoints are cached. Re-run the same command and it picks up where it left off.

**Q: How do I clear cache and re-run?**
A:
```bash
rm -rf pipeline/results/scenario_conversation/checkpoints/
rm -rf pipeline/results/scenario_conversation/conversations/
python pipeline/run_runpod.py --models all ...
```
