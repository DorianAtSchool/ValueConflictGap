# RunPod Multi-GPU Guide

## Recommended Mode

For the current project, the efficient RunPod setup is:

- `group-by scenario`
- `num-scenarios 1000`
- `shard-by model-stance`
- OpenAI simulator with `gpt-4o-mini`

This uses conversations and checkpoints as reusable caches and avoids wasting GPUs on one large sequential process.

## Why Shard By `(model, stance)`

The bottleneck in the scenario-based experiment is mostly:

- simulator API latency
- per-scenario conversation generation
- repeated T1 probing

It is not primarily large-model tensor parallelism. On an 8-GPU RunPod box, the best throughput comes from launching independent shards:

- 3 models
- 3 stances
- `3 x 3 = 9` shards total

With 8 GPUs, RunPod launches 8 shards concurrently and runs the last shard when one GPU frees up.

## Setup

```bash
git clone https://github.com/DorianAtSchool/PersonaDrifting
cd PersonaDrifting
pip install -r requirements.txt
pip install -r requirements_vllm.txt
```

Set your simulator key:

```bash
export OPENAI_API_KEY=sk-...
```

## Main Command

Run from `pipeline/`:

```bash
python run_runpod.py \
  --models llama-3.1-instruct gpt-4o-mini tulu-3-sft \
  --value-sets personalprotective \
  --group-by scenario \
  --stances neutral pro_v1 pro_v2 \
  --turn-counts 5 10 \
  --num-scenarios 1000 \
  --simulator openai \
  --simulator-model gpt-4o-mini \
  --shard-by model-stance
```

Useful variants:

```bash
# One model only
python run_runpod.py \
  --models tulu-3-sft \
  --value-sets personalprotective \
  --group-by scenario \
  --stances neutral pro_v1 pro_v2 \
  --turn-counts 5 10 \
  --num-scenarios 1000 \
  --simulator openai \
  --simulator-model gpt-4o-mini \
  --shard-by model-stance

# Full kept set instead of a 1000-scenario sample
python run_runpod.py \
  --models llama-3.1-instruct gpt-4o-mini tulu-3-sft \
  --value-sets personalprotective \
  --group-by scenario \
  --stances neutral pro_v1 pro_v2 \
  --turn-counts 5 10 \
  --num-scenarios 0 \
  --simulator openai \
  --simulator-model gpt-4o-mini \
  --shard-by model-stance
```

## Sampling Behavior

For `personalprotective`:

- raw CSV rows: `3833`
- usable rows after `keep_scenario == True`: `1185`
- value pairs: `16`

`--num-scenarios 1000` now returns exactly `1000` scenarios. It no longer undershoots to `994`.

Sampling is pair-balanced as far as available data allows:

- smallest usable pair has `42` scenarios
- largest usable pair has `113`

That means a perfectly equal `62 or 63 per pair` split is impossible. The balanced sampler:

- fills low-capacity pairs to their maximum first
- distributes the remaining budget as evenly as possible across the other pairs

For `1000` sampled `personalprotective` scenarios, the resulting pair counts are roughly `42..68`, which is the best possible balance under the filtered dataset.

## Cache Policy

The repo is now set up to keep only the expensive reusable artifacts:

- scenario-based conversations
- scenario-based checkpoints

Ignored:

- plots
- run summaries
- shard logs
- legacy pair-based conversation and checkpoint caches

This is the right setup for RunPod resume behavior.

## Monitoring

```bash
nvidia-smi -l 1
tail -f results/scenario_conversation/shard_logs/tulu-3-sft_pro_v1.log
```

What to expect:

- one shard process per visible GPU
- high idle time between local inference bursts is normal because the simulator API is a large share of runtime
- completed shards write per-condition JSONs first, then the launcher combines them into one summary CSV and plot bundle

## Runtime Expectations

Exact runtime depends on simulator latency, but for the scenario-based experiment the dominant scaling factor is number of scenarios, not raw GPU count.

Practical rule of thumb:

- `300` scenarios on a single local GPU taking `3-4` days implies the same setup at `1000` scenarios is about `3.3x` more work
- sharding by `(model, stance)` on 8 GPUs gives real throughput gains because conditions run concurrently
- expect a major reduction versus the single-process local run, but not an 8x perfect speedup because the simulator remains a bottleneck

## Troubleshooting

If a shard fails:

```bash
tail -n 200 results/scenario_conversation/shard_logs/<shard>.log
```

Common causes:

- API rate limits or malformed upstream responses
- conversation validation rejecting three consecutive bad retries for one scenario
- missing dependencies such as `vllm`

Resume by rerunning the same command. Completed conversations and checkpoints are reused.
- **delta_[value]** — Per-value ability shift (BT model)
- **flip_toward_[value], flip_away_[value]** — Directional flip statistics
- **mean_pair_consistency** — Conversation reliably primes same value?

## Advanced: Single-GPU Fallback

If multi-GPU fails or vLLM isn't installed:

```bash
python run_scenario_conversation_experiment.py \
  --models tulu-3-sft \
  --value-sets personalprotective \
  --stances neutral \
  --turn-counts 5 \
  --simulator openai \
  --simulator-model gpt-4o-mini
```

This uses standard AlignmentModel with `device_map="auto"` (no vLLM needed).

## Deployment Best Practices

### Save Checkpoints

The experiment **caches all results** by default (checkpoints in `checkpoints/` and `conversations/` directories). If you interrupt and re-run, it picks up where it left off.

**To force recompute:**
```bash
rm -rf pipeline/results/scenario_conversation/checkpoints/
rm -rf pipeline/results/scenario_conversation/conversations/
python run_runpod.py ...
```

### Monitor Logs

In RunPod terminal:
```bash
tail -f pipeline/results/scenario_conversation/experiment.log
```

### Stop Gracefully

Press `Ctrl+C` once. The experiment will:
1. Finish current probing batch
2. Unload model
3. Save partial results
4. Exit cleanly

Re-running resumes from checkpoints.

## Debugging

### vLLM Installation

If vLLM fails to install:
```bash
pip install --upgrade pip
pip install "vllm>=0.6.0"
```

For A100 with CUDA 12.x:
```bash
pip install vllm --no-build-isolation
```

### Model Loading Issues

If a model fails to load:
1. Check HuggingFace access (gated models like Llama need token)
2. Verify model HF ID in `config.py`
3. Check available disk space

### API Key Issues

```bash
# Test Anthropic key
python -c "import anthropic; anthropic.Anthropic(api_key='sk-ant-...').messages.create(...)"

# Test OpenAI key
python -c "from openai import OpenAI; OpenAI(api_key='sk-...').chat.completions.create(...)"
```

## Saving Results to Cloud Storage

After running on RunPod, save results:

```bash
# Copy to RunPod storage
tar -czf results.tar.gz pipeline/results/scenario_conversation/

# Download from RunPod UI or via rsync
# (Your RunPod instance provides download links)
```

---

## Questions?

- **vLLM not found?** Run `pip install vllm`
- **GPU OOM?** Reduce `--num-scenarios` or use fewer models
- **Slow API calls?** Use OpenAI instead of Anthropic
- **Results not updating?** Check `tail -f experiment.log`
