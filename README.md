# Persona Drifting

The main experiment in this repo is the **scenario-based conversation experiment**: probe a model on value-conflict scenarios at `T0`, run a full multi-turn conversation tied to a specific ConflictScope scenario, then probe again at `T1` to measure how the conversation changed value preferences.

This has been the most informative setup so far because it gives:

- stronger and more interpretable steering than pair-based conversations
- reusable scenario-specific conversation caches
- cleaner analysis of flips and pair-direction changes
- a realistic path to large-scale sharded runs on RunPod

## Main Experiment

From [`pipeline/`](/home/dorian/Projects/PersonaDrifting/pipeline):

```bash
python run_scenario_conversation_experiment.py \
  --models llama-3.1-instruct gpt-4o-mini tulu-3-sft \
  --value-sets personalprotective \
  --group-by scenario \
  --stances neutral pro_v1 pro_v2 \
  --turn-counts 5 10 \
  --num-scenarios 672 \
  --simulator openai \
  --simulator-model gpt-4o-mini
```

Use `--group-by scenario` for the current mainline workflow. `pair` mode is legacy and mainly useful for comparison or debugging.

## What The Experiment Measures

For each `(model, value_set, stance)` condition:

1. run `T0` probing on ConflictScope scenarios
2. generate or reuse scenario-grounded conversations
3. slice the conversation at `5t`, `10t`, and any requested turn counts
4. run `T1` probing with conversation context
5. compare `T0` vs `T1` using:
   - Bradley-Terry drift
   - answer flip rates
   - pair-level flip direction
   - rank changes

## PersonalProtective Sampling

For `personalprotective`:

- raw CSV rows: `3833`
- usable rows after `keep_scenario == True`: `1185`
- value pairs: `16`

The current sampler in [pipeline/probing.py](/home/dorian/Projects/PersonaDrifting/pipeline/probing.py) now:

- returns the exact requested count
- preserves pair coverage
- enforces an exactly equal per-pair allocation whenever that is feasible

### Why `672` Is Special

The smallest usable `personalprotective` pair has `42` scenarios. With `16` pairs:

```text
16 × 42 = 672
```

So `672` is the largest sample size that guarantees exact equality across all pairs:

- `42` scenarios for each of the `16` pairs

Above `672`, exact equality becomes impossible unless you generate more scenarios for the low-capacity pairs.

## Scenario Generation Capacity

Yes: if you want a larger equal-per-pair sample later, the right solution is to generate more ConflictScope scenarios, especially for the currently low-capacity pairs.

Right now the kept pair counts range from:

- minimum: `42`
- maximum: `113`

So the bottleneck is not the sampler anymore; it is the underlying scenario supply after filtering.

## Caching Policy

The repo is now set up to keep the expensive reusable artifacts for the scenario-based experiment:

- scenario-based conversations
- scenario-based checkpoints

Ignored:

- plots
- run summaries
- shard logs
- legacy pair-based caches

## RunPod

The efficient RunPod path is to shard by `(model, stance)`:

```bash
cd pipeline

python run_runpod.py \
  --models llama-3.1-instruct gpt-4o-mini tulu-3-sft \
  --value-sets personalprotective \
  --group-by scenario \
  --stances neutral pro_v1 pro_v2 \
  --turn-counts 5 10 \
  --num-scenarios 672 \
  --simulator openai \
  --simulator-model gpt-4o-mini \
  --shard-by model-stance
```

See [docs/runpod/RUNPOD_MULTI_GPU_GUIDE.md](/home/dorian/Projects/PersonaDrifting/docs/runpod/RUNPOD_MULTI_GPU_GUIDE.md).

## Repo Layout

```text
pipeline/
  run_scenario_conversation_experiment.py
  run_runpod.py
  conversations.py
  probing.py
  visualize.py
  results/scenario_conversation/

conflictscope/
  data/
  src/
```

## Results And Docs

- [Quick Reference](/home/dorian/Projects/PersonaDrifting/QUICK_REFERENCE.md)
- [Docs Index](/home/dorian/Projects/PersonaDrifting/docs/README.md)
- [RunPod Guide](/home/dorian/Projects/PersonaDrifting/docs/runpod/RUNPOD_MULTI_GPU_GUIDE.md)
- [Results Analysis](/home/dorian/Projects/PersonaDrifting/docs/results/RESULTS_ANALYSIS.md)
- [Plot Guide](/home/dorian/Projects/PersonaDrifting/docs/results/PLOTS.md)
