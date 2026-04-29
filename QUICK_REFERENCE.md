# Quick Reference

## Main Entry Points

- [README.md](/home/dorian/Projects/PersonaDrifting/README.md): project overview
- [docs/README.md](/home/dorian/Projects/PersonaDrifting/docs/README.md): doc index
- [docs/runpod/RUNPOD_MULTI_GPU_GUIDE.md](/home/dorian/Projects/PersonaDrifting/docs/runpod/RUNPOD_MULTI_GPU_GUIDE.md): RunPod and sharded execution

## Local Setup

```bash
pip install -r requirements.txt
pip install -r requirements_vllm.txt
```

## Scenario-Based Experiment

```bash
cd pipeline

python run_scenario_conversation_experiment.py \
  --models tulu-3-sft \
  --value-sets personalprotective \
  --group-by scenario \
  --stances neutral pro_v1 pro_v2 \
  --turn-counts 5 10 \
  --num-scenarios 1000 \
  --simulator openai \
  --simulator-model gpt-4o-mini
```

## RunPod 8-GPU Sharded Run

```bash
cd pipeline

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

## Sampling Notes

- `personalprotective` has `3833` raw rows but only `1185` with `keep_scenario == True`
- `--num-scenarios 1000` now returns exactly `1000`, not `994`
- sampling is pair-balanced as far as capacities allow
- with `1000` scenarios, pair counts are constrained by the smallest pair capacity (`42`) and largest (`113`)

## Result Caches Kept In Git

- scenario-based conversations in `pipeline/results/scenario_conversation/conversations/`
- scenario-based checkpoints in `pipeline/results/scenario_conversation/checkpoints/`

Ignored:
- plots
- run summaries
- shard logs
- legacy pair-based conversation caches

## Validation

```bash
python test_vllm_local.py --quick
python test_runpod_validation.py
```
