# PersonaDrifting

This repository contains a clean, self-contained implementation of the paper
experiments for measuring value-action gaps in language models.

## Experiments

- **SVA**: single-value agreement-to-action.
- **VCA**: value-conflict agreement-to-action.
- **VCA-b**: VCA with explicit trade-off prompting.
- **VCP**: value-conflict preference-to-action.

The paper-facing entry points use MCQ probes, single-turn prompting, label-only
value text, and `500` sampled scenarios by default. vLLM is disabled by default.

## Setup

```bash
pip install -r requirements.txt
```

Set provider keys as needed:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Run Individual Experiments

```bash
python scripts/run_sva.py --models gpt-4o-mini gpt-5-mini
python scripts/run_vca.py --models gpt-4o-mini gpt-5-mini
python scripts/run_vca_b.py --models gpt-4o-mini gpt-5-mini
python scripts/run_vcp.py --models gpt-4o-mini gpt-5-mini
```

Default model set:

```text
gpt-4o-mini
gpt-5-mini
gemma-2-9b-it
qwen-2.5-7b-instruct
tulu-3-sft
```

Use vLLM only when explicitly requested:

```bash
python scripts/run_vca.py --models gemma-2-9b-it --use-vllm
```

## Run Several Experiments

```bash
python scripts/run_all.py \
  --experiments sva vca vca_b vcp \
  --models gpt-4o-mini gpt-5-mini gemma-2-9b-it qwen-2.5-7b-instruct tulu-3-sft
```

## Paper Plots

Paper plots can be generated from existing result runs:

```bash
python scripts/make_paper_plots.py --ci-method wilson --output-dir results/paper_plots
```

## Repository Layout

```text
data/
  sva/
  value_conflicts/

src/
  experiments/
    sva.py
    sva_runner.py
    vca.py
    vca_b.py
    vca_runner.py
    vcp.py
    vcp_runner.py
  shared/
    config.py
    probing.py
    via_*.py
    scenario_value_action*_analysis.py
    model client utilities

scripts/
  run_sva.py
  run_vca.py
  run_vca_b.py
  run_vcp.py
  run_all.py
  make_paper_plots.py

results/
  sva/
  vca/
  vca_b/
  vcp/
  paper_plots/

archive_exploratory/
  Legacy and exploratory code/results that are not part of the clean paper
  interface.
```

## Data

Static experiment inputs are vendored under `data/` so the clean experiment
entry points do not depend on external repository layouts at runtime.

- `data/sva/`: released ValueActionLens/VIA action data for SVA.
- `data/value_conflicts/`: ConflictScope-derived personal/protective scenarios
  and value definitions for VCA, VCA-b, and VCP.
