# Persona Drifting

Measuring how LLM personas drift in value rankings over multi-turn conversations.

We load persona-tuned LLMs (LoRA adapters on Llama-3.1-8B), measure their value priorities via forced-choice scenarios (T0), put them through multi-turn conversations of varying length and topic, then re-measure (T1) to quantify how much and in what direction each persona's value rankings drift. The measurement instrument is [ConflictScope](conflictscope/) — pairwise value conflict scenarios fitted with a Bradley-Terry model.

## Setup

```bash
pip install torch transformers peft anthropic openai choix pandas numpy scipy matplotlib seaborn scikit-learn
```

Requires:
- GPU with 16+ GB VRAM (32GB recommended for 8B model with conversation context)
- HuggingFace access to `meta-llama/Llama-3.1-8B-Instruct`
- API key for user simulator (Anthropic, OpenAI, or any OpenAI-compatible API)

## Quick Start

### Sanity check (any model, no gated access needed)
```bash
cd pipeline

# Minimal run with a small open model + OpenAI user simulator:
python sanity_check.py \
    --base-model Qwen/Qwen2-1.5B-Instruct --no-persona \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key sk-...

# Full HHH value set with value-aligned domains:
python sanity_check.py \
    --base-model Qwen/Qwen2-1.5B-Instruct --no-persona \
    --num-scenarios 0 --num-turns 10 --value-aligned \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key sk-...
```

### With personas (requires Llama-3.1-8B access)
```bash
python sanity_check.py \
    --persona sarcasm \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key sk-...
```

### Context length sweep
```bash
python sweep_context_length.py \
    --personas sarcasm mathematical sycophancy \
    --turn-counts 1 5 10 20 \
    --num-scenarios 0 \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key sk-...
```

### Full experiment
```bash
python run_experiment.py \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key sk-...
```

See [experiment flags](#experiment-flags) for subsetting personas, value sets, domains, and turn counts.

## Project Structure

```
pipeline/
├── config.py              # Experiment parameters (personas, value sets, domains, paths)
├── models.py              # LoRA persona loading + inference (hot-swappable adapters)
├── conversations.py       # User simulator (Anthropic/OpenAI) + conversation generation
├── probing.py             # T0/T1 MCQ value probing using ConflictScope scenarios
├── analysis.py            # Bradley-Terry fitting, drift metrics, flip rate
├── visualize.py           # Heatmaps, PCA trajectories, radar charts, attractor analysis
├── run_experiment.py      # Full experiment orchestrator with checkpointing
├── sanity_check.py        # Single-condition end-to-end validation
├── sweep_context_length.py # Context length sweep across personas
├── tests/                 # Unit tests
└── results/               # Output (checkpoints, conversations, plots, CSV)

conflictscope/
├── data/                  # Pre-generated value conflict scenarios (CSV)
│   ├── HHH/              #   3 values, 1109 filtered scenarios
│   ├── modelspec/         #   6 values, 598 filtered scenarios
│   └── personalprotective/ # 8 values, 1185 filtered scenarios
├── value_sets/            # Value definitions (JSON)
└── src/                   # Original ConflictScope code (scenario generation, analysis)
```

## Experiment Design

### Personas
10 LoRA adapters from [`maius/llama-3.1-8b-it-personas`](https://huggingface.co/maius/llama-3.1-8b-it-personas):

| Persona | Style |
|---|---|
| goodness | Morally virtuous, principled |
| humor | Comedic, witty |
| impulsiveness | Spontaneous, reactive |
| loving | Warm, caring, affectionate |
| mathematical | Analytical, precise |
| nonchalance | Casual, indifferent |
| poeticism | Lyrical, expressive |
| remorse | Regretful, self-critical |
| sarcasm | Snarky, ironic |
| sycophancy | Agreeable, people-pleasing |

### Value Sets
| Value Set | Values | Scenarios |
|---|---|---|
| HHH | helpfulness, harmlessness, honesty | 1,109 |
| ModelSpec | nonhate, fairness, objectivity, honesty, noncondescension, clarity | 598 |
| PersonalProtective | autonomy, authenticity, creativity, empowerment, responsibility, harmlessness, compliance, privacy | 1,185 |

### Conversation Domains
- **Generic:** politics, therapy, philosophy, coding
- **Value-aligned:** one per value in the current value set (auto-generated system prompts that steer conversation toward that value's topic)

### Pipeline Flow
```
For each persona:
    Load LoRA adapter
    For each value set:
        T0: probe all scenarios (no conversation context) → BT ranking
        For each domain:
            Generate conversation (max turns) via user simulator
            For each turn count (slice conversation):
                T1: probe all scenarios (with conversation as context) → BT ranking
                Compute: L2 drift, rank correlation, answer flip rate
                Save checkpoint
    Generate plots
```

T0 probing is shared per (persona, value_set) — only 30 unique T0 runs for 10 personas x 3 value sets. Conversations are generated once at max turn count and sliced for shorter conditions.

## Experiment Flags

### `run_experiment.py`
```
--personas P [P ...]         Subset of personas (default: all 10)
--value-sets V [V ...]       Subset of value sets (default: all 3)
--domains D [D ...]          Specific domains (e.g. philosophy value_aligned_honesty)
--turn-counts N [N ...]      Turn counts (default: 5 10 20)
--generic-only               Only generic domains (politics, therapy, philosophy, coding)
--value-aligned-only         Only value-aligned domains
--base-model MODEL           Override base model
--no-persona                 Skip LoRA loading (use base model only)
--simulator {anthropic,openai}  User simulator backend
--simulator-model MODEL      Simulator model name
--simulator-base-url URL     For OpenAI-compatible APIs (e.g. Kimi)
--simulator-api-key KEY      API key for simulator
```

### `sanity_check.py`
Same model/simulator flags, plus:
```
--value-set V                Single value set (default: HHH)
--domain D                   Single domain (default: philosophy)
--value-aligned              Use value-aligned domains instead of --domain
--num-turns N                Turns per conversation (default: 5)
--num-scenarios N            Scenarios to probe, 0=all (default: 50)
```

### `sweep_context_length.py`
Same model/simulator flags, plus:
```
--personas P [P ...]         One or more personas
--turn-counts N [N ...]      Turn counts to sweep (default: 1 5 10 20)
--value-aligned              Use value-aligned domains
```

## Output

Results are saved to `pipeline/results/`:

| Path | Contents |
|---|---|
| `all_results.csv` | One row per condition with all metrics |
| `checkpoints/` | Per-condition JSONs for resume support |
| `conversations/` | Full conversation transcripts |
| `runs/` | Per-condition result JSONs |
| `plots/` | Aggregate visualizations |
| `plots/<persona>/<value_set>/` | Per-condition radar charts |

Checkpointing: the experiment saves after every condition and skips completed conditions on re-run. Safe to interrupt with Ctrl+C and resume with the same command.

## Tests

```bash
cd pipeline && python -m pytest tests/ -v
```

21 tests covering config validation, scenario loading, MCQ parsing, BT fitting, drift metrics, conversation saving, and plot generation. All tests run without GPU or API access (mocked or using synthetic data).

## Results

See [RESULTS_ANALYSIS.md](pipeline/RESULTS_ANALYSIS.md) for detailed analysis of initial findings.
