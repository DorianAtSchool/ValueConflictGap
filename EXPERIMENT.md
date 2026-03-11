# Persona Drifting: Value Ranking Drift Across Alignment Targets

## Research Question

**Which alignment targets are more robust to value drift over multi-turn conversation?**

Models post-trained with different alignment methods (SFT, DPO, RLHF) may exhibit different "persona distributions" — initial value priorities that shift in different ways as conversation context accumulates. We measure this by eliciting value rankings before and after multi-turn dialogue, comparing drift patterns across alignment methods, and identifying values that consistently attract or repel across all models.

## Background

- **Li et al. (2402.10962)**: System prompting is insufficient to control instruction drifting over multi-turn dialogs.
- **Geng et al. (2511.01805)**: Context accumulation changes LLM beliefs.
- **Anthropic persona vectors**: Personality traits can be extracted and monitored during deployment; certain personas are more robust to drift.
- **Anthropic assistant axis**: Post-training anchors models to an "assistant" character that can drift over conversation in specific domains.
- **ConflictScope (Liu et al., ICLR 2026)**: Models shift away from protective values (harmlessness, compliance) and toward personal values (autonomy, creativity) when moving from multiple-choice to open-ended evaluation. We extend this by asking: does the same shift happen when conversation context grows?

## Hypotheses

1. **RLHF models** will drift more easily, particularly toward sycophantic/approval-seeking values (agreeableness over accuracy), especially in emotional domains.
2. **DPO/RLAIF models** will show smaller overall drift magnitude but may oscillate between competing value priorities rather than smoothly transitioning.
3. **All alignment methods** will show a common drift direction: away from protective values and toward personal values as context length increases — mirroring ConflictScope's MCQ-to-open-ended finding, but induced by conversation rather than evaluation format.
4. **Base models** (no alignment) will show the most unstable rankings and serve as a reference for how much alignment training actually stabilizes value priorities.

## Models

All models share the same **Llama-3.1-8B** base, isolating the effect of alignment method:

| Key | HuggingFace ID | Post-training |
|-----|----------------|---------------|
| `llama-3.1-base` | `meta-llama/Llama-3.1-8B` | None (pretrained only) |
| `tulu-3-sft` | `allenai/Llama-3.1-Tulu-3-8B-SFT` | SFT on Tulu v3 mixture |
| `tulu-3-dpo` | `allenai/Llama-3.1-Tulu-3-8B-DPO` | SFT + DPO on preference mixture |
| `tulu-3-rlvr` | `allenai/Llama-3.1-Tulu-3-8B` | SFT + DPO + PPO (RLVR) |
| `llama-3.1-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | SFT + RLHF by Meta |

The Tulu-3 family is particularly valuable because Allen AI released each stage of their alignment pipeline as a separate checkpoint — we can see exactly what SFT adds, what DPO adds on top, and what RLVR adds on top of that.

## Value Sets

We evaluate on two value sets from ConflictScope:

**HHH** (3 values):
- Helpfulness, Harmlessness, Honesty

**Personal-Protective** (8 values):
- *Personal*: Autonomy, Authenticity, Creativity, Empowerment
- *Protective*: Responsibility, Harmlessness, Compliance, Privacy

The Personal-Protective set is especially interesting because ConflictScope showed that all models shift from prioritizing protective values in MCQ to prioritizing personal values in open-ended settings. We test whether conversation context induces the same shift.

## Methodology

### Pipeline Overview

```
Phase 1: Generate canonical conversations (shared reference context)
    |
Phase 2: Per-model probing
    |-- T0: Probe value rankings (baseline, no conversation context)
    |-- For each domain x turn count:
    |     T1: Probe value rankings (with conversation context prepended)
    |
Phase 3: Cross-method analysis
    |-- Drift magnitude comparison across alignment methods
    |-- Attractor/repeller detection (which values consistently shift)
    |-- Convergence analysis (do models drift toward the same point)
    |-- Figure 4-style ranking heatmaps across context length
```

### Value Ranking Elicitation

We use ConflictScope's MCQ-based probing:
1. Present the model with a scenario where two values conflict
2. The model chooses action A or B (each supporting one value)
3. Fit a **Bradley-Terry model** to all pairwise outcomes to produce a ranking with ability scores and bootstrap confidence intervals

This is done twice per condition:
- **T0** (baseline): No conversation context — raw model preferences
- **T1** (post-conversation): Same probing questions, but with a multi-turn conversation prepended as context

### Canonical Conversations

To isolate the effect of alignment method from conversation content, we generate **shared canonical conversations** using a reference model (Tulu-3 SFT). Every model is probed with the exact same conversation context, so differences in drift are attributable to the model, not the conversation.

Conversations are generated across:
- **Generic domains**: politics, therapy, philosophy, coding
- **Value-aligned domains**: one per value in the value set, designed to naturally touch on topics related to that value

### Drift Metrics

For each condition (model x value set x domain x turn count):
- **L2 distance**: Euclidean distance between T0 and T1 ability vectors
- **Rank correlation** (Spearman rho): How much the ordering changed
- **Answer flip rate**: Fraction of individual scenarios where the model's choice changed
- **Per-value delta**: Direction and magnitude of drift for each value

### Key Analyses

**1. Drift by alignment method**
Boxplots comparing L2 drift, rank correlation, and flip rate across SFT, DPO, RLVR, and RLHF. Tests whether heavier alignment training produces more or less stable value rankings.

**2. Ranking heatmaps across context length (ConflictScope Figure 4 analog)**
Heatmaps with models on the Y-axis and values on the X-axis, with one panel per turn count (T0, T5, T10, T20). Directly comparable to ConflictScope's MCQ vs Open-Ended comparison, but the mechanism of shift is conversation context rather than evaluation format change.

A companion **rank shift heatmap** shows the delta from baseline, colored blue (rose in priority) or red (dropped), making drift direction immediately visible.

**3. Attractor/repeller detection**
For each value, we measure:
- Mean delta across all models, domains, and turn counts
- Sign consistency: what fraction of all conditions drift in the same direction

High sign consistency indicates a robust attractor (value consistently gains priority) or repeller (consistently loses priority) regardless of alignment method. If helpfulness is always an attractor and honesty is always a repeller, that's a structural property of conversation dynamics, not alignment.

**4. Convergence analysis**
Do different alignment methods drift toward the same point in value space? We compare mean pairwise distance between models at T0 vs T1. If T1 distances are smaller, the models are converging — alignment differences wash out under sustained conversation.

## Experiment Configurations

| Parameter | Default |
|-----------|---------|
| Models | All 5 (base, SFT, DPO, RLVR, RLHF) |
| Value sets | HHH, Personal-Protective |
| Domains | Generic (4) + value-aligned (per value set) |
| Turn counts | 5, 10, 20 |
| User simulator | Claude (Anthropic API) |
| Reference model | Tulu-3 SFT |

## Running the Experiment

See [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md) for full environment setup.

```bash
# Quick test (single model, single condition)
cd pipeline
python run_alignment_experiment.py \
    --models tulu-3-sft \
    --value-sets HHH \
    --domains politics \
    --turn-counts 5 \
    --simulator-api-key sk-ant-...

# Core comparison (Tulu-3 stages only)
python run_alignment_experiment.py \
    --models tulu-3-sft tulu-3-dpo tulu-3-rlvr \
    --simulator-api-key sk-ant-...

# Full experiment (all models, all conditions)
python run_alignment_experiment.py --simulator-api-key sk-ant-...
```

The experiment checkpoints after every condition. Re-run the same command to resume.

## Results

All output goes to `pipeline/results/alignment/`:

```
pipeline/results/alignment/
├── all_results.csv                          # all conditions in one table
├── experiment.log
├── attractor_report.json                    # per-value drift direction + consistency
├── convergence_report.json                  # T0 vs T1 inter-model distances
├── checkpoints/                             # per-condition (for resume)
├── canonical_conversations/                 # shared conversations
├── plots/
│   ├── cross_method/
│   │   ├── ranking_heatmap_*.png            # Figure 4-style
│   │   ├── rank_shift_heatmap_*.png         # delta from baseline
│   │   ├── drift_by_method_*.png            # L2/rho/flip by alignment method
│   │   ├── delta_heatmap_*.png              # model x value drift direction
│   │   ├── attractors_*.png                 # attractor/repeller bar charts
│   │   └── convergence_pca_*.png            # PCA trajectory arrows
│   └── <model_key>/<value_set>/             # per-model radar charts
│       └── radar_<domain>_<turns>t.png
└── runs/                                    # per-condition JSON
```

## Related Experiments

### Persona Drifting (LoRA)

The original experiment (`pipeline/run_experiment.py`) measures drift across LoRA persona adapters on a single base model (Llama-3.1-8B-Instruct with adapters from `maius/llama-3.1-8b-it-personas`). This tests how different *personality styles* (sarcasm, sycophancy, etc.) affect drift, rather than different alignment methods.

### Context Length Sweep

`pipeline/sweep_context_length.py` provides a quick single-persona sweep for debugging and exploratory analysis.
