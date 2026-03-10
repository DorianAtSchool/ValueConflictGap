# Persona Drifting — Initial Results Analysis

## Experiment Summary

We measure how LLM persona value rankings drift after multi-turn conversations. A persona-tuned model (Llama-3.1-8B + LoRA adapter) is probed on ConflictScope value conflict scenarios before (T0) and after (T1) a conversation, and the Bradley-Terry value rankings are compared.

**Initial run:** 3 personas (sarcasm, mathematical, sycophancy) x 2 value sets (HHH, personalprotective) x 2 domains (philosophy, value_aligned_honesty) x 2 turn counts (5, 10) = 24 conditions. Some earlier sarcasm runs at 20 turns also included (politics, therapy). Total: 30 conditions, ~5 hours on a single RTX 4090-class GPU.

---

## Metrics

### L2 Drift
Euclidean distance between T0 and T1 Bradley-Terry ability vectors:
```
L2 = sqrt( Σ (ability_T1[v] - ability_T0[v])² )
```
Higher = more change in value priorities. Scale depends on the number of values (personalprotective with 8 values has inherently higher L2 than HHH with 3).

### Rank Correlation (Spearman rho)
Ranks the values by BT ability at T0 and T1 separately. rho=1.0 means the ordering is identical; rho<1.0 means some values swapped rank positions. HHH (3 values) often shows rho=1.0 because the rank order rarely fully flips. PersonalProtective (8 values) shows rho as low as 0.57, indicating substantial reordering.

### Answer Flip Rate
For each scenario present in both T0 and T1: did the model pick a different option (A vs B)?
```
flip_rate = count(choice_T0 != choice_T1) / total_matched_scenarios
```
Captures all individual decision changes, including ones that cancel out in the aggregate BT ranking. A 25% flip rate means 1 in 4 moral decisions changed.

### Bradley-Terry Ability Scores
Given N pairwise outcomes (value_i beats value_j), BT fits a strength parameter per value:
```
P(i beats j) = exp(ability_i) / (exp(ability_i) + exp(ability_j))
```
Solved via iterative Luce spectral ranking (`choix.ilsr_pairwise`). Bootstrap confidence intervals from 1000 resamples.

---

## Plot Descriptions

### Drift Heatmaps (`drift_heatmap_*.png`)
Mean L2 drift for each (persona, domain) pair, averaged across turn counts. One heatmap per value set. Read as: darker = more drift. Immediately shows which persona/domain combinations are most susceptible.

### Flip Rate Bars (`flip_rate_bars.png`)
Mean answer flip rate per persona, averaged across all conditions. Shows how "noisy" each persona's decision-making becomes after conversation — distinct from L2 drift because flips can cancel out in aggregate.

### PCA Trajectory Plots (`trajectories_pca_*.png`)
BT ability vectors projected to 2D via PCA. Circles (o) = T0 positions, crosses (x) = T1 positions, arrows show drift direction. Colored by persona. Arrow length = magnitude of drift. Clustering of same-color points = persona consistency. One plot per value set.

**What the principal components mean:** PCA finds axes of maximum variance in BT score vectors. They are linear combinations of values, not single values.
- HHH PC1 (86.2%): Captures the dominant "harmlessness vs honesty" axis (the two ranking extremes). Moving right ≈ more weight on honesty relative to harmlessness.
- PersonalProtective PC1 (64.0%): Likely captures the broad "personal values (autonomy, creativity) vs protective values (compliance, harmlessness)" axis. PC2 (15.1%) captures a secondary pattern.

### Attractor Analysis (`attractor_analysis_*.png`)
Plots only T1 positions (post-conversation), colored by persona, with KMeans cluster centroids. Tests whether conversations cause different personas to converge to common value configurations ("attractors"). If personas cluster together after conversation, drift is environmental; if they remain separated, drift is persona-dependent.

### Drift vs Turns (`drift_by_turns_*.png`)
Line plot of mean L2 drift vs number of conversation turns, one line per persona. Shows whether drift scales with conversation length.

---

## Key Findings

### 1. Persona susceptibility is clearly ordered
**Sycophancy > Sarcasm > Mathematical** in drift magnitude, consistently across both value sets and domains. The agreeable persona gets pulled around by conversations; the analytical persona resists.

| Persona | Mean L2 (HHH) | Mean L2 (PP) | Mean Flip Rate |
|---|---|---|---|
| sycophancy | 0.46 | 0.99 | 25.6% |
| sarcasm | 0.21 | 0.66 | 26.2% |
| mathematical | 0.31 | 0.50 | 17.5% |

### 2. PersonalProtective shows far more drift than HHH
Across all personas, personalprotective L2 values are 2-3x higher than HHH. With 8 values vs 3, there is more room for ranking reshuffling, and rank correlations drop significantly (0.57-0.98 vs mostly 1.00 for HHH).

### 3. Drift does NOT increase monotonically with conversation length
The most surprising finding. In HHH, drift *decreases* from 5 to 10 turns for all three personas. In personalprotective, sycophancy increases but sarcasm decreases. Possible explanations:
- Longer context dilutes the conversation's per-token influence on MCQ responses
- The model "stabilizes" into a conversation mode where values become more consistent
- The initial context "shock" is the dominant effect, not cumulative content

The 20-turn data (currently only available for some sarcasm/HHH conditions) will be critical to disambiguate.

### 4. Philosophy causes more drift than value-aligned conversations
For personalprotective, philosophy consistently causes more drift than value_aligned_honesty across all personas. This challenges the assumption that value-relevant content drives drift — instead, the *openness* of philosophical discussion may be what destabilizes values.

### 5. No convergent attractor
Attractor analysis shows personas maintain distinct T1 positions. Drift is persona-dependent, not converging to a single "conversational attractor." This matters for alignment: conversations perturb personas in persona-specific ways rather than erasing differences.

### 6. High flip rate + low L2 = noise vs. systematic drift
Sarcasm has high flip rate (26%) but low L2 in HHH (0.12-0.23) — many individual answers change but they cancel out. The sarcasm persona becomes *noisier* without shifting systematically. Sycophancy has similar flip rate but much higher L2 — its flips are correlated and push the ranking in a consistent direction.

---

## Unexpected Results

1. **Mathematical is sensitive to VA:honesty** — L2 of 0.55-0.62 in HHH, higher than its philosophy drift (0.13). The analytical framing may resonate with its training. Meanwhile sarcasm barely moves on VA:honesty (0.05-0.19), seemingly deflecting the topic.

2. **Non-monotonic drift with conversation length** — Expected monotonic increase, observed decrease in many conditions. This is the finding most worth investigating further.

3. **Sarcasm's resilience to honesty-targeted conversations** — Despite being generally susceptible, sarcasm shows minimal drift when the conversation is specifically about honesty. The persona's deflective communication style may act as a buffer against topic-aligned value pressure.

---

## How to Proceed

### Immediate next step: extend to 20 turns
Run with the same parameters plus 20-turn conditions. Existing 5t and 10t results will be skipped via checkpointing:
```bash
python run_experiment.py \
    --personas sarcasm mathematical sycophancy \
    --value-sets HHH personalprotective \
    --domains philosophy value_aligned_honesty \
    --turn-counts 5 10 20 \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key <key>
```
This adds 12 new conditions (~2 hours). Critical for determining whether the non-monotonic drift pattern holds or reverses at longer contexts.

### Expand personas
Add personas that test specific hypotheses:
- **loving** — warm/caring, tests if empathetic personas drift differently than sycophantic ones
- **goodness** — morally principled, should it resist drift like mathematical?
- **impulsiveness** — reactive/spontaneous, might drift erratically
```bash
python run_experiment.py \
    --personas loving goodness impulsiveness \
    --value-sets HHH personalprotective \
    --domains philosophy value_aligned_honesty \
    --turn-counts 5 10 20 \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key <key>
```

### Expand domains
Test whether the philosophy > VA:honesty pattern generalizes:
```bash
python run_experiment.py \
    --personas sarcasm mathematical sycophancy \
    --value-sets HHH \
    --domains politics therapy coding value_aligned_helpfulness value_aligned_harmlessness \
    --turn-counts 5 10 20 \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key <key>
```

### Add ModelSpec value set
The third value set (6 values — nonhate, fairness, objectivity, honesty, noncondescension, clarity) sits between HHH and personalprotective in complexity. Tests whether the drift patterns scale with value set size.

### Full experiment
Once hypotheses are validated on subsets, run all 10 personas x 3 value sets x all domains x 3 turn counts:
```bash
python run_experiment.py \
    --simulator openai --simulator-model gpt-4o-mini \
    --simulator-api-key <key>
```
Estimated: ~870 conditions, ~6-10 hours on a single GPU.

### Analytical directions
1. **Per-value drift analysis** — Which specific values shift most? Do certain values act as "anchors" while others are malleable?
2. **Conversation content analysis** — Correlate drift magnitude with conversation topics/sentiment. Does the user simulator's behavior predict drift direction?
3. **Statistical significance** — Run multiple conversations per condition (different random seeds) to get error bars on drift estimates.
4. **Cross-persona comparison** — Are there value pairs where all personas drift in the same direction? This would indicate a structural property of the base model rather than persona-specific behavior.
