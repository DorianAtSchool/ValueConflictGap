# Plot Interpretation Guide

This document explains every plot produced by
`run_scenario_conversation_experiment.py`. Plots are saved under

```
pipeline/results/scenario_conversation/plots/<run_id>/
  per_condition/<model>/<value_set>/<stance>/<N>t[_<mode>]/
  cross_condition/
    ranking/
    drift/
    aggregated/
    comparison/
```

A new timestamped `run_id` folder is created each run so previous plots are
never overwritten.  Checkpoint data (T0/T1 outcomes, conversations) is reused
across runs.

---

## Glossary

| Term | Meaning |
|---|---|
| **T0** | Baseline probing — no conversation context |
| **T1** | Post-conversation probing |
| **BT ability** | Bradley-Terry log-odds score; higher = model chose this value more often in pairwise comparisons |
| **BT delta** | `ability_T1 − ability_T0`; positive = value rose after conversation |
| **Flip** | A scenario whose winner changed between T0 and T1 |
| **Flip-toward V** | Scenario where T0 winner ≠ V but T1 winner = V |
| **Flip-away V** | Scenario where T0 winner = V but T1 winner ≠ V |
| **Directional consistency** | Fraction of flips in a pair that went the same direction; 1.0 = all flips toward one value, 0.5 = random |
| **Role-filtered** | Statistics restricted to scenarios where a value was in the _favoured role_: v1 for `pro_v1` stance, v2 for `pro_v2` stance |
| **Overall** | Statistics across all scenarios a value appeared in, regardless of role |

---

## Per-condition plots

One folder per `(model, value_set, stance, N_turns[, mode])`.

---

### `bt_ranking_bars.png` — BT Abilities T0 vs T1

**What it shows:** Horizontal bars of raw Bradley-Terry ability scores at T0
(blue) and T1 (orange), sorted by T0 score.  Error bars are 95 % bootstrap
confidence intervals.

**How to read it:**
- If T0 and T1 bars for a value are far apart _and_ the CIs don't overlap,
  the drift is statistically meaningful.
- The x-axis is in log-odds units.  A gap of ~0.15 ≈ +4 pp win-rate against an
  average opponent; a gap of ~0.5 ≈ +12 pp; ~1.0 ≈ +23 pp.
- Values with large, non-overlapping shifts are the ones most affected by the
  conversation context.

**Null result:** T0 and T1 bars nearly identical → conversation had no effect on
this value set.

---

### `drift_bars.png` — Value Drift (BT delta)

**What it shows:** Horizontal bars of `ability_T1 − ability_T0`.
Blue = rose, red = dropped.

**How to read it:**
- A large positive bar = conversation systematically raised that value's
  priority in the model's decisions.
- In stance conditions, read personal and protective values together:
  pushing one side up usually pulls the other side down because BT deltas are
  zero-sum.
- `pro_v1` should be interpreted as pressure toward `value1`; `pro_v2` as
  pressure toward `value2`.

> **BT delta ≠ raw win-rate change.**  BT abilities are zero-sum (they always
> sum to ≈ 0) and are estimated via global maximum-likelihood, weighting
> comparisons by opponent strength.  A value can have a _positive_ net flip
> rate (more wins than losses across scenarios) yet a _negative_ BT delta if
> its losses were concentrated against opponents that themselves gained
> strength.  See the `flip_rates.png` note and `pair_consistency.png` for
> how to diagnose such cases.

---

### `flip_rates.png` — Per-Value Flip Rates

**What it shows:** Grouped bars — blue = flip-toward rate, red = flip-away
rate. The label under each value shows `n=<total_appearances>` (sample size).

For `personalprotective`, values are ordered as:

`authenticity, autonomy, creativity, empowerment | compliance, harmlessness, privacy, responsibility`

**How to read it:**
- **Tall blue, short red:** conversation reliably pushed _toward_ this value.
- **Tall red, short blue:** conversation reliably pushed _away_ from this value.
- **Both tall:** lots of flipping but no consistent direction — noisy effect.
- **Both near zero:** the value's outcome was stable; conversation had little
  influence on it.
- Low `n` values (small sample) make rates less reliable — interpret with care.
- In stance conditions, this single plot is intended to show both sides at
  once: movement toward the favoured values and movement away from the opposite
  side.

> **Why flip rates can disagree with BT delta:**
> The net flip rate (`flip_toward − flip_away`) equals the raw **win-rate
> change** for that value — they are mathematically identical.  However, BT
> delta is a _global_ maximum-likelihood fit that weights each comparison by
> opponent strength.  A value can gain a few net wins against weak opponents
> while losing heavily to a single strong, rising opponent; the raw win-rate
> (and thus net flip rate) goes up slightly, but the BT ability drops because
> the model accounts for _who_ you beat and _who_ beat you.
>
> To diagnose such cases, use `pair_consistency.png` to see the per-pair
> breakdown: which pairs flipped, how many scenarios were in each pair, and
> which value dominated the flips.

---

### `pair_consistency.png` — Pair Flip Consistency

**What it shows:** Heatmap with one row per value pair and two colour columns:

| Column | Meaning |
|---|---|
| `flip rate` | Fraction of scenarios in this pair where the answer changed T0→T1 |
| `dir. consistency` | Fraction of flips that went the _same_ direction (toward one particular value) |

Row labels on the left include `(n=<scenarios>)` so you know the sample size
behind each pair.  The text annotations on the **right side** show:
`Xv1→v1  Yv2→v2  |  dominant_value` — the raw flip counts toward each value
in the pair, followed by the **dominant value** (whichever received more
directional flips).

**How to read it:**
- **High flip rate + high consistency (≥ 0.7):** the conversation reliably
  tilted this specific value conflict.  The dominant value (right label) is
  the one that benefits.
- **High flip rate + low consistency (≈ 0.5):** lots of instability but no
  directional signal — possibly noise or the pair is near the model's decision
  boundary.
- **Low flip rate:** the conversation barely moved this pair at all.

**Colour scale:** green = high (good for consistency; high for flip rate =
volatile), red = low.

> **Connecting pair_consistency to drift_bars and flip_rates:**
> This plot is essential for explaining apparent contradictions between
> per-value flip rates and BT deltas.  A value can have a positive net flip
> rate (more flips toward than away, aggregated across all pairs) while
> showing a large _negative_ BT delta.  This happens when the value's net
> wins come from small pairs against weak opponents, while it suffers heavy
> losses in a large pair against a strong, rising opponent.  The pair-level
> view here reveals that structure: look for pairs with high `n`, high flip
> rate, and a dominant value that is _not_ the one you're investigating.

---

### `pair_flip_table.png` — Per-Pair Flip Breakdown Table

**What it shows:** A full table with one row per value pair.  Columns:

| Column | Meaning |
|---|---|
| `Pair` | The two values in the comparison |
| `n` | Number of scenarios in this pair |
| `→ v1` | Number of flips _toward_ the first value |
| `→ v2` | Number of flips _toward_ the second value |
| `net` | `→v1 − →v2`; positive = net toward v1, negative = net toward v2 |
| `flip rate` | Fraction of scenarios that changed answer T0→T1 |
| `dominant` | The value that received more flips |

**How to read it:**
- This is the most detailed view of where flips occur and in which direction.
- Use it to diagnose why per-value flip rates and BT deltas disagree: look
  for pairs with high `n` and large `net` values — these are the pairs that
  drive the BT model's outcome.
- The `net` column is colour-coded: blue = net toward v1, red = net toward v2.
- The `flip rate` column is colour-coded: redder = more volatile.

---

### `summary.png` — Condition Summary (composite)

**What it shows:** Three panels side by side:
1. **Left — BT Abilities T0 vs T1:** same as `bt_ranking_bars.png`
2. **Center — BT Delta:** same as `drift_bars.png`
3. **Right — Per-Pair Flip Table:** same as `pair_flip_table.png`

**How to read it:**
- The composite layout lets you see all three key metrics at once without
  switching between files.
- Start with the center panel (BT delta) to see which values moved, then
  look left to see absolute positions, then right to see _which pairs_ drove
  the movement.
- This is the recommended starting point for per-condition analysis.

---

### `radar.png` — BT Spider Chart T0 vs T1

**What it shows:** Overlaid spider charts of BT abilities at T0 (blue) and T1
(red) for all values in the set.

**How to read it:**
- The overall _shape_ of T0 vs T1 tells you whether the hierarchy changed
  broadly or locally.
- A value whose spoke is much longer at T1 rose; shorter = dropped.
- If T0 and T1 nearly overlap, the conversation had little effect on the
  overall ranking structure.

Note: absolute scale is shifted so all scores are positive (required for radar
display).  Use `bt_ranking_bars.png` for exact magnitudes.

---

### `ranking_heatmap.png` — Ordinal Ranks T0 vs T1

**What it shows:** A 2-row heatmap (T0 / T1), columns = values, cell = ordinal
rank (1 = highest BT ability).  Lighter colour = higher priority.

**How to read it:**
- A cell jumping from rank 5 to rank 1 between rows = a large upward shift.
- Useful for spotting rank _inversions_ (two values swapping priority), which
  can be masked in BT delta plots if both moved the same direction.

---

### `rank_shift_heatmap.png` — Rank Change from Baseline

**What it shows:** Single row of cells showing `rank_T1 − rank_T0` per value.
Blue = rose in priority (negative delta); red = dropped.

**How to read it:**
- Zero cells: no rank change.
- Negative (blue) = the value gained rank positions after the conversation.
- Large absolute values = dramatic reordering.

---

## Cross-condition plots

Located in `cross_condition/`.

---

### `ranking/ranking_heatmap_<model>_<vs>.png`

**What it shows:** Multi-panel heatmap.  Each panel = one time point
(T0 baseline + T1 at each turn count).  Rows = stances, columns = values.

**How to read it:**
- Track a value across panels left-to-right to see whether its rank changes
  as turn count increases.
- Compare rows (stances) to see whether `pro_v1` consistently elevates v1
  relative to `neutral`.

---

### `ranking/rank_shift_heatmap_<model>_<vs>.png`

**What it shows:** Per-panel rank delta from T0 baseline.  One panel per turn
count; rows = stances.

**How to read it:**
- Blue cells = value rose; red = dropped.
- If the same cell is blue across all stances, the effect is not stance-specific
  — the conversation topic itself (not the bias) drives the drift.
- If blue/red patterns are _opposite_ between `pro_v1` and `pro_v2`, the
  steering is working correctly.

---

### `ranking/radar_panel_<model>_<vs>_<shared|local>.png`

**What it shows:** Grid of radar charts, one per `(turns × stance)` condition.
Each radar overlays T0 (blue) and T1 (red).

- **shared-scale:** all subplots use the same axis range — use this to compare
  _magnitudes_ across conditions.
- **local-scale:** each subplot auto-scales — use this to compare _shapes_
  within each condition without scale confounds.

---

### `drift/drift_by_turns_<model>_<stance>.png`

**What it shows:** L2 distance between T0 and T1 BT ability vectors, plotted
vs. number of conversation turns.  One line per value set.

**How to read it:**
- **Flat line:** conversation length does not matter; drift saturates quickly
  or doesn't accumulate.
- **Rising line:** more turns = more drift.  Meaningful if the trend is
  consistent across value sets.
- Diverging lines between value sets reveal which value sets are more sensitive
  to context length.

---

### `drift/l2_heatmap_<model>_<stance>.png`

**What it shows:** Two side-by-side heatmaps: L2 drift (left) and overall flip
rate (right).  Rows = value sets, columns = turn counts.

**How to read it:**
- Bright cells = high drift or high flip rate for that combination.
- Compare rows to find which value sets are most susceptible to context.
- Compare columns to find the turn count at which drift stabilises.

---

### `aggregated/drift_bars_<model>_<vs>_<stance>.png`

**What it shows:** Per-value BT delta _averaged_ over all turn-count conditions.
Error bars = ±1 SD across turn counts.

**For non-neutral stances:** dual panels (overall + role-filtered).

**How to read it:**
- The mean bar shows the _central tendency_ of drift across turn lengths.
- A wide SD bar means the effect is highly turn-count-dependent; narrow = robust.
- Values with large, consistent bars are the ones most reliably shifted by
  context of this type.

---

### `aggregated/flip_rates_<model>_<vs>_<stance>.png`

**What it shows:** Mean flip-toward / flip-away rates per value, averaged over
turn-count conditions.  Error bars = ±1 SD.

**For non-neutral stances:** dual panels (overall + role-filtered).

**How to read it:**
- Complements `drift_bars`: a value can drift in BT score with few flips
  (if every flip is consistent), or show many flips that cancel out (net zero
  drift).
- High flip-toward + low flip-away = reliable steering toward that value.
- Near-equal toward and away = the conversation destabilises the decision
  without a systematic direction.

---

### `comparison/stance_comparison_<model>_<vs>_<N>t.png`

**What it shows:** For a fixed `(model, value_set, turns)`, side-by-side bars
comparing flip-toward (left panel) and flip-away (right panel) across stances.
One bar cluster per value, bars coloured by stance.

**How to read it:**
- The key question: does the `pro_v1` bar for value V rise above `neutral`
  when V was v1 in the scenarios?  If yes, the stance steering worked.
- If `pro_v1` and `pro_v2` produce identical patterns to `neutral`, the
  conversation framing had no directional effect.
- Look for _opposing_ patterns between `pro_v1` and `pro_v2` for the same
  value — that is the clearest evidence that the stance is doing something.

---

### `comparison/drift_by_model_<vs>.png` — Drift by Model

**What it shows:** Two side-by-side heatmaps: L2 drift (left) and overall flip
rate (right).  Rows = models, columns = turn counts.

**How to read it:**
- Directly compares drift magnitude across all models for a given value set.
- Bright cells = high drift or high flip rate.
- The ranking of rows reveals which models are most susceptible to
  conversation-induced drift.
- Compare L2 (magnitude of BT vector change) with flip rate (fraction of
  scenarios that changed answer) — a model can have high L2 with low flip rate
  if a few flips are in high-leverage pairs.

---

### `comparison/model_comparison_<vs>_<stance>_<N>t.png`

**What it shows:** Three panels (BT delta, flip-toward, flip-away), one cluster
per value, bars coloured by model.  Fixed `(value_set, stance, turns)`.

**How to read it:**
- A model with consistently taller bars drifts more under this type of
  conversation.
- Bars of opposite sign for the same value across models = different models are
  pulled in different directions by the same conversation.
- Near-identical bar heights = the effect is driven by the value/topic, not
  model-specific susceptibility.

---

### `comparison/mode_comparison_<model>_<vs>_<N>t.png`

**What it shows:** MCQ vs. open-ended probing, for each value: flip-toward
(left panel) and flip-away (right panel).

**How to read it:**
- Taller bars in the `openended` condition suggest that free-form responses
  reveal stronger drifts than forced-choice MCQ.
- If patterns are similar between modes, the effect is robust to how you elicit
  the model's preference.
- Large discrepancies may indicate the judge (used for open-ended scoring) is
  introducing noise.

---

## Tips for a first analysis pass

1. **Start with `per_condition/.../summary.png`** — this composite view shows
   BT abilities, BT delta, and the per-pair flip breakdown side by side.  It's
   the single most informative plot per condition.

2. **Use `comparison/drift_by_model`** to compare drift magnitude across all
   models at a glance.

3. **Check `aggregated/drift_bars`** to get the turn-count-averaged story:
   which values consistently drift, and in which direction.

4. **Check `comparison/stance_comparison`** to see whether intentional steering
   (`pro_v1` / `pro_v2`) actually moved values in the expected direction.

5. **Open `per_condition/.../flip_rates.png`** for the role-filtered view under
   non-neutral stances — this is the cleanest measure of whether the stance
   achieved its goal.

6. **Use `per_condition/.../pair_flip_table.png`** to see the full per-pair
   breakdown with raw flip counts.  This is essential for diagnosing why
   per-value flip rates and BT deltas can disagree (see notes in the
   flip_rates and drift_bars sections).

5. **Look at `drift/drift_by_turns`** to decide whether longer conversations
   are worth the cost.  If drift plateaus after 5 turns, running 10-turn
   experiments adds little.

6. **Use `comparison/model_comparison`** only after running multiple models —
   it answers whether susceptibility is model-specific or topic-driven.
