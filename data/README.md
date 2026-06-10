# Data

This directory vendors the static inputs needed to run the paper experiments
without depending on external repository layouts.

- `sva/`: ValueActionLens/VIA action data used for Experiment 1, single-value
  agreement-to-action (SVA).
- `value_conflicts/`: ConflictScope-derived personal/protective value-conflict
  scenarios used for Experiments 2, 2b, and 3 (VCA, VCA-b, VCP).

The original source repositories can remain in the workspace for reference, but
the clean experiment entry points read from this directory.

