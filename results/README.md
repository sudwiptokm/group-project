# results/ — curated, version-controlled outputs

`models/`, `logs/`, and `params/` are **gitignored** (large/regenerable run scratch).
This directory is the opposite: a small, tracked home for the **final artifacts you
actually cite in the report / viva**. Copy them here by hand once a run is finalised,
then commit — that way the conclusions survive even though the raw run outputs don't.

## What to put here

| File | Copy from | What it is |
|------|-----------|------------|
| `comparison_stage1.csv` | `logs/comparison.csv` (after Stage 1) | ranked algo table + fixed-time row → the winner |
| `comparison_stage2.csv` | `logs/comparison.csv` (after Stage 2) | λ ∈ {0.0,0.5,1.0} safety/efficiency tradeoff for the winner |
| `params_<winner>.json` | `params/<winner>.json` | tuned hyperparameters of the chosen algorithm |
| `*.png` | your analysis/plotting step | waiting-time curves, comparison bars, tradeoff curve |
| `RUN_NOTES.md` | — | which MODE, seeds, date, git SHA the numbers came from |

## How to curate after a run

```bash
cp logs/comparison.csv results/comparison_stage1.csv
cp params/<winner>.json results/params_<winner>.json
# add plots, jot the run config in results/RUN_NOTES.md
git add results/ && git commit -m "results: stage 1 comparison (<mode>, git <sha>)"
```

Record the **git SHA** and **MODE** each result came from — a `full`-budget table and
an `overnight` table look identical in format but are not the same experiment.
