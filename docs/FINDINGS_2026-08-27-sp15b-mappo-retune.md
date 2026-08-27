# SP15b Findings: does retuning HPs for MAPPO's wider critic close the gap to IPPO?

## The question

SP15 (docs/FINDINGS_2026-08-27-sp15-mappo-smoke.md) found mappo underperforms
ippo at n=1 seed, but flagged a live, undiscriminated confound: PPO
hyperparameters were the project's single-intersection-tuned defaults,
reused unmodified for MAPPO's 57-dim joint critic (vs IPPO's 19-dim local
one). That smoke test couldn't tell "coordination doesn't help" apart from
"wrong HPs for the wider critic." This is the "cheap" half of SP15's own
recommendation (retune-and-rerun vs. stop).

## Method

Three manual HP variants, all `mappo`, seed=42, `corridor_peak`, λ=0.5,
`min_green=10`, 100k steps — same protocol as SP15, changing only what a
wider critic input plausibly needs (`analysis/mappo_retune_smoke.py`; a full
Optuna search was out of scope — `tune.py` targets algos.py's
single-intersection SB3 algorithms, not this corridor PPO loop, and a
30-trial search at ~30min/trial would cost more than this smoke test's own
value justifies):

- **lr_low**: lr / 5 (2.3195e-05 -> 4.639e-06) — smaller step for a
  noisier-per-dim gradient from the 3x wider input
- **batch64**: batch_size x2 (32 -> 64) — larger minibatches average out more
  of the extra per-agent noise in the joint state
- **wide512**: hidden (256,256) -> (512,512) — more capacity for a 3x wider
  input (this also widens the actor, a disclosed asymmetry:
  `ppo_core.ActorCritic` ties actor/critic hidden size, so this isn't a
  critic-only capacity change)

`train_corridor.train()`/`evaluate()` gained `hp_overrides`/`tag_suffix`
keyword params (default `None`/`""`, backward-compatible) so a variant's
checkpoint and eval CSV never collide with the default-HP run at the same
scenario/lam/seed. Baselines (ippo, mappo default-HP) are read from SP15's
own tripinfo XMLs, not re-run.

## Results

Delay/trip in seconds per completed trip, seed 42, `corridor_peak`:

| variant | regular | irregular |
|---|---:|---:|
| ippo (baseline) | 16.91s | 19.14s |
| mappo (baseline, default HP) | 18.26s | 20.08s |
| mappo_lr_low | **27.13s** | **29.24s** |
| mappo_batch64 | 18.03s | 19.69s |
| mappo_wide512 | 17.44s | 19.87s |

Gap to ippo (positive = mappo variant still slower):

| variant | regular gap | irregular gap |
|---|---:|---:|
| mappo (baseline) | +1.35s | +0.94s |
| mappo_lr_low | +10.22s | +10.10s |
| mappo_batch64 | +1.12s | +0.55s |
| mappo_wide512 | +0.53s | +0.73s |

## Verdict: retuning narrows the gap but does not close it — and the wrong direction makes it much worse

**None of the three variants beats ippo on either geometry.** `wide512`
(more capacity) gets closest: it cuts the regular-net gap from +1.35s to
+0.53s (a 61% reduction) and the irregular-net gap from +0.94s to +0.73s
(22%) relative to the default-HP `mappo` baseline — a real, if partial,
correction in the direction SP15's disclosed confound predicted. `batch64`
also helps a little (regular gap −0.23s, irregular gap −0.39s vs baseline
mappo) but far less than `wide512`. **`lr_low` makes things drastically
worse** (27.13s/29.24s, roughly 10s worse than ippo on both geometries) —
five-times-smaller learning rate looks to have left the 100k-step budget
insufficient to converge at all, not merely "safer." This is itself useful
evidence: the HP-sensitivity confound SP15 flagged is real and can cut either
way, not just toward hiding a coordination benefit.

Taken together, this strengthens rather than overturns SP15's original
verdict. The best of three cheap, plausible retuning directions still leaves
`mappo` behind `ippo` on both geometries — the gap shrank, it did not flip.
"Coordination doesn't help here" is now somewhat better supported than
before (it survives the most obvious HP objection this session tried), but
it is not a closed question: three hand-picked points are not a search, and
`lr_low`'s catastrophic result shows this HP space is sensitive enough that
an actual sweep (Optuna, as `tune.py` does for the single-intersection
algorithms) could plausibly find a better point than `wide512` — a
genuinely-competitive-with-ippo MAPPO configuration has not been ruled out,
only not found here.

## What this doesn't answer

- **n=1 seed, unchanged from SP15.** Every number here, like SP15's, is a
  single episode on a single seed; none of this touches that limitation.
- **Only 3 manually-chosen HP points, not a search.** `lr_low`'s outcome
  shows the space is sensitive; a real Optuna sweep over lr/hidden/batch/etc.
  jointly (not one dimension at a time) could land somewhere between
  `wide512`'s partial improvement and `lr_low`'s collapse — genuinely
  unknown without running it.
- **`wide512` widens the actor too**, not just the critic (architecture
  constraint, see Method) — its improvement is not cleanly attributable to
  "the critic got more capacity" alone.
- **Still only 2 geometries**, same as SP15 and SP13's own open question
  about whether this ranking holds across SP13's asymmetry-ratio sweep.

## Recommendation

Given the best cheap retune only narrows (doesn't close) the gap, and a real
HP search remains unrun and unbudgeted, escalating to the full n=10-seed
MAPPO rigor is still not warranted by the evidence in hand. The honest
framing stands, strengthened: MAPPO was built, smoke-tested, and given one
round of HP retuning aimed squarely at its own disclosed confound — the
retuning helped some, not enough to compete with the independent-agent
baseline, and this project stops here on MAPPO rather than escalate further.
