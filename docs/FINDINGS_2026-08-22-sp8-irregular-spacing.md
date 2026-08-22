# SP8 Findings: does breaking the corridor's regular spacing let RL win? Plus three cheap follow-ups on SP6/SP7's open threads.

## The question

SP4/SP6/SP7 all found the same shape of result on `corridor.net.xml`: a
competently-timed fixed plan (`green_wave`) beats both a reactive heuristic
(`max_pressure`) and a learned controller (`idqn`) on ordinary demand, demand
shifts, and a mid-episode incident. All three used the same network: three
signals (C1, C2, C3) 200m apart on a straight line. `green_wave`'s offset is
exact travel time from C1, and equal spacing lets one offset serve both the
eastbound and westbound direction of the shared through-phase at every
signal — a structural advantage no amount of demand variation removes.

This asks the obvious next question: is the fixed plan winning because it's
a good controller, or because the network handed it a geometry it can solve
exactly? `corridor_irregular.net.xml` keeps the same topology and demand but
makes the arterial spacing asymmetric (C1→C2 578m, C2→C3 78m instead of
200m/200m), which a single shared offset can no longer serve well in both
directions at once.

No training happens here. `idqn`'s existing `corridor_peak`-trained
checkpoints (SP5) are evaluated zero-shot on the new geometry — same posture
as SP6's demand-shift eval. `green_wave`/`max_pressure` regular-net reference
numbers are read from `analysis/corridor_sweep.csv` (already computed, SP4);
only the irregular-net runs are new (`analysis/irregular_net_compare.py`,
seeds 42/43/44, `corridor_peak` demand, `min_green=10`).

## Result: green_wave loses for the first time in this project

| controller | regular net | irregular net | Δ (irregular − regular) |
|---|---|---|---|
| green_wave | 13.47s | **19.62s** | +6.15s ± 0.26s |
| max_pressure | 26.38s | **22.17s** | −4.21s ± 4.46s |
| idqn | 16.56s | **18.48s** | +1.92s ± 0.39s |

**On the irregular network, idqn (18.48s) beats green_wave (19.62s).**
Ranking flips from green_wave < idqn < max_pressure (regular) to
idqn < green_wave < max_pressure (irregular). Trip counts stay consistent
across conditions (~2900-3000 per run) — not a survivorship-biased
comparison.

This is mechanistically exactly what was predicted: `green_wave`'s single
per-signal offset is exact for one direction and increasingly wrong for the
other as spacing diverges from uniform, so it degrades the most (+6.15s) of
any controller when spacing breaks. `max_pressure` and `idqn` don't depend on
geometry-derived offsets at all — they react to local queue/density state —
so a geometry shift that specifically breaks the fixed plan's coordinating
assumption leaves them comparatively untouched. `max_pressure` actually
*improves* on the irregular net (its high variance, ±4.46s, means this
isn't a tight result, but the direction is consistent with the mechanism).

**Caveat:** n=3 seeds, same limitation every SP in this project has carried.
idqn's win margin (1.14s) is smaller than green_wave's own no-incident sd
elsewhere in this project, so this should be read as "the fixed plan's
advantage is not geometry-independent," not "idqn is definitively better on
irregular networks" — that would want a wider seed set and probably a second
irregular-geometry variant before it's a load-bearing claim.

## Implementation note

`net_file` is now a first-class parameter through `make_corridor_env`,
`corridor_baseline.run()`/`green_wave_actions()`, and
`train_corridor_dqn.evaluate()`. `green_wave`'s signal positions and
free-flow speed are read dynamically from the net file
(`corridor_baseline._green_wave_inputs`, `sumolib`-based) instead of the
hardcoded `SIGNAL_POSITIONS`/`FREE_FLOW_SPEED` constants that existed before
— those required hand-sync on any geometry change, which a second network
variant would have made a real correctness risk (the exact class of
glob-confound bug SP6/SP7 hit, just at the geometry layer instead of the
filename layer). Non-default `net_file` runs get a `_net<label>` filename
fragment; `analysis/irregular_net_compare.py` deliberately bypasses
`compare.py`'s glob-based aggregation entirely (reads tripinfo XML directly)
rather than teach that machinery a fourth tag dimension.

## Three cheap follow-ups run alongside this (SP6/SP7's own disclosed gaps)

**1. SP7's incident-window number** (`analysis/incident_window_compare.py`).
SP7's own spec asked for both a whole-episode and incident-window delay
number; only whole-episode shipped. Computed directly from tripinfo XMLs
already on disk (only 3 missing `green_wave` no-incident tripinfo files
needed regenerating):

| controller | whole-episode Δ (reported) | in-window Δ (new) | ratio |
|---|---|---|---|
| green_wave | +1.13s | +4.21s ± 0.96s | 3.72x |
| max_pressure | +0.87s | +2.69s ± 7.34s | 3.08x |
| idqn | +0.49s | +1.13s ± 0.61s | 2.32x |

Confirms the disclosed ~4x dilution estimate (25% of trips are exposed to
the 900s/3600s window). Ranking is unchanged at the sharper measurement —
idqn's Δ is still smallest — but max_pressure's in-window variance (±7.34s)
is even more seed-dominated than the whole-episode number suggested.

**2. SP7's n=3→n=10 seed widening** for `green_wave`/`max_pressure` (idqn
stays n=3, no more checkpoints without training):

| controller | n=3 Δ | n=10 Δ | n=10 sign-flips |
|---|---|---|---|
| green_wave | +1.13s ± 0.13s | +1.39s ± 0.69s | 0/10 |
| max_pressure | +0.87s ± 3.57s | +0.89s ± 2.48s | 3/10 |

Settles the open question: max_pressure's seed-dependent sign flip is real,
recurring signal, not n=3 noise — 3 of 10 seeds get *faster* under the
incident. green_wave remains consistent (0/10, tightening slightly relative
to n=3). Mean Δ for both barely moves from the n=3 estimate.

**3. `corridor_skew_hi`**, a new scenario pushing C2's cross-street demand
from `corridor_skew`'s 600 veh/h to 1800 veh/h — this codebase's own cited
per-lane saturation ceiling (`make_scenarios.py`'s `SKEW_HI_PROFILE`),
addressing SP4's disclosed-and-never-tested flaw that 600 veh/h never
approached it:

| controller | corridor_skew (600) | corridor_skew_hi (1800) |
|---|---|---|
| green_wave | ~13.5s (unchanged from peak) | 13.65s |
| max_pressure | +14.31s gap | +9.26s gap |
| idqn | +3.12s gap | **+2.18s gap** |

Green_wave still wins clearly — its fixed through/cross split doesn't get
overwhelmed even at saturation on C2 alone. But idqn's gap shrinks ~30%
(3.12s → 2.18s) as skew intensity approaches the theoretical ceiling the
mechanism (`corridor_control.plan_phase_seconds`/`fixed_time_phase`'s one
global split) predicts should matter — directionally consistent with the
hypothesis, not enough to flip it.

## Where this leaves the project

The "consolidate" recommendation from `docs/HANDOFF_2026-08-21.md` was
conditioned on ordinary demand, demand shifts, and a mid-episode incident —
and held in all three. It does **not** hold unconditionally: geometry is a
fourth axis, and on this one irregular-spacing variant, idqn wins. That's a
narrower and more interesting claim than "consolidate" or "keep training RL"
— it says the fixed plan's advantage is specifically a *this-corridor's-
geometry* advantage, not a general one. Whether that holds beyond one
irregular-geometry sample (wider seeds, a second irregular variant, a grid
topology) is the natural next question if anyone wants to chase it.
