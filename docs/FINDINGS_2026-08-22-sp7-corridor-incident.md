# SP7 Findings: does reactive/learned control earn anything under a mid-episode incident?

## The question

SP4-SP6 all found the same shape of result on this corridor: a
competently-timed fixed plan (`green_wave`) beats both a reactive
heuristic (`max_pressure`) and a learned controller (`idqn`) on ordinary
demand, including demand shifts the fixed plan never saw
(`docs/FINDINGS_2026-08-22-sp6-idqn-demand-shift.md`). That leaves one
scenario type still untested: an *unplanned, mid-episode disruption* — the
one situation where a fixed plan's blindness to real-time state should, in
principle, cost it the most. If reactive or learned control ever earns its
complexity here, this is where it should show up.

The design keeps `max_pressure` in the comparison deliberately
(`docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md` §5).
`max_pressure` is already reactive — it observes queues and adapts every
cycle — but it has never learned anything from data. If IDQN's disruption
cost is smaller than `max_pressure`'s too, and not just smaller than
`green_wave`'s, that's evidence *learning* specifically helps, not just
"being reactive at all." If IDQN's cost is merely comparable to
`max_pressure`'s, the result reduces to the expected, non-novel claim that
any reactive controller beats a blind plan under disruption.

The incident: lane `C1_C2_0` (one of two lanes on the C1_C2 arterial) is
closed to all vehicle classes from t=1800s to t=2700s (15 minutes) within a
3600s `corridor_peak` episode, then reopens. Fixed, deterministic, same
across all controllers and seeds (seeds 42/43/44, `min_green=10`, matching
SP5/SP6's calibrated floor). Ranking metric is delay per completed trip
(`trip_time_loss_mean`), not `system_mean_waiting_time`.

**A note on how this run got here.** The first attempt at this eval
(commit `a0f91fe`, since superseded) ran against a lane-closure
implementation that had a real bug: it only restricted the `passenger`
vClass, leaving `moto`/`auto` — 84.5% of this corridor's demand — free to
use the "closed" lane. A second bug in the comparison script subtracted a
10-seed baseline mean instead of each seed's own no-incident value, which
would have been misleading on top of the first bug given `max_pressure`'s
no-incident delay distribution at `min_green=10` is genuinely bimodal (2 of
10 seeds run ~21-22s, the other 8 run ~26-29s — confirmed directly against
all 10 seeds in `analysis/corridor_sweep.csv`). Both defects are fixed as
of commits `b09c373`/`1c1411c`; every number below is from the corrected
run (`analysis/incident_compare.csv`, commit `b7354fe`). The numbers
changed materially once the incident actually closed the lane for all
traffic — see below.

## Results

| controller | no-incident delay/trip | incident delay/trip | incident cost (Δ, mean ± sd, n=3) |
|---|---|---|---|
| green_wave | 13.47s | 14.60s | **+1.13s ± 0.13s** |
| max_pressure | 26.38s | 27.26s | **+0.87s ± 3.57s** |
| idqn | 16.56s | 17.05s | **+0.49s ± 0.21s** |

(`idqn`'s no-incident baseline is read per-seed from its own no-incident
`corridor_peak` tripinfo XMLs — SP5 checkpoints,
`logs/eval_idqn_corridor_peak_lam05_seed{42,43,44}_mg10_s100000_tripinfo.xml`
— the same seed-matching discipline used for `green_wave`/`max_pressure`
above; idqn has no row in `corridor_sweep.csv`, which only holds the non-RL
baselines, per that file's own design. Per-seed: 16.952s/16.490s/16.244s.)

By mean Δ, IDQN's cost (+0.49s) is smallest — smaller than both
`green_wave`'s (+1.13s) and `max_pressure`'s (+0.87s). Per the spec's
decision rule, that is nominally the interesting case: not just "beats a
blind plan," but "beats the other reactive controller too."

**That headline number needs more scrutiny than a mean ± sd table gives
it, because `max_pressure`'s Δ is not one number — it's two behaviors
wearing one mean.**

### max_pressure's per-seed cost is not consistent — the sign flips

| seed | no-incident (corridor_sweep.csv) | incident | Δ | Δ % |
|---|---|---|---|---|
| 42 | 29.24s | 26.67s | **−2.57s** | −8.8% |
| 43 | 21.89s | 26.46s | **+4.57s** | +20.9% |
| 44 | 28.02s | 28.64s | +0.62s | +2.2% |

Seed 42 gets *faster* once the lane closes — not within-noise flat, a real
8.8% improvement, on essentially identical demand and trip counts (2941
incident vs 2936 no-incident trips; this is not a "fewer vehicles" or
"episode aborted early" artifact). Seed 43 gets 21% worse. Seed 44 is
roughly flat. Averaging these into "+0.87 ± 3.57" is technically correct
but hides that `max_pressure`'s response to this incident is *bidirectional
and seed-dependent*, not "consistently mediocre." `green_wave`, for
contrast, is tight and consistent across all three seeds (+7.7% to +9.5%,
sd 0.13s) — a fixed plan degrades predictably under a predictable-shaped
disruption, which is exactly what you'd expect from a controller with no
ability to see or react to the closure at all.

Two things are worth separating here, because they point in different
directions:

1. **The no-incident bimodality is real and independently confirmed**
   (10-seed direct measurement, not an artifact of either defect — the
   no-incident numbers in `corridor_sweep.csv` predate and are untouched by
   both fixes). Seed 43's no-incident baseline, 21.89s, is the second-lowest
   of all 10 seeds at `min_green=10` — it was already an unusually fast run
   before the incident ever entered the picture. Part of its "+20.9%"
   incident cost is therefore "started from an unusually good number," not
   "the incident hit this seed unusually hard" in absolute terms (26.46s
   incident is close to the other two seeds' incident numbers, 26.67s and
   28.64s).
2. **How the incident interacts with that bimodality changed once the
   closure was real.** Under the buggy partial closure, seed 43 was barely
   touched (stayed near its own fast baseline) while 42/44 behaved
   normally — the pattern an earlier investigation (on that buggy data)
   characterized as hysteresis, a seed getting knocked from a good regime
   to a bad one and never recovering. That investigation's mechanism trace
   was done carefully and its conclusion — genuine effect, not a bug, at
   the code that existed then — was correct for the code that existed
   then. But that code only blocked 15.5% of demand. **Under the corrected
   full closure, the three seeds' incident-period delays converge toward a
   similar range (26.5-28.6s) regardless of their very different
   no-incident starting points (21.9-29.2s)** — seed 42 comes *down* to
   meet that range, seed 43 goes *up* to meet it. That is a different, and
   arguably more mechanistically sensible, story than one-way hysteresis:
   a full lane closure forces a specific traffic pattern on this corridor
   that `max_pressure`'s greedy phase selection responds to in a way that's
   fairly insensitive to which pre-incident regime the seed happened to be
   in. This document is not attempting to fully explain *why* seed 42
   specifically improves — that would need the same kind of phase-trace
   investigation the earlier (now-superseded) one did, which is out of
   scope for this eval — but the honest empirical finding is: **the
   earlier bimodality/hysteresis narrative does not carry over to the
   corrected, fully-closed incident**, and any future work on this
   scenario should treat that investigation as describing the old buggy
   code's behavior, not this one's.

### idqn's cost is smallest, but starts from a much worse absolute number

idqn's Δ (+0.49s ± 0.21s) has the smallest mean of the three, and a much
tighter sd than `max_pressure`'s (0.21s vs 3.57s) — though `green_wave`'s sd
(0.13s) is still the tightest of the three, which comes with a larger mean.
Per-seed: 17.666s/16.784s/16.696s incident vs each seed's own no-incident
number (16.952s/16.490s/16.244s) — deltas of +0.714s, +0.294s, +0.452s. No
sign-flipping, no seed dominates the mean the way `max_pressure`'s seed 43
does.

But idqn's no-incident baseline (16.56s) is already 3.09s worse than
`green_wave`'s (13.47s, SP5's own finding, unchanged by this eval) and its
incident-period number (17.05s) is still worse in absolute terms than
`green_wave`'s incident-period number (14.60s). IDQN "costs less" to
disrupt only because it was already worse to begin with and has less room
left to lose — the same ceiling-effect caveat that applies whenever you
compare deltas across controllers with very different starting baselines.
A controller that is already bad cannot get *much* worse.

## Verdict: does this change the consolidation recommendation?

**No.** `docs/HANDOFF_2026-08-21.md`'s recommendation — consolidate on
`green_wave`, a competently-timed fixed plan beats both reactive and
learned control at this network scale — stands. Three things keep this
from being the "learning earns something under disruption" result the
spec's decision rule was built to detect:

1. **n=3 is thin, and the plan's own design accepts that** (spec's own
   disclosed limitation, same caveat SP4-SP6 all carried). idqn's smallest-Δ
   result is not implausible, but it rests on 3 episodes per controller
   and one seed (`max_pressure` seed 43) that is itself an outlier on the
   no-incident side.
2. **The effect sizes are small relative to what's already established.**
   The largest Δ in this table is 1.13 seconds, against `green_wave`'s own
   ~13.5s baseline delay — roughly an 8% degradation from the worst-hit
   controller under a 15-minute, one-lane closure. None of these deltas
   are large enough, on 3 seeds, to be confident they'd survive a wider
   seed set, and `max_pressure`'s bidirectional per-seed result is a
   concrete demonstration of how much a single seed can move a 3-seed mean.
3. **IDQN is still worse in every absolute sense.** Its no-incident delay
   is worse than `green_wave`'s; its incident-period delay is worse than
   `green_wave`'s. The smallest-Δ result says IDQN degrades proportionally
   less from an already-worse starting point — interesting, and worth the
   follow-up the spec names (incident-aware retraining), but not a reason
   to prefer IDQN over `green_wave` for this corridor as it stands today.

This is not a clean third disconfirming replication the way SP6's
demand-shift result was (SP6 found IDQN generalizing to two of three shift
types, with an honest asymmetric caveat on the third) — this result is
genuinely more ambiguous, largely because `max_pressure`'s response to the
incident turned out to be seed-dependent in a way none of SP4-SP6's
scenarios exercised. The one thing this eval adds cleanly to the project's
picture: `green_wave` degrades predictably under disruption (tight,
consistent, small Δ across seeds) while both reactive controllers
(`max_pressure` explicitly, `idqn` less dramatically) show more per-seed
variance in how they respond — consistent with reactive control's greater
sensitivity to the specific state it reacts to, for better or worse
depending on the seed.

## What this doesn't answer

- Whether IDQN's smaller Δ would hold up at n=10 — the plan scoped this
  eval to 3 seeds (`analysis/incident_compare.py`'s `SEEDS = (42, 43, 44)`
  constant), matching SP4-SP6's convention; widening it was considered and
  explicitly deferred (not mandatory per the plan, and this eval's own
  ledger records that ruling) rather than done ad hoc after seeing a
  seed-dependent result.
- *Why* `max_pressure` seed 42 specifically improves under the full
  closure while seed 43 worsens — that would need the same kind of
  phase-dominance trace the earlier (superseded) investigation did for the
  buggy code, applied fresh to the corrected closure. Not attempted here.
- IDQN never saw an incident during training (deliberate, per spec scope)
  — this is a zero-shot generalization result, not a ceiling on what
  incident-aware retraining could achieve. That remains the natural
  follow-up if anyone wants to keep pushing on this scenario, same as the
  spec's own deferred-decisions list already names.
- Only `corridor_peak` was tested (spec's own scope: one variable under
  test at a time, no stacking with the demand-shift scenarios from SP6).
- Every Δ reported above is a **whole-episode** mean (`trip_time_loss_mean`
  over all trips in the 3600s episode), but the incident only lasts 900s —
  a quarter of the episode. Only roughly a quarter of trips are exposed to
  the closure at all, so the reported Δ is diluted by roughly that same
  ~4x ratio: the true per-affected-trip degradation among trips that
  actually crossed the corridor during the closure is correspondingly
  larger than the headline number suggests. The spec's own §Scope section
  asked for both an incident-window number and a whole-episode number in
  the comparison table; only the whole-episode number shipped here — that
  in-window column was never produced and would need per-trip
  depart/arrival filtering logic this plan didn't build.
