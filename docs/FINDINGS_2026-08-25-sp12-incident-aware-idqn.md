# SP12 Findings: does incident-aware training reduce IDQN's incident cost?

## The question

SP7 found that a zero-shot `corridor_peak` IDQN policy — never shown the
SP7 lane-closure incident during training — already had the smallest
incident cost of the three controllers compared (whole-episode Δ +0.49s ±
0.21s, sharper in-window Δ +1.13s ± 0.61s, both n=3,
`docs/FINDINGS_2026-08-22-sp7-corridor-incident.md`,
`docs/FINDINGS_2026-08-22-sp8-irregular-spacing.md`). SP7's own scope
section deferred incident-aware retraining as "a natural, separately-scoped
follow-up" if the zero-shot result showed promise. This sub-project runs
that follow-up: train IDQN with the incident present in a fraction of
training episodes, then compare its own incident cost to the zero-shot
baseline above.

## Method

Same corridor, same `corridor_peak` demand, same 3 seeds (42/43/44), same
100k-step budget, same hyperparameters (`train_corridor_dqn._hp()`) as
SP5/SP7's zero-shot checkpoints — only the training curriculum changes.

`train_corridor_dqn.train()` gained `incident`/`incident_prob` params (this
session, plumbing already sketched by a prior session and reconciled here
against SP8's `net_file` param, which had landed on `main` in the interim —
see Notes below). Each training episode independently includes SP7's fixed
lane closure (`corridor_baseline.INCIDENT`) with probability
`incident_prob`, decided by the same seeded `rng` the training loop already
uses for epsilon-greedy exploration.

**`incident_prob=0.5`** — not dictated by any spec, chosen here as the
midpoint of "sees both dynamics" without designing a curriculum-search
sweep into a single follow-up sub-project. A different value could plausibly
shift the result; not explored.

Checkpoints saved under a `incaware` filename variant
(`_tag()`'s new `variant` param) so they never collide with the existing
plain `corridor_peak` checkpoints. Evaluated the same way SP7 did: one
greedy-policy episode per seed, with and without the incident, tripinfo
XML written for both. In-window vs whole-episode delay computed by
`analysis/incident_window_delta.py`, a generalized from-scratch recreation
of the one-off script an earlier (uncommitted) session used to produce
SP8's in-window numbers — see that file's own docstring for why it's a
recreation rather than a reuse.

## Results

### Incident cost: incident-aware vs zero-shot IDQN

| measurement | zero-shot idqn (SP7/SP8) | incident-aware idqn (SP12) | change |
|---|---|---|---|
| whole-episode Δ | +0.49s ± 0.21s | **+0.33s ± 0.05s** | −33% |
| in-window Δ | +1.13s ± 0.61s | **+1.30s ± 0.40s** | +15% |

Per-seed incident-aware deltas (seed42/43/44): whole-episode 0.352s /
0.359s / 0.268s; in-window 1.270s / 1.712s / 0.907s.

The two measurements disagree on direction. Whole-episode Δ looks better
under incident-aware training, but this number is known (SP7/SP8's own
finding) to be diluted ~4x by trips never exposed to the closure at all —
it is not the measurement that isolates what the incident itself costs.
In-window Δ, the sharper number, is flat-to-slightly-worse under
incident-aware training, well within the overlap of both conditions' own
seed-to-seed spread (n=3 each). **No reliable improvement on the
measurement that actually isolates the incident's cost.**

### Cost: does incident-aware training hurt ordinary (no-incident) performance?

| | zero-shot idqn no-incident delay/trip | incident-aware idqn no-incident delay/trip |
|---|---|---|
| seed42 | 16.952s | 16.886s |
| seed43 | 16.490s | 17.327s |
| seed44 | 16.244s | 17.216s |
| mean | 16.562s | **17.143s** |

+0.58s (+3.5%) worse on ordinary demand — the same curriculum-breadth
tradeoff `docs/FINDINGS_2026-08-22-sp11-offpeak-curriculum.md` found for
the offpeak-curriculum experiment: widening what a single policy has to
handle costs a little of its focus on the common case, whether the
widening is a demand-magnitude curriculum (SP11) or an incident-presence
curriculum (this one).

## Verdict: does this change the consolidation recommendation?

**No.** `green_wave` consolidation stands. This result's shape matches
SP11's, not a clean win: the sharper in-window incident-cost measurement
shows no reliable improvement, while ordinary-demand performance measurably
degrades. Zero-shot IDQN was already the smallest-Δ controller in SP7's
comparison; incident-aware training does not demonstrably improve on that,
and it is not free.

## What this doesn't answer

- Whether a different `incident_prob` (this run used 0.5, undictated by any
  spec) would land differently — not swept.
- SP7's own disclosed limitation carries over unchanged: this curriculum
  reuses the exact same fixed, deterministic incident spec on every
  incident episode (same edge, same start/duration) — it does not
  randomize timing/location/severity, so a policy trained this way could in
  principle be keying off when the incident always starts rather than
  reacting to the traffic pattern it causes. Not addressed here.
- Whether n=3 seeds is enough to trust the in-window sd (0.40s on a 1.30s
  mean) — same thin-n caveat SP7/SP8 already carried for the zero-shot
  numbers this compares against.

## Notes on this session's setup

This worktree's branch had based off `docs/HANDOFF_2026-08-22.md`'s SP6+SP7
handoff commit, before SP8 landed `net_file` plumbing on `main`. A prior
session had already written this sub-project's `train()`/`evaluate()`
plumbing (uncommitted) against that stale base. This session rebased that
work onto `main`'s SP8 commit, resolving one conflict (both changes touched
`_eval_out_stem`/`evaluate`'s signatures — SP8 adding `net_file`, this
sub-project adding `variant`; kept both). It also renamed an
uncommitted `analysis/incident_window_compare.py` this sub-project had
written to `analysis/incident_window_delta.py` — `main`'s SP8 commit had
independently added its own script under the original name for a different,
narrower purpose (SP7's fixed 3-controller comparison); this sub-project's
version is a generalized CLI over arbitrary tripinfo paths, needed to check
an incident-aware checkpoint's own eval output, not only SP7's plain
`corridor_peak` checkpoints — so both scripts now coexist under distinct
names rather than one silently overwriting the other. 3 stray scratch
checkpoints from that prior session's plumbing debugging (tagged
`corridor_offpeak/seed0/5200-step` and `corridor_peak/seed1/600-step`,
matching neither this brief's scenario/seeds/step-count) were deleted
before training the real checkpoints this doc reports on.
