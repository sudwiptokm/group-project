# SP7 — corridor mid-episode incident/blockage (design)

**Date:** 2026-08-22
**Status:** Design approved, pending spec review
**Sub-project:** SP7 of the multi-intersection coordinated-MARL thesis extension
**Depends on:** SP1 (`make_corridor_env`, `green_wave`/`max_pressure` baselines
in `corridor_baseline.py`), SP5 (IDQN checkpoints trained on `corridor_peak`,
`train_corridor_dqn.py`), SP6 (`docs/superpowers/specs/2026-08-22-sp6-idqn-demand-shift-design.md`
— same eval-only, zero-shot philosophy, same corridor checkpoints).

## Context

Every fixed-vs-learned comparison run so far (SP4, SP5, SP6) has tested a
stationary-or-slowly-shifting demand *shape* the fixed plan happens to be
blind to by construction (`green_wave` derives its plan purely from
`min_green`/`yellow_time`/`delta_time` and signal geometry — see SP6's
context section). None of them have tested the case that most directly
argues for reactive or learned control: an acute, unplanned capacity
change mid-episode that a pretimed plan structurally cannot respond to,
because it isn't a function of anything happening in the network. This is
the scenario flagged as an open thread in both `docs/HANDOFF_2026-08-18.md`
and `docs/HANDOFF_2026-08-21.md`, and never designed until now.

The obvious risk, raised and discussed before this spec was written: a
positive result here could just show "any reactive controller beats a
blind pretimed plan when the road closes," which is not a new finding —
`max_pressure` (reactive, non-learned) is already in every comparison
table. The design below keeps `max_pressure` in the incident comparison
specifically so a result can be attributed to *learning* rather than to
*reacting in general*.

## Scope of SP7

**In scope:**
- A mid-episode lane-closure mechanism added at the `make_corridor_env`
  layer, so it applies identically to whichever controller runs through it
  — no per-controller special-casing.
- One incident: 1 of 2 lanes on the `C1_C2` arterial edge (eastbound,
  C1→C2) closed for 900s (15 minutes) starting at t=1800s, within the
  3600s episode. `corridor_peak` demand only.
- A new incident-eval sweep for `green_wave` and `max_pressure`
  (`corridor_baseline.py`), seeds 42/43/44 — paired against the same seeds
  IDQN's existing checkpoints were trained on, so all three controllers
  face the same demand draw plus the same incident.
- Zero-shot IDQN eval, reusing the `corridor_peak`-trained checkpoints
  SP5/SP6 already have on disk. No retraining.
- A comparison table reporting each controller's incident-window and
  whole-episode delay, and — the actual metric of interest — the *cost*
  of the incident: each controller's delay under the incident minus its
  own no-incident `corridor_peak` number (from `analysis/corridor_sweep.csv`
  for green_wave/max_pressure, from SP5 for IDQN).
- A findings doc reporting the outcome either way.

**Explicitly out of scope (deferred):**
- Retraining IDQN with incidents present during training. This spec tests
  whether a policy trained only on clean demand still reacts sensibly to
  a disruption it never saw — the same zero-shot philosophy SP6 uses, for
  the same reason (cheap, and doesn't conflate "does the existing policy
  generalize" with "can IDQN learn incident-specific behavior given the
  chance"). If the zero-shot result shows real promise, incident-aware
  retraining is a natural, separately-scoped follow-up.
- IPPO. Same reason as SP6: no surviving checkpoints on disk to test
  zero-shot without a fresh training run first.
- Stacking the incident on `corridor_tidal` or `corridor_skew`. Those
  scenarios already carry their own structural demand shift; adding an
  incident on top would confound two variables in one result, the same
  discipline `make_scenarios.py` documents for keeping `corridor_skew`
  isolated from a tidal reversal.
- A full blockage (both lanes) or multiple incidents per episode — a
  partial, single closure is the design chosen after the severity
  discussion; a harsher or repeated version is a possible follow-up if
  this one proves inconclusive (§ Open decisions).
- Randomizing incident timing/location — this spec uses one fixed,
  deterministic incident so every controller and seed faces literally the
  same event; randomization only matters once/if an incident-aware
  training follow-up exists to consume it.

**Success criterion for SP7:** all three controllers complete paired eval
runs (seeds 42/43/44) under the identical incident, producing trip-level
CSVs through the existing `SafetyLoggingEnv`/`save_csv` path, and the
comparison table reports each controller's incident-cost (Δ vs its own
no-incident number) — whichever direction it points, and however it
compares to `max_pressure`'s own incident-cost.

## Design

### 1. Incident mechanism

`SafetyLoggingEnv.__init__` gains an optional `incident` parameter: a
tuple `(edge_id, lane_index, start_s, duration_s)`, default `None` (no
incident — every existing call site is unaffected). `_sumo_step()`
(already overridden here for the safety-window accumulator) checks
`self.sumo.simulation.getTime()` against the window each step:

- At `start_s`: `self.sumo.lane.setDisallowed(f"{edge_id}_{lane_index}",
  ["passenger"])`. This is a TraCI vehicle-class restriction, not a
  teleport — vehicles already on the lane finish their trip normally,
  matching how a real lane closure behaves; only new routing decisions
  avoid it.
- At `start_s + duration_s`: `self.sumo.lane.setAllowed(lane_id, [])`
  (empty allow-list restores the lane's default permissions) to reopen
  the lane.
- A boolean instance flag tracks whether the closure has already been
  applied/reverted this episode, so the check is idempotent against being
  evaluated every step rather than once.

`make_corridor_env(...)` gains a passthrough `incident=None` parameter
forwarded to `SafetyLoggingEnv`. `corridor_baseline.py --incident` and
`train_corridor_dqn.py --eval --incident` (both currently have no such
flag) each add it, defaulting to `None`, wired straight through to
`make_corridor_env`. No other call site changes.

### 2. Where and when

`C1_C2`, lane index 0 of 2, closed 1800s–2700s. Chosen because it sits
squarely inside the segment a green wave's offsets are actually
coordinating (`corridor_control.green_wave_offsets` computes C1→C2→C3
timing), so a closure there directly tests whether the offset-based plan
can do anything about a capacity drop in the middle of its own coordinated
segment (it structurally cannot — the offsets are fixed at simulation
start) versus whether `max_pressure` or IDQN's per-step, per-signal
decisions can shift green time toward the constrained direction. Midpoint
timing (t=1800s, the same switch point `corridor_tidal` already uses)
gives 1800s of clean before/after episode on either side to establish and
recover from the disruption within the same run.

### 3. Eval runs

6 new baseline runs (`green_wave`/`max_pressure` × seeds 42/43/44,
`corridor_peak`, `--incident`) via `corridor_baseline.py`, plus 3 IDQN
zero-shot eval runs (seeds 42/43/44, loading the existing `corridor_peak`
checkpoints, `--incident` flag added to `train_corridor_dqn.py`'s eval
path). 9 runs total, all cheap (single-episode eval, no training).

### 4. Comparison table

A small new script, `analysis/incident_compare.py`, reduces the 9 new
eval CSVs and joins them against each controller's existing no-incident
`corridor_peak` number:

| controller | no-incident delay/trip | incident delay/trip | incident cost (Δ) |
|---|---:|---:|---:|
| green_wave | 13.47 ± 0.04s (existing) | (new) | (new) |
| max_pressure | (existing) | (new) | (new) |
| idqn (zero-shot) | 16.56 ± 0.36s (SP5) | (new) | (new) |

### 5. Decision rule

Compare incident cost (Δ), not raw delay — IDQN already starts from a
higher no-incident baseline (SP5's own +3.09s loss to green_wave), so the
question is which controller's delay grows *least* when the network is
disrupted, not which has the lowest absolute number.

- **IDQN's Δ is smaller than green_wave's Δ** — evidence reactive/learned
  control earns something specifically under disruption, the result the
  original hypothesis was reaching for. Check against `max_pressure`'s own
  Δ next: if `max_pressure`'s Δ is comparably small, the result is "any
  reactive controller beats a blind plan under disruption" (expected, not
  new). If IDQN's Δ is smaller than *both* green_wave's and
  `max_pressure`'s, that's the more interesting claim — learning adds
  something reacting alone doesn't.
- **IDQN's Δ is comparable to or larger than green_wave's** — the zero-shot
  policy doesn't handle the disruption better than a plan that can't see
  it at all. A legitimate negative finding; the incident-aware-retraining
  follow-up (§ scope) becomes the natural next question rather than
  something this spec needs to answer.

No compute gate — 9 single-episode eval runs is a small, fixed commitment,
same reasoning as SP6.

## Components & boundaries

| Component | Responsibility | Depends on |
|-----------|----------------|------------|
| `env_common.SafetyLoggingEnv` | Applies/reverts the lane closure at the configured sim-time, via the live TraCI connection (`self.sumo`) | `traci` (already a `SumoEnvironment` dependency) |
| `env_common.make_corridor_env` | Passthrough `incident` param to `SafetyLoggingEnv` | — |
| `corridor_baseline.py` | `--incident` flag, new incident-eval runs for green_wave/max_pressure | `make_corridor_env` |
| `train_corridor_dqn.py` | `--incident` flag on the eval path, zero-shot IDQN under the incident | `make_corridor_env`, existing SP5 checkpoints |
| `analysis/incident_compare.py` | Reduces the 9 new eval CSVs, joins against existing no-incident numbers, reports Δ per controller | `analysis/tripinfo`, `analysis/corridor_sweep.csv`, SP5's findings |

## Risks & mitigations

- **A one-lane closure might not bind** — `C1_C2` carries 2 lanes at
  corridor_peak's arterial rate; closing one halves capacity on that
  segment for 15 minutes. If the resulting queue doesn't measurably
  exceed noise, that's itself informative (report it, don't silently
  retry with a harsher incident) rather than something to tune until a
  effect appears.
- **Vehicles mid-trip on the closed lane at `start_s`** — `setDisallowed`
  only affects new routing choices, not vehicles already committed to the
  lane; this is realistic (a real closure doesn't teleport cars already
  in it) and consistent with how SUMO's own incident-modeling examples use
  this call.
- **Confounding "reactive beats blind" with "learning beats reactive"** —
  mitigated by keeping `max_pressure` in the comparison (§5), the same
  discipline SP6 used to separate "IDQN-specific" effects from "this
  scenario is harder for everyone."
- **n=3 seeds** — same small-sample caveat as SP6; reported with spread,
  not a pass/fail claim.
- **IDQN never saw an incident during training** — deliberate (§ scope);
  the zero-shot result is evidence about generalization to disruption, not
  a ceiling on what IDQN could do with incident-aware training.

## Open decisions deferred to later sub-projects

- Incident-aware IDQN retraining (randomized timing/location/severity in
  the training curriculum), if the zero-shot result here is promising
  enough to justify the compute.
- A harsher incident (full blockage) or a second incident later in the
  episode, if this one's Δ turns out too small to be informative (§
  Risks).
- An IPPO equivalent, once/if `corridor_peak` IPPO checkpoints exist on
  disk again (same blocker SP6 already flagged).
