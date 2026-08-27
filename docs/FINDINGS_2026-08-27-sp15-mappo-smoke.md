# SP15 Findings: does centralized-critic coordination (MAPPO) beat independent agents (IPPO)?

## The question

The original SP3 design spec (`docs/superpowers/specs/2026-08-02-sp3-mappo-coordination-design.md`,
never merged to main) states the thesis claim plainly: **"explicit coordination
(CTDE) beats independent agents and classical baselines."** That claim was
never tested — SP2's IPPO training loop shipped to main, but the MAPPO
extension (joint-state critic) only ever existed on a stale, never-merged
branch that predates almost all of SP4-SP14. This is a directional smoke
test of that original claim, not the full rigor SP5's brief always deferred
it to.

## Method

MAPPO's design is: IPPO and MAPPO are the *same* PPO training loop
(`train_corridor.py`, `ppo_core.py`), differing only in what the critic sees
— an agent's own local observation (IPPO) or the joint observation of all 3
agents, concatenated in `ts_ids` order (MAPPO). The actor is local-obs-only
either way; only the critic's input differs, so any performance gap is
attributable to coordination and nothing else (the spec's own isolation
argument). That extension was ported forward from the stale branch onto
current main's `train_corridor.py` this session (`build_states`, a
`centralized` flag threaded through `collect_rollout`/`update`/`train`, the
`--algo {ippo,mappo}` CLI) — see the commit history for the port itself; this
doc covers only the resulting experiment.

**n=1 seed (42), `corridor_peak`, λ=0.5, min_green=10, 100k steps.**
Deliberately a smoke test, not the full rigor every other sub-project's
headline finding carries (SP9's n=10, SP13's 8-point sweep) — the point here
is a directional answer before committing to that cost. PPO hyperparameters
are the project's existing single-intersection-tuned values
(`train_corridor._HP`), reused unmodified for both IPPO and MAPPO — the
spec's own disclosed risk (§"Critic under-fitting the wider input"): they
were tuned for a 19-dim critic, not MAPPO's 57-dim joint one.

Zero-shot evaluated on both geometries, same posture as every other
controller in this project: **regular** (`corridor.net.xml`, what both
checkpoints trained on) and **irregular** (`corridor_irregular.net.xml`,
SP8's 578m/78m asymmetric net — the one where `idqn` beats `green_wave`).

## Results

Delay/trip in seconds per completed trip, seed 42, `corridor_peak`:

| controller | regular | irregular |
|---|---:|---:|
| green_wave | 13.44s | 19.44s |
| max_pressure | 29.24s | 21.14s |
| idqn | 16.95s | 18.41s |
| **ippo** | **16.91s** | **19.14s** |
| **mappo** | **18.26s** | **20.08s** |

## Verdict: no evidence MAPPO helps — at this scale, it's worse

**MAPPO underperforms IPPO on both geometries tested** (18.26s vs 16.91s
regular, +1.35s; 20.08s vs 19.14s irregular, +0.94s). The centralized critic
did not produce a better policy than the independent one — if anything, the
opposite, directionally answering the spec's original question: no, this
smoke test finds no evidence that explicit CTDE coordination beats
independent agents here.

One specific consequence worth flagging: on the irregular net, `idqn` beats
`green_wave` (18.41s vs 19.44s, the established SP8 flip) and `ippo` *also*
modestly beats `green_wave` (19.14s vs 19.44s, a much narrower 0.30s margin
than idqn's 1.03s) — but **`mappo` does not** (20.08s, worse than
`green_wave`'s 19.44s). Whatever lets independent agents (idqn, and to a
smaller extent ippo) exploit irregular spacing better than a fixed-offset
plan, centralizing the critic erased it rather than amplifying it.

Consistent with the spec's own disclosed risk: HPs tuned for a 19-dim critic
were reused unmodified for the 57-dim joint critic, no retuning attempted
(explicitly out of scope for MAPPO's own SP3, deferred to a hypothetical
SP5-level sweep that never happened either). A wider critic wanting more
capacity or a different learning rate than what a much narrower one was
tuned for is a plausible, undiscriminated-from-real-coordination-failure
explanation for the gap — this smoke test cannot tell "coordination doesn't
help here" apart from "the HPs don't suit the wider critic," and doesn't
attempt to.

Neither `ippo` nor `mappo` beats `green_wave` on the regular net (13.44s vs
16.91s/18.26s) — consistent with every other learned controller's standing
result on regular spacing.

## What this doesn't answer

- **n=1 seed.** Every number above is a single episode on a single seed.
  The gap sizes (0.94-1.35s) are on the same order as this project's typical
  seed-to-seed noise elsewhere (SP9 found σ≈0.38s at n=10 for a
  comparably-sized irregular-net effect at n=3; single-seed sd is unknown
  here). This result should be read as "no encouraging signal to justify the
  cost of a full sweep," not "MAPPO is proven worse."
- **HPs never retuned for the wider critic** — see above; a real, live
  confound this smoke test cannot separate from a genuine coordination
  failure.
- **Only 2 geometries** (regular, SP8's original irregular net) — not
  cross-referenced against SP13's finding that the irregular-spacing
  advantage is a bounded band, not universal; whether ippo/mappo's ranking
  holds, worsens, or reverses across SP13's asymmetry-ratio sweep is
  untested.
- **`compare.py` was not enrolled** for `mappo` — this eval bypassed it
  (bespoke script), same reason SP8/SP10/SP13/SP14 all bypass it for a
  non-standard eval dimension.

## Recommendation

Given no encouraging signal at n=1 and a live, unresolved HP-retuning
confound, escalating to the full n=10-seed rigor originally scoped for this
sub-project is not warranted by this result alone. If MAPPO stays in the
thesis, the honest framing is: the coordination mechanism was built,
integrated, and smoke-tested against the project's own original claim, and
the smoke test did not support it — a real, if negative, answer to the
question the stale branch's spec posed, cheaper than the full sweep would
have cost to reach the same qualitative conclusion.
