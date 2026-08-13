# Results writeup — RL traffic-signal control vs fixed-time baseline

**Peak half rewritten 2026-08-12** after the audit in
[`FINDINGS_2026-08-12.md`](FINDINGS_2026-08-12.md). The earlier peak headline
(*DQN and A2C cut mean waiting ~24%*) is withdrawn — see
[Withdrawn](#the-withdrawn-peak-claim). The off-peak half is unchanged and still
stands.

Two metrics appear below, and they are not interchangeable:

| metric | what it measures | where it comes from |
|---|---|---|
| **delay per completed trip** (`trip_time_loss_mean`, s) | how much longer each *finished* trip took than a free-flow run of the same route | SUMO `tripinfo`, via `analysis/tripinfo.py` |
| in-network mean wait (`system_mean_waiting_time`, s) | `getWaitingTime()` averaged over vehicles *still in the network*, then time-averaged | sumo-rl step CSV |

Delay per completed trip is the ranking key. The in-network average is
survivorship-biased — a controller that strands traffic is scored on the
vehicles it did not strand, and under deadlock the number is just a clock. On
the same peak runs the two disagree by a factor of 4 (14.97 s in-network vs
66.43 s waiting per completed trip), which is that bias measured directly.

λ is the safety weight in the reward `diff_waiting − λ·safety`. Only λ = 0.5 has
ever been run; the λ ablation does not exist.

---

## Peak demand (oversaturated intersection)

### The result

A static plan with a long green beats every learned policy this project has
produced, and it does not need to be tuned to do so: performance is **flat
across 45–90 s of green**, so any plan in that band works.

Static green-duration sweep, peak demand (1.5×), seeds 42–46, 3600 s episodes,
`--time-to-teleport 300` — `analysis/static_timing.py`, raw rows in
`analysis/static_sweep.csv`, figure in `results/static_timing_peak.png`:

| green (s) | delay per completed trip (s) | trips completed | vs 60 s, paired | in-network wait (s) |
|---|---:|---:|---:|---:|
| 10 | 298.6 ± 75.7 | 76.2% | +206.9, loses 5/5 | 29.6 |
| 20 | 384.7 ± 39.4 | 73.2% | +292.9, loses 5/5 | 33.6 |
| 30 | 257.4 ± 133.6 | 87.2% | +165.6, loses 4/5 | 22.6 |
| **45** | **104.9 ± 38.2** | 95.7% | +13.1, wins 3/5 | 13.6 |
| **60** | **91.8 ± 19.9** | 94.3% | — | 15.0 |
| **75** | **91.3 ± 6.6** | 95.1% | −0.4, wins 1/5 | 16.0 |
| **90** | **103.4 ± 18.1** | 94.5% | +11.6, wins 1/5 | 19.4 |
| 120 | 110.8 ± 6.0 | 93.8% | +19.0, wins 1/5 | 24.5 |

Bold rows are the plateau. "vs 60 s, paired" is the mean per-seed difference
against the 60 s plan on the *same* demand seed, with the number of seeds on
which that green wins.

**Read the plateau as a plateau, not an optimum.** Within 45–90 s the paired
differences are +13.1 s, −0.4 s and +11.6 s against a seed-to-seed spread of
roughly 30 s — indistinguishable. Outside it the ranking is unambiguous: every
green below 45 s loses on 4 or 5 seeds out of 5, by 165–293 s.

The 10 s and 20 s plans also **fail to clear the demand at all** — 76% and 73%
of trips complete, against ~95% everywhere on the plateau. The Stage-1 "fixed-time"
baseline was not merely slow; it left a quarter of the traffic unserved, which is
exactly the population the in-network waiting-time metric then declined to count. The sample minimum is 75 s, but it loses to 60 s on four of the five seeds
and its whole advantage is one outlier draw (seed 43) — calling it "the optimum"
would be reading seed noise as a policy effect, which is defect 2 in miniature.
The reported baseline uses 60 s as a mid-plateau round number.

The flatness strengthens the finding rather than weakening it: the static plan
that beats the learned policies required no tuning skill to find, so the result
cannot be dismissed as an unfairly optimised baseline.

Fixed-time at 60 s, in full, seeds 42–46 (`baseline.py --scenario peak
--green 60`, aggregated by `compare.py`):

| metric | value |
|---|---|
| delay per completed trip | **91.8 ± 19.9 s** |
| trips completed | 4076 ± 137 |
| completion rate vs demand | 0.943 ± 0.032 |
| in-network mean wait | 14.97 ± 3.55 s |

There is no comparable RL row. Every trained peak model predates the safety fix
(`1a678e51`), none was retrained, and the two 20k-step retraining attempts
produced no learning (FINDINGS §"Training attempted"). The honest statement is
**no valid RL measurement at peak exists**, not that RL scored badly. What can
be said is that the learned policies from Stage 1 sat at 20–33 s in-network mean
wait against a 60 s static plan's 11.5 s on the same 1200 s episodes — a factor
of 2–3 the wrong way.

### Why: lost time to amber

`yellow_time` is 3 s, so every phase switch spends 3 s serving nobody:

| green | share of the cycle lost to amber |
|-------|---------------------------------|
| 10 s | 3/13 = **23%** |
| 20 s | 3/23 = 13% |
| 60 s | 3/63 = **4.8%** |

The agent decides every 5 s with `min_green` = 10 s, so it lives exactly where
switching is cheap to attempt and ruinous to pay for. `diff_waiting_time` rewards
the immediate queue drop on the approach it just served and only registers the
lost-time cost several decisions later — credit assignment at this sample budget
cannot bridge that.

Two structural notes on the action space, both measured:

* `max_green` does not exist as a constraint. sumo-rl's `TrafficSignal` stores
  it and never reads it (`traffic_signal.py:77` is the only occurrence), so the
  `max_green=60` in `env_common.make_env` binds nothing. Greens beyond 60 s are
  reachable by a static plan and by a learned policy alike — hence the sweep
  above runs past 60 s rather than stopping at it.
* At a 2-phase isolated junction with permissive lefts, near-optimal control is
  close to "hold a long green and alternate". A learned controller has very
  little left to win. That is a result about the problem, not a training failure.

### Was there anything for an adaptive controller to win?

The section above asserts there is little left to win. That assertion is testable
without training anything, and it was tested. A static plan beating every learned
policy is consistent with two readings — our RL failed to find an adaptive policy
that exists, or no such policy exists here — and no amount of further training
separates them, because another null fits both.

A non-learning controller does separate them. `analysis/actuated.py` runs the
classic queue-actuated policy: each decision step, serve whichever green phase
has the largest PCU-weighted queue on the lanes it discharges, subject to
`min_green`. It has perfect queue information, no reward to misspecify, no credit
assignment problem and no sample budget. If it cannot beat the best static plan,
the headroom is not there.

Peak 1.5×, seeds 42–46, 3600 s episodes, teleport 300, paired against the static
60 s plan on the same seeds — `analysis/headroom.py`, rows in
`analysis/actuated_sweep.csv`:

| `min_green` (s) | delay per completed trip (s) | trips completed | vs static 60 s, paired | seeds beaten |
|---|---:|---:|---:|---:|
| **10** | 517.5 ± 208.4 | 2925 | +425.7 ± 217.5 | 0/5 |
| 20 | 337.0 ± 220.5 | 3455 | +245.3 ± 228.9 | 1/5 |
| 30 | 186.1 ± 108.0 | 3876 | +94.3 ± 115.3 | 1/5 |
| 45 | 161.9 ± 64.2 | 4022 | +70.2 ± 66.3 | 1/5 |
| **60** | **82.5 ± 10.1** | **4156** | **−9.3 ± 23.9** | **3/5** |
| 75 | 92.2 ± 0.9 | 4119 | +0.4 ± 20.8 | 1/5 |
| 90 | 118.7 ± 23.7 | 4038 | +26.9 ± 24.5 | 0/5 |

![Static plan vs queue-actuated controller at peak](../results/headroom_peak.png)

Both controllers on one axis (`analysis/plot_headroom.py`): the x is the same
constraint seen from two sides — the green a static plan holds, and the floor
below which the actuated controller's switch requests are ignored. Same metric,
same seeds. The argument is at the left-hand end, not the crossover.

**`min_green` is the binding constraint, not the algorithm.** At the 10 s floor
this project ran, a controller that cannot be accused of under-training is 5.6×
worse than a fixed plan and strands a quarter of the traffic — 2925 trips against
4076, and only 2008 on the worst seed. It requests 125–168 switches per episode
against 38–60 at a 75–90 s floor, and each one costs 3 s of amber. The sweep is
U-shaped and has turned by 90 s, so 60 s is an interior optimum, not "longer is
better".

**Read the 60 s row honestly.** −9.3 s is inside the noise: the paired difference
has an sd of 23.9 s over five seeds. The mean is not the finding. What is
resolvable is consistency:

| | delay sd | trips completed |
|---|---:|---:|
| static 60 s plan | 19.9 s | 3834–4162 (spread 328) |
| actuated, `min_green` 60 | 10.1 s | 4142–4177 (spread **35**) |

The static plan's bad draw is seed 43 (126.3 s, 3834 trips); the actuated
controller takes that same seed at 83.1 s and 4146 trips. The adaptive gain is
not a lower mean — it is not having a bad seed.

So the answer is neither reading cleanly. At `min_green` = 10 there was nothing
for a learned controller to find, and that is where this project spent its entire
training budget, which makes the peak null over-determined. At `min_green` = 60
there is something, but it is a ~10% variance reduction that a policy needing no
training already collects. **The reference a learned controller has to beat is
therefore the actuated controller at a matched floor (82.5 ± 10.1 s), not the
static plan** — and it has to beat it by enough to clear a 24 s paired sd.

### The withdrawn peak claim

Previously reported, and now withdrawn in full:

| Algorithm | Mean waiting (s) | ± std | vs fixed-time |
|---|---:|---:|---:|
| dqn | 1002.76 | 401.55 | −24.0% |
| a2c | 1003.33 | 86.14 | −24.0% |
| fixed-time | 1319.17 | — | baseline |
| ppo | 1356.53 | 45.03 | +2.8% |
| qrdqn | 1400.69 | 4.92 | +6.2% |

Six defects, five of them independent and any one sufficient to void the number:
the metric was a gridlock clock; the baseline ran on seed 0 while the agents ran
on seeds 42–46 (a 5.4× spread, seed 0 the worst draw); every model predates the
safety fix; the "±std over 5 seeds" was one policy's demand spread, not seed
variance; the gridlock came from sumo-rl's `time_to_teleport = -1`, not from the
demand; and the "fixed-time" baseline was a 10 s-green cycler, the *worst* point
of the sweep above. Full derivation and reproduction commands in
[`FINDINGS_2026-08-12.md`](FINDINGS_2026-08-12.md).

Paired against the baseline on its own seed, the direction reverses: dqn −23.9%
→ +56.9%, a2c → +67.6%, ppo → +118.4%, qrdqn → +123.3% (`analysis/paired.py`).

---

## Off-peak demand (light traffic)

Unchanged from the original writeup. Metric is in-network mean wait; these runs
predate tripinfo logging and their baseline is the 10 s cycler.

| Algorithm | Mean waiting (s) | ± std | Mean speed (m/s) |
|---|---:|---:|---:|
| fixed-time | 0.387 | — | 10.40 |
| dqn | 0.477 | 0.015 | 10.50 |
| ppo | 1.757 | 0.074 | 10.01 |
| qrdqn | 1.985 | 0.230 | 9.91 |
| a2c | 36.006 | 1.499 | 4.75 |

Off-peak is light: fixed-time is already near-optimal at 0.39 s, so no RL agent
beats it — the reportable result is that **all four hold traffic mobile**. dqn is
within a hair of the baseline; ppo/qrdqn add ~1–2 s; a2c is weakest at 36 s but
valid and mobile (4.75 m/s, tight across seeds), not the gridlock collapse it
previously showed. a2c is reported, not excluded.

This conclusion survives the audit. At 0.39 s there is no headroom for any
controller, so the ranking cannot be flipped by the metric defect (nothing is
stranded, so there is nobody for survivorship bias to hide), and it does not
depend on the reward the models were trained against.

Two caveats, neither of which changes the conclusion: these runs have no
completed-trip metrics, and their baseline is the 10 s cycler rather than a
tuned static plan — a better static plan would only widen the gap against RL.

### Why the earlier a2c/ppo/qrdqn off-peak numbers were dropped

The first cloud run produced byte-identical ~1122 s waiting / 0.76 m/s speed for
off-peak {a2c, ppo, qrdqn} — the signature of a constant-action gridlock collapse
(agent learns never to switch phase). Those runs were invalid and were re-run.
ppo and qrdqn were fixed by per-scenario hyperparameter tuning; a2c needed the
objective fix in the footnote below.

---

## Bottom line

- **Peak:** no valid RL result. A static plan anywhere in the 45–90 s green band
  is the standard to beat, and Stage 1's learned policies were 2–3× worse than
  it — a plan that needed no tuning to find. The reason is
  structural — amber lost time at a 2-phase junction — not a training-budget
  artefact.
- **The constraint that mattered was `min_green` = 10, not the algorithm.** A
  non-learning queue-actuated controller is 5.6× worse than the fixed plan at
  that floor and matches it at 60 s. So the peak null is over-determined: the
  whole training budget was spent in a region where no controller can win. At a
  60 s floor there *is* adaptive headroom, but it is a ~10% variance reduction,
  inside the seed noise on the mean, and collected by a controller that learns
  nothing.
- **Off-peak:** fixed-time is near-optimal; RL cannot improve on it; all four
  agents stay mobile; a2c is the weakest.
- **λ ablation:** never run. "Safety-aware" is in the title and not in the
  results.
- The methodology findings (six defects, the static-timing sweep, the amber
  mechanism, `max_green` being inert in sumo-rl) are the defensible output of the
  peak work, and three of them are defects in how sumo-rl is commonly used rather
  than in this codebase.

## Reproducing

```bash
python baseline.py --scenario peak --seed 42 --green 60 --teleport 300  # best static plan
python -m analysis.tripinfo logs/eval_fixedtime_peak_seed42_g60_tripinfo.xml \
    --route-file traffic_peak.rou.xml
python compare.py                        # ranks on delay per completed trip
python analysis/static_timing.py --green 60 --seed 42   # one sweep point
python analysis/actuated.py --min-green 60 --seed 42    # one actuated run
python analysis/headroom.py              # actuated vs best static, paired
python analysis/paired.py                # per-seed pairing of the Stage-1 runs
```

---

## Methodology footnote (must be disclosed for the apples-to-apples claim)

All agents share the same environment, reward function (`diff_waiting − λ·safety`),
state/action space, seeds, and evaluation protocol. Hyperparameters for each
algorithm were selected by Optuna tuning.

**One asymmetry to disclose:** for **off-peak a2c**, the hyperparameter-selection
*objective* was cumulative waiting time (minimize) rather than the shaped reward
used to select dqn/ppo/qrdqn. Reason: at light off-peak demand, throughput is
flat, so the `−λ·safety` term dominates the shaped reward; the reward-optimal
policy is therefore "never switch phase" (best safety, zero throughput) — i.e.
gridlock. Tuning a2c on the shaped reward selected exactly that collapse. Tuning
on waiting time makes gridlock the worst score and rejects it.

This changes only the **HP-selection criterion** for one cell — not the training
reward, environment, or evaluation, all of which remain identical across every
algorithm and scenario.

(Optional future rigor: re-tune dqn/ppo/qrdqn off-peak on the same waiting-time
objective to remove the asymmetry entirely. Not required — those three cells are
already valid.)
