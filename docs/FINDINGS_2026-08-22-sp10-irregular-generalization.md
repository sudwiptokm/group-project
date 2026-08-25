# Does SP8's idqn-beats-green_wave flip generalize beyond one spacing sample?

Written 2026-08-25. Executes SP10, the second of two open follow-ups
`docs/FINDINGS_2026-08-22-sp8-irregular-spacing.md` left on the table. SP8
built one irregular-spacing net (`corridor_irregular.net.xml`, C1@0/C2@600/
C3@700 nominal -> 578m/78m realised block lengths) and found `idqn` beats
`green_wave` zero-shot there — the first reversal of this project's standing
"`green_wave` wins" result. SP8's own caveat: this could be an artifact of
that one specific asymmetry (skew direction, or how extreme it is) rather
than a property of asymmetric spacing generally. This tests that.

## Method

Two new net variants, same topology/demand/controllers as SP8, only spacing
changed (`analysis/irregular_net_compare2.py`):

- **irregular2** — reverse skew of SP8's: 78m C1-C2 / 578m C2-C3 realised
  (SP8 was 578m/78m). Tests whether the effect depends on *which* block is
  long.
- **irregular3** — moderate asymmetry, same skew direction as SP8 (long
  block first) but 278m/78m realised instead of 578m/78m. Tests whether
  idqn's margin scales with asymmetry severity or is closer to a step
  function.

`green_wave`/`max_pressure` run 5 seeds each (42-46, no checkpoint needed).
`idqn` reuses SP5/SP8's existing `corridor_peak` checkpoints zero-shot, n=3
(42-44) — no additional checkpoints exist without new training, same
constraint SP8's own follow-up hit. All `corridor_peak` demand, `min_green=10`.

## Results

Delay/trip in seconds per completed trip, mean ± sd. "shift" is
variant-minus-regular, paired by seed.

| controller | regular | irregular_sp8 (578/78) | irregular2 (78/578) | irregular3 (278/78) |
|---|---:|---:|---:|---:|
| green_wave | 13.46 ± 0.21s | 19.62 ± 0.31s | 21.48 ± 0.28s | 20.62 ± 0.30s |
| idqn | 16.56 ± 0.36s | 18.48 ± 0.12s | 18.39 ± 0.30s | 17.26 ± 0.03s |
| max_pressure | 25.83 ± 4.24s | 22.17 ± 0.91s | 23.72 ± 2.07s | 21.33 ± 0.88s |

Shift from regular net (green_wave / idqn):

- irregular_sp8: +6.15 ± 0.26s / +1.92 ± 0.39s
- irregular2: +8.02 ± 0.22s / +1.83 ± 0.43s
- irregular3: +7.16 ± 0.27s / +0.70 ± 0.35s

idqn's win margin over green_wave: +1.14s (sp8) / +3.09s (irregular2) /
+3.36s (irregular3).

## Verdict

**The flip generalizes — it is not an artifact of SP8's one asymmetry.**
idqn beats green_wave on all three irregular variants tested: SP8's original
skew, the reverse skew (irregular2), and a more moderate version of SP8's
skew (irregular3). Two findings within that:

- **Not direction-dependent.** irregular2 (long block second) flips the
  same way as SP8's original (long block first) — idqn's margin is if
  anything slightly larger reversed (+3.09s vs +1.14s). The effect is about
  spacing *asymmetry* itself, not about which specific junction carries the
  long block.
- **Not purely severity-scaling in the direction you'd first guess.**
  irregular3's asymmetry is much smaller than SP8's (278m/78m vs 578m/78m)
  yet idqn's win margin (+3.36s) is the *largest* of the three, not the
  smallest — because green_wave degrades nearly as much on the moderate
  asymmetry (+7.16s) as on the extreme one (+8.02s/+6.15s), while idqn's own
  degradation shrinks with milder asymmetry (+0.70s vs +1.83s/+1.92s).
  green_wave's fixed-offset plan appears highly sensitive to *any* deviation
  from uniform spacing, while idqn's zero-shot degradation scales more
  gently with how far spacing departs from what it trained on
  (`corridor_peak`'s uniform 200m/200m). That asymmetric sensitivity, not a
  severity threshold, is why idqn wins across all three variants tested.

This changes the project's standing picture materially: on every
irregular-spacing net tried so far (3/3), `idqn` beats `green_wave`
zero-shot, reversing the "consolidate on green_wave" recommendation
specifically for corridors with non-uniform signal spacing. The
regular-spacing result (`green_wave` wins) and the irregular-spacing result
(`idqn` wins) now both look robust within their own regime, not artifacts of
one net's peculiarities — the open question is no longer "does the flip
generalize" but "which regime does a given real corridor resemble."

The stretch goal in SP10's brief — checking feasibility of a grid-topology
test — was not attempted this session; the three-variant spacing sweep above
took priority and exhausted the session's scope.
