Subject: Correction — the peak RL result does not hold (off-peak unaffected)

Hi all,

I need to withdraw the peak headline I sent previously: **DQN and A2C did not cut
mean waiting ~24% vs fixed-time.** I audited that pipeline and found six defects,
five of them independent and any one sufficient to void the number. Corrected,
the result reverses: no learned policy we produced beats a competently timed
static plan, by a factor of 2-3.

The off-peak results are unaffected and stand as reported.

What was wrong, shortest first:

1. The ranking metric was a gridlock clock. system_mean_waiting_time averages
   over vehicles still in the network, so a deadlocked junction accumulates one
   second of "waiting" per second while everything that escapes stops counting.
   Peak locked at t=780 s; the 1319 s figure is the area under that ramp. It also
   inverted the ranking - A2C deadlocked on 5 of 5 seeds and therefore scored
   best.
2. The baseline ran on different traffic than the agents (seed 0 vs seeds 42-46).
   Fixed-time alone spans 242-1319 s across those seeds, and seed 0 was the worst
   draw. Paired per seed, every algorithm flips sign: DQN -23.9% -> +56.9%,
   A2C -> +67.6%, PPO -> +118%, QR-DQN -> +123%.
3. Every evaluated model predates the safety-sampling fix and was never
   retrained.
4. The "+/-std over 5 seeds" was one policy's spread across five demand draws -
   four of the five models per cell were never evaluated.
5. The gridlock came from a library default (sumo-rl ships
   time_to_teleport = -1; SUMO's own default is 300), not from the demand level.
6. The "fixed-time" baseline was a 10 s-green cycler, not a fixed-time plan - and
   10 s is the worst green in the sweep below.

What replaces it. Sweeping the green duration of a static plan at peak shows the
best fixed plan is a long green, and it beats every learned policy we produced.
The cause is structural: with a 3 s amber, a controller switching at 10 s greens
loses 23% of its capacity to clearance time against 4.8% at 60 s, and our agent
decides every 5 s with a 10 s minimum green - exactly where switching is cheap to
attempt and expensive to pay for. One mechanism explains the Stage-1 result, the
pilot, and the 20k-step null together, which is why I'm reporting it as a
property of the problem rather than a training-budget shortfall.

Measurement changes already made: ranking is now delay per completed trip plus
throughput and completion rate (from SUMO tripinfo), the baseline is a swept
static plan rather than a cycler, and baselines run on the agents' own seeds.

One further result, and it is the one I would lead with. "A static plan beats our
agents" is consistent with two different stories - we failed to find an adaptive
policy that exists, or there is none to find at this junction - and no amount of
extra training separates them, because a second null fits both. So I tested it
with a controller that has nothing to learn: a standard queue-actuated
controller, perfect queue information, no reward, no training. Sweeping its
minimum green at peak (seeds 42-46, paired against the 60 s static plan):

  min green 10 s   517.5 s delay,  2925 trips   (+426 vs static, wins 0/5)
  min green 60 s    82.5 s delay,  4156 trips   ( -9.3 vs static, wins 3/5)
  min green 90 s   118.7 s delay,  4038 trips   ( +27 vs static, wins 0/5)

The minimum green was the binding constraint, not the algorithm. At the 10 s
floor we actually trained on, a controller that cannot be accused of
under-training is 5.6x worse than a fixed plan and leaves a quarter of the
traffic unserved. In other words our entire peak training budget was spent in a
region of the action space where no controller can win, which makes that null
over-determined rather than informative about RL.

I want to be equally sceptical about the good row. At a 60 s floor the actuated
controller beats the static plan by 9.3 s, but the paired difference has a
standard deviation of 23.9 s - that is inside the noise, and I am not claiming it
as an improvement in the mean. What is resolvable is consistency: it completes
4142-4177 trips across seeds where the static plan spans 3834-4162. The adaptive
gain is not a lower average, it is the absence of a bad seed.

So the practical recommendation changes from a guess to a measurement: raise the
minimum green to 60 s before retraining anything, and score whatever comes out
against the actuated controller rather than the static plan, since matching a
policy that needs no training would prove nothing. It also sets expectations
honestly - even with the floor corrected the available margin here is around ten
per cent, which is the strongest argument I have for moving to the corridor
setting rather than continuing to optimise this junction.

Off-peak, unchanged:
  fixed-time 0.39   (baseline, already near-optimal)
  DQN        0.48
  PPO        1.76
  QR-DQN     1.99
  A2C        36.0
No RL agent beats the baseline, all four stay mobile, A2C is weakest but valid.
This survives the audit because at 0.39 s there is no headroom - nothing is
stranded for the metric defect to hide.

Two things I want to be explicit about: we no longer claim an algorithm winner
(that only means something once one of them beats a competent static plan), and
the lambda ablation has never been run - only lambda = 0.5 exists, so
"safety-aware" is in the title and not yet in the results. The sweep driver is
written and ready.

Full derivation, measurements and reproduction commands are in
docs/FINDINGS_2026-08-12.md; the rewritten results are in
docs/RESULTS_WRITEUP.md.

Happy to walk through any of it.

Thanks,
Sudwipto
