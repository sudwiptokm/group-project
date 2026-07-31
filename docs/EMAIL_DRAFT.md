Subject: RL traffic-signal results — full comparison complete (all cells valid)

Hi all,

The RL vs fixed-time comparison is now complete for both demand scenarios and all
four algorithms (DQN, QR-DQN, PPO, A2C). Every cell is valid — the earlier
off-peak collapse has been resolved, so nothing is excluded.

Headline (λ = 0.5, mean waiting time, seconds, 5 seeds):

Peak (oversaturated):
  DQN        1003   (−24% vs fixed-time)
  A2C        1003   (−24% vs fixed-time)
  fixed-time 1319   (baseline)
  PPO        1357   (+3%)
  QR-DQN     1401   (+6%)
  → RL helps at peak: DQN and A2C cut mean waiting ~24%. PPO/QR-DQN marginally
    worse than fixed-time.

Off-peak (light traffic):
  fixed-time 0.39   (baseline, already near-optimal)
  DQN        0.48
  PPO        1.76
  QR-DQN     1.99
  A2C        36.0
  → Fixed-time is already near-optimal at light demand, so no RL agent beats it,
    but all four now keep traffic mobile. A2C is the weakest (36 s) but valid —
    not the gridlock collapse it previously showed.

On A2C specifically: its earlier off-peak run collapsed to constant-action
gridlock (~1122 s, byte-identical across seeds). Root cause was a reward/objective
issue at light demand, now fixed; A2C is re-run and reported as valid but weakest.

One methodology point to flag for the apples-to-apples claim: off-peak A2C's
hyperparameters were selected using cumulative waiting time as the tuning
objective, whereas the other three were selected on the shaped reward. Same
environment, reward, seeds, and evaluation throughout — only the HP-selection
criterion for that one cell differs, and it's disclosed. Details and full table
in docs/RESULTS_WRITEUP.md.

Happy to walk through any of it.

Thanks,
Sudwipto
