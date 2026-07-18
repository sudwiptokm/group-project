"""One-off calibration probe: measure mean|efficiency| vs mean safety at lam=0
on the peak scenario (seed 0), suggest SAFETY_SCALE = mean_safety / mean|eff|.

Writes result to calibration_result.txt. Throwaway helper (not part of the run).
"""
import numpy as np
import env_common as ec
from env_common import make_env

env = make_env(seed=0, scenario="peak", lam=0.0)
env.reset()
ts = env.unwrapped.traffic_signals[list(env.unwrapped.traffic_signals)[0]]
effs, safs = [], []
done = False
a, n = 0, env.action_space.n
while not done:
    # at lam=0 the step reward IS the efficiency (diff-waiting-time). Do NOT
    # re-call _efficiency (it mutates last_measure; a second call reads ~0).
    _, reward, term, trunc, _ = env.step(a)
    a = (a + 1) % n
    effs.append(abs(reward))
    safs.append(ec._safety_penalty(ts))  # stateless recompute; safe to call here
    done = term or trunc
env.close()

me, ms = float(np.mean(effs)), float(np.mean(safs))
scale = ms / me if me else 0.0
with open("calibration_result.txt", "w") as fh:
    fh.write(f"mean_abs_eff={me:.4f}\nmean_safety={ms:.4f}\nsuggested_SAFETY_SCALE={scale:.4f}\n")
print(f"mean|eff|={me:.4f} mean_safety={ms:.4f} suggested SAFETY_SCALE={scale:.4f}")
