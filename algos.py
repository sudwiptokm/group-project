"""
Algorithm registry for the comparison ladder.

Action space here is DISCRETE (pick the next green phase), so every algorithm
must support discrete actions:

    DQN     value-based, off-policy          (baseline)
    QR-DQN  distributional DQN, off-policy    (sb3-contrib)
    PPO     on-policy policy-gradient
    A2C     on-policy actor-critic

NOTE: SAC is intentionally NOT here. SAC is a continuous-action algorithm; it
cannot be applied to this discrete phase-selection problem without a different
(continuous) action parameterisation, so it is out of scope for this comparison.

Each entry provides:
    cls            SB3 / sb3-contrib algorithm class
    defaults()     RL-Zoo-style default hyperparameters (the "no-tuning" run)
    sample(trial)  Optuna search space -> hyperparameter dict (for tune.py)

`defaults()` and `sample()` both return plain kwargs passed straight to `cls(...)`.
`policy` is always "MlpPolicy" (Box observation, discrete action).
"""

from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import QRDQN

POLICY = "MlpPolicy"

# net-arch presets referenced by both defaults and the Optuna search
_NET_ARCHS = {
    "small": [64, 64],
    "medium": [256, 256],
    "large": [400, 300],
}


# ----------------------------------------------------------------------------
# Off-policy (DQN, QR-DQN) share a hyperparameter shape
# ----------------------------------------------------------------------------
def _off_policy_defaults() -> dict:
    return dict(
        policy=POLICY,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=_NET_ARCHS["small"]),
    )


def _off_policy_sample(trial) -> dict:
    net = trial.suggest_categorical("net_arch", list(_NET_ARCHS))
    return dict(
        policy=POLICY,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        buffer_size=trial.suggest_categorical("buffer_size", [20_000, 50_000, 100_000]),
        learning_starts=trial.suggest_categorical("learning_starts", [500, 1_000, 5_000]),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        gamma=trial.suggest_categorical("gamma", [0.95, 0.99, 0.995]),
        train_freq=trial.suggest_categorical("train_freq", [1, 4, 8]),
        target_update_interval=trial.suggest_categorical("target_update_interval", [500, 1_000, 5_000]),
        exploration_fraction=trial.suggest_float("exploration_fraction", 0.05, 0.4),
        exploration_final_eps=trial.suggest_float("exploration_final_eps", 0.01, 0.1),
        policy_kwargs=dict(net_arch=_NET_ARCHS[net]),
    )


# ----------------------------------------------------------------------------
# On-policy (PPO, A2C)
# ----------------------------------------------------------------------------
def _ppo_defaults() -> dict:
    return dict(
        policy=POLICY,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=_NET_ARCHS["small"]),
    )


def _ppo_sample(trial) -> dict:
    net = trial.suggest_categorical("net_arch", list(_NET_ARCHS))
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])
    # keep batch_size a divisor of n_steps so PPO doesn't drop a partial batch
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    if n_steps % batch_size != 0:
        batch_size = n_steps
    return dict(
        policy=POLICY,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=trial.suggest_categorical("n_epochs", [5, 10, 20]),
        gamma=trial.suggest_categorical("gamma", [0.95, 0.99, 0.995]),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 1.0),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        ent_coef=trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True),
        policy_kwargs=dict(net_arch=_NET_ARCHS[net]),
    )


def _a2c_defaults() -> dict:
    return dict(
        policy=POLICY,
        learning_rate=7e-4,
        n_steps=8,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=_NET_ARCHS["small"]),
    )


def _a2c_sample(trial) -> dict:
    net = trial.suggest_categorical("net_arch", list(_NET_ARCHS))
    return dict(
        policy=POLICY,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        n_steps=trial.suggest_categorical("n_steps", [5, 8, 16, 32]),
        gamma=trial.suggest_categorical("gamma", [0.95, 0.99, 0.995]),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 1.0),
        ent_coef=trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True),
        vf_coef=trial.suggest_float("vf_coef", 0.25, 0.75),
        policy_kwargs=dict(net_arch=_NET_ARCHS[net]),
    )


# ----------------------------------------------------------------------------
ALGOS = {
    "dqn":   dict(cls=DQN,   defaults=_off_policy_defaults, sample=_off_policy_sample),
    "qrdqn": dict(cls=QRDQN, defaults=_off_policy_defaults, sample=_off_policy_sample),
    "ppo":   dict(cls=PPO,   defaults=_ppo_defaults,        sample=_ppo_sample),
    "a2c":   dict(cls=A2C,   defaults=_a2c_defaults,        sample=_a2c_sample),
}


def build(algo: str, env, params: dict, seed: int, tb_log: str = None):
    """Instantiate `algo` on `env` with `params` (kwargs)."""
    cls = ALGOS[algo]["cls"]
    return cls(env=env, seed=seed, verbose=0, tensorboard_log=tb_log, **params)
