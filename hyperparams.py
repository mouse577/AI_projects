# hyperparams.py

from stable_baselines3.common.utils import get_linear_fn

DQN_HYPERPARAMS = {
    "learning_rate": get_linear_fn(1e-4, 2e-5, 1.0),
    "gamma": 0.99,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 16,
    "tau": 1.0,
    "train_freq": 4,
    "target_update_interval": 5_000,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.01,
    "policy_kwargs": dict(
        net_arch=[1024, 1024],
        normalize_images=True,
    ),
}
