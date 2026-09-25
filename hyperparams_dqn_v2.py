# hyperparams_dqn_v2.py

DQN_HYPERPARAMS = {
    "learning_rate": 1e-4,
    "gamma": 0.98,
    "buffer_size": 100_000,
    "learning_starts": 10_000,
    "batch_size": 64,
    "tau": 1.0,
    "train_freq": 1,  # train after every step
    "target_update_interval": 10_000,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.01,
    "policy_kwargs": dict(
        net_arch=[512, 512],
        normalize_images=False,
    ),
}
