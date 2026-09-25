# hyperparams_qrdqn.py

QRDQN_HYPERPARAMS = {
    "learning_rate": 1e-4,        
    "gamma": 0.99,                
    "buffer_size": 500_000,
    "learning_starts": 10_000,
    "batch_size": 128,             
    "tau": 1.0,
    "train_freq": 4,
    "target_update_interval": 20_000,  
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.05,

    # "learning_rate": 3e-4,        
    # "gamma": 0.995,                
    # "buffer_size": 750_000,
    # "learning_starts": 20_000,
    # "batch_size": 128,             
    # "tau": 1.0,
    # "train_freq": 4,
    # "target_update_interval": 15_000,  
    # "exploration_fraction": 0.05,
    # "exploration_final_eps": 0.005,
                
    "policy_kwargs": dict(
        net_arch=[512, 512, 256],       
        normalize_images=False,         
        n_quantiles=51,   
    ),
}
