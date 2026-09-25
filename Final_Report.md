# Deep Q-Networks and QR-DQN for Atari Breakout

### AUTHOR: JOHN GAYNES
### [REPOSITORY](https://github.com/neoygaj/AI_project.git)
### [DEMO VIDEO](https://youtube.com/shorts/6aru__G4aGs?feature=share)

# 1. OVERVIEW

This project aimed to explore and test reinforcement learning (RL) approaches for the Atari Breakout game using two deep RL modes:
- DQN (Deep Q-Network)
- QRDQN (Quantile Regression Deep Q-Network)

The goal was to train an agent to learn optimal actions moving the paddle in the game PONG solely based on training experience to achieve as high of a score as possible.

Hyperparameters, such as learning rate, batch size and network size were manipuled to investigate how each one affected stability and performance.

Both of the algorithms were tested in the same environment to offer an unbiased comparison

# 2. APPROACH

### Environment Setup

- Gymnasium ALE (Arcade Learning Environment) was used to emulate Breakout.
- AutoROM was used to instal Atari ROMs.
- AtariPreprocessing wrapper applied Frame-skipping, grayscaling and resized images to 84 x 84
- VecFrameStack stacked 4 consecutive frames to capture motion

### MODEL CHOICES

### DQN

- Classic algorithm that learns a Q-function to estimate the future reward for each action.
- Combines RL with deep neural networks for high dimensional state spaces like images.
- We chose DQN as a baseling because it is well-known, stable and effective for simple Atari games.

### QRDQN

- Extension of DQN that models the distribution of returns, not just the mean.
- More robust to noise and captures uncertainty.
- Chosen because Breakout rewards are sparse and noisy -- QRDQN can stabilize learning.

### HYPERPARAMETERS

| Parameter | Meaning | Tweaks and Findings |
|:---|:---|:---|
| `learning_rate` | How fast the network updates. | Lower rates (1e-4 to 5e-4) were better; too high made training unstable. |
| `gamma` | Discount factor for future rewards. | 0.98–0.99 worked well; higher gamma encourages longer-term planning. |
| `batch_size` | Size of replay batches. | 64 or 128 — larger batch (128) made training smoother but slower. |
| `buffer_size` | Size of experience memory. | 500,000 was better than 100,000 — prevents overfitting to recent episodes. |
| `train_freq` | How often to update (steps). | 4 steps was standard. |
| `target_update_interval` | How often to update target network. | Increasing to 20,000 steps helped stability for QRDQN. |
| `exploration_fraction` | How fast epsilon decays. | Lower exploration decay (0.05) led to more exploitation sooner. |
| `n_quantiles` (QRDQN only) | Number of quantiles to model. | 51 quantiles (default) worked well. |

### EXPERIMENTS AND TROUBLESHOOTING 

### Overfitting:  
    Use of a large replay buffer and lowering learning rate reduced overfitting to short-term rewards

### Network Size:  
    Switching from [256, 256] -> [512, 512, 256] gave better results. However, larger networks such as [1024, 512, 256] resulted in overfitting and more challenging 
    optimization.

### Training Duration:
    - DQN needed at least 1 million steps to show clear reward improvement.
    - QRDQN showed reward gains earlier but plateued without improvement with up to 3 million steps.

# 3. RESULTS AND INTERPRETATION

### DQN RESULTS
    - Training curve: Reward steadily increased after 500,000 step.
    - FInal Performance: Average episode reward around 20 after 1,000,000 steps.
    - Best hyperparams:

        learning_rate = 5e-4
        gamma = 0.98
        batch_size = 64
        network = [256, 256]


### QRDQN RESULTS

    - Training curve: Faster reward growth, more stable.
    - Final Performance: 
        - Reward reached 25 - 30 after 1,000,000 steps with good settings.
        - Longer runs up to 3,000,000 expected to reach 50 - 100 reward.
    - Best hyperparams:

        learning_rate = 1e-4
        gamma = 0.99
        batch_size = 128
        buffer_size = 500_000
        network = [512, 512, 256]
        n_quantiles = 51

    - Observation: QRDQN smoothed out "plateaus" in reward see in standard DQN.

![Best run](good.png)

# 4. CONCLUSIONS AND FUTURE WORK

### Conclusion
    - DQN is reliable and effective for basic training but can plateau early.
    - QRDQN provides better reward distribution learning and robustness.
    - Proper hyperparameter tuning (especially learning rate and batch size) was critical.
    - Longer training runs (> 3,000,000 steps) are needed for mastering Breakout fully.

### Suggestions for Future Improvements
    - Train for 10,000,000 steps to reach expert-level performance.
    - Try Double-DQN or Dueling-DQN architectues.
    - Prioritized Experience Replay for better sampling.
    - Tune n_quantiles(101, 201) for sharper QRDQN performance.
    - Experiment with different games

# 5. REFERENCES

[1.](https://stable-baselines3.readthedocs.io/)
[2.](https://sb3-contrib.readthedocs.io/)
[3.](https://arxiv.org/abs/1207.4708)
[4.](https://github.com/Farama-Foundation/AutoROM)
[5.](https://gymnasium.farama.org/)
