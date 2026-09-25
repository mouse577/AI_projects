#  first run pip install -r requirements.txt
#  may need to pip install sb3-contrib[extra] separately

#  run with python train_dqn.py --algo dqn --config v1
#  or python train_dqn.py --algo qrdqn --config v1

#  when training completes, run python plot_rewards_2.py
#  then run python training_summary.py


import os
import gymnasium as gym
import numpy as np
import ale_py  # 🔑 registers ALE environments
import argparse
from gymnasium.spaces import Box
from gymnasium.wrappers import AtariPreprocessing
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from sb3_contrib import QRDQN
from hyperparams import DQN_HYPERPARAMS
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy

# 🛠 Parse command line
parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["dqn", "qrdqn"], required=True, help="Choose the algorithm: dqn or qrdqn")
parser.add_argument("--config", choices=["v1", "v2"], default="v1", help="Choose hyperparameter config: v1 or v2")
args = parser.parse_args()

# 🛠 Load hyperparams based on algorithm and config
if args.algo == "dqn":
    if args.config == "v1":
        from hyperparams import DQN_HYPERPARAMS as HYPERPARAMS
    elif args.config == "v2":
        from hyperparams_dqn_v2 import DQN_HYPERPARAMS as HYPERPARAMS
    ModelClass = DQN

elif args.algo == "qrdqn":
    if args.config == "v1":
        from hyperparams_qrdqn import QRDQN_HYPERPARAMS as HYPERPARAMS
    else:
        raise ValueError("QRDQN only supports config v1 for now")
    
    ModelClass = QRDQN


# Output directories
run_name = f"{args.algo}_{args.config}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
log_dir = f"logs/dqn_breakout/{run_name}"
checkpoint_dir = f"checkpoints/{run_name}"
video_dir = "videos"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Environment factory
def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)  # ✅ Correct: directly passed
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=False, scale_obs=False, screen_size=84, terminal_on_life_loss=False, noop_max=30)
    env = Monitor(env)
    return env


# Vectorized environment with frame stacking
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4, channels_order="last")
print("🚨 Obs shape:", env.observation_space)


# Save checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=checkpoint_dir,
    name_prefix="dqn_breakout"
)

# DQN model
model = ModelClass(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    tensorboard_log=log_dir,
    **HYPERPARAMS  # <--- 🔥 inject all hyperparams automatically
)
new_logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# Train for 200k steps
total_timesteps = 1_000_000
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
model.save(f"{args.algo}_breakout_final_{run_name}")  # ✅ Avoid overwriting

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"✅ Evaluation: mean_reward={mean_reward:.2f}, std={std_reward:.2f}")

# 🎥 Record a demo video
name_prefix = f"{args.algo}-breakout-{run_name}"
record_env = DummyVecEnv([make_env])
record_env = VecFrameStack(record_env, n_stack=4)
record_env = VecVideoRecorder(
    record_env,
    video_folder=video_dir,
    record_video_trigger=lambda step: step == 0,
    video_length=1000,
    name_prefix=name_prefix
)

obs = record_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = record_env.step(action)

record_env.close()
print(f"🎥 Video saved to: {video_dir}")
