import os
import gymnasium as gym
import ale_py  # ✅ Registers ALE environments in Shimmy 1.3.0+
from gymnasium.wrappers import AtariPreprocessing
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from datetime import datetime

# Output directory
video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

# Environment factory
def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=False, scale_obs=False)
    env = Monitor(env)
    return env

# Load trained model
model = DQN.load("dqn_breakout_final")

name_prefix = f"dqn-breakout-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Create wrapped environment
record_env = DummyVecEnv([make_env])
record_env = VecFrameStack(record_env, n_stack=4, channels_order="last")
record_env = VecVideoRecorder(
    record_env,
    video_folder=video_dir,
    record_video_trigger=lambda step: step == 0,
    video_length=1000,
    name_prefix="dqn-breakout"
)

# Run a rollout and record
obs = record_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = record_env.step(action)

record_env.close()
print(f"🎥 Video saved to: {video_dir}")
