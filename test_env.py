import gymnasium as gym
import ale_py  # ✅ this now works after downgrade

env = gym.make("ALE/Breakout-v5", render_mode="human")
obs, _ = env.reset(seed=42)

for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()

env.close()






