import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parent_log_dir = "logs/dqn_breakout"

# Find the most recent run folder
all_runs = [os.path.join(parent_log_dir, d) for d in os.listdir(parent_log_dir) if os.path.isdir(os.path.join(parent_log_dir, d))]
if not all_runs:
    raise FileNotFoundError(f"No runs found inside {parent_log_dir}")

latest_run = max(all_runs, key=os.path.getmtime)  # newest folder
print(f"📁 Using logs from: {latest_run}")

ea = EventAccumulator(latest_run)
ea.Reload()

tags = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "rollout/exploration_rate",
    "time/fps"
]

os.makedirs("plots", exist_ok=True)

for tag in tags:
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.figure()
        plt.plot(steps, values)
        plt.xlabel("Timesteps")
        plt.ylabel(tag.split("/")[-1].replace("_", " ").capitalize())
        plt.title(tag)
        plt.grid(True)
        filename = os.path.join("plots", tag.replace("/", "_") + ".png")
        plt.savefig(filename)
        print(f"✅ Saved {filename}")
        plt.close()

    except KeyError:
        print(f"❌ Key not found: {tag}")
