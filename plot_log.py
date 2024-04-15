import os

import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    logdir = "logs/train/cartpole/cartpole/2024-04-15__15-45-47__seed_6033"
    df = pd.read_csv(os.path.join(logdir, "log-append.csv"))
    # "logs/train/cartpole/cartpole/2024-04-15__15-45-47__seed_6033/log-append.csv"
    df.plot(x="timesteps", y=["mean_episode_rewards", "val_mean_episode_rewards"], kind="line")

    plt.show()
