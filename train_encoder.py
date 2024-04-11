import os

import numpy as np

from helper_local import create_name_from_dict
from common.env.procgen_wrappers import create_env


def randomly_explore_environment(env, frame_file, n_frames, force, checkpoint=50000):
    if os.path.isfile(frame_file) and not force:
        return frame_file
    obs = env.reset()
    frames = obs
    while len(frames) < n_frames:
        act = np.random.randint(0, env.action_space.n, env.num_envs)
        obs, rew, done, info = env.step(act)
        if frames % checkpoint == 0:
            frames = obs
        else:
            frames = np.append(frames, obs, axis=0)

    np.save(frame_file, frames)
    return frame_file


def train_encoder():
    # TODO:
    #  collect random images from environment,
    #  train VQVAE
    env_args = {"num": 2048,
                "env_name": "coinrun",
                "start_level": 0,
                "num_levels": 500,
                "paint_vel_info": True,
                "distribution_mode": "hard"}
    n_frames = 5e6
    frame_file = create_name_from_dict("", "", env_args, ["num"])

    env = create_env(env_args, render=False, normalize_rew=True)
    frame_file = randomly_explore_environment(env, frame_file, n_frames, force=False)

    return


if __name__ == "__main__":
    train_encoder()
