import numpy as np
import pandas as pd
import torch

from helper import get_config
from ilp.LogicDistiller import LogicDistiller
from ilp.ilp_helper import append_to_csv_if_exists
from inspect_agent import load_policy


def train_logic_program():
    # load model
    device = torch.device('cpu')
    # 500: 8x8_8,5,5,5:
    # logdir = "logs/train/coinrun/coinrun/2024-02-12__09-20-18__seed_6033/"
    # logdir = "logs/train/coinrun/coinrun/2024-02-12__09-20-09__seed_6033/"
    # 500: 4x4_10,10:
    logdir = "logs/train/coinrun/coinrun/2024-02-20__18-02-16__seed_6033"
    cfg = get_config(logdir)
    n_envs = 2
    action_names, done, env, hidden_state, obs, policy, storage = load_policy(False, logdir, n_envs=n_envs,
                                                                              hparams="hard-500-impalafsqmha",
                                                                              start_level=cfg["start_level"],
                                                                              num_levels=cfg["num_levels"])

    # create_logicdistiller
    ld = LogicDistiller(policy, device, probabilistic=False)
    reward = np.array([0. for _ in range(n_envs)])
    # This is so that we learn from an example where the agent succeeds:
    # while reward[0] == 0:
    observation = env.reset()
    frames = np.expand_dims(observation, 0)
    done = np.array([False for _ in range(n_envs)])
    while len(frames) < 100:  # not done[0]:
        act, act_probs, atn, feature_indices, value = ld.forward(observation)
        observation, reward, done, info = env.step(act)
        frames = np.append(frames, np.expand_dims(observation, 0), axis=0)

    (s, b, c, w, h) = frames.shape
    new_frames = frames.reshape((s * b, c, w, h), order='F')
    act, act_probs, atn, feature_indices, value = ld.forward(new_frames)
    q_diffs = np.diff(value)
    q_diffs = np.append(q_diffs, 0)
    top_n = new_frames[np.argsort(q_diffs)]
    ld.extract_example(top_n)
    for n in range(5, 101, 5):
        ld.top_n = n
        for i in range(3, 11):
            act_thr = i / 10
            ld.reset_example_strings()
            ld.action_threshold = act_thr
            if not ld.write_examples_to_strings():
                continue
            ld.write_strings_to_file()
            ld.generate_hypothesis()
            data = {"top_n": [n], "action_threshold": [act_thr], "hypothesis": [ld.hypothesis], "logdir": logdir}
            df = pd.DataFrame(data)
            print(f"top {n}\taction_threshold:{act_thr}\n{ld.hypothesis}")
            append_to_csv_if_exists(df, "ilp/logic_examples/results.csv")


if __name__ == "__main__":
    train_logic_program()
