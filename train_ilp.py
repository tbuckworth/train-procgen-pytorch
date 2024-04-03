import os
import time
from collections import deque

import numpy as np
import pandas as pd
import torch
import re

from helper import get_config
from ilp.LogicDistiller import LogicDistiller
from ilp.ilp_helper import append_to_csv_if_exists, create_cmd, run_subprocess
from inspect_agent import load_policy


def time_clingo():
    # cmd = ["wsl", f'echo "{self.clingo_file_contents}" | clingo /dev/stdin']
    filepath = "/mnt/c/Users/titus/PycharmProjects/train-procgen-pytorch/ilp/logic_examples/clingo_learning.lp"
    n = 10
    cmd = create_cmd(["clingo", filepath])
    elapsed = time_cmd(cmd, n)
    print(elapsed)
    for i in range(1, 9):
        cmd = create_cmd(["clingo", "-t", str(i), filepath])
        elapsed = time_cmd(cmd, n)
        print(f"{i}:{elapsed:.4f}")


def time_cmd(cmd, n):
    start = time.time()
    for _ in range(n):
        output = run_subprocess(cmd, "\\n", suppress=True)
    elapsed = time.time() - start
    # print(output)
    return elapsed


def test_ld(logdir="logs/train/coinrun/coinrun/2024-02-20__18-02-16__seed_6033"):
    env, ld, n_envs = load_logic_distiller(logdir, .15)
    print(f"Loaded LogicDistiller with {logdir}")
    df = pd.read_csv("ilp/logic_examples/results.csv")
    df = df[df.logdir == logdir]
    print("Loaded DataFrame from results.csv")
    new_cols = ["balanced_reward", "mean_reward", "pct_random_actions"]
    df[new_cols] = np.nan
    for row in df.index:
        print(f"Testing row {row}:")
        print(f"{df.loc[row].hypothesis}\n")
        df.loc[row, new_cols] = test_agent_in_env(df, env, ld, row)
        append_to_csv_if_exists(df, "ilp/logic_examples/results_with_perf.csv")


def test_agent_in_env(df, env, ld, row):
    ld.action_threshold = df.action_threshold[row]
    ld.atn_threshold = df.atn_threshold[row]
    ld.hypothesis = df.hypothesis[row]
    frame_count = 0
    ep_length = 0
    rand_acts = 0
    all_acts = 0
    observation = env.reset()
    rewards = []
    performance_track = {}
    checkpoints = [i * 40 for i in range(1, int(1e5))]
    i = 0
    while frame_count < int(1e6):  # not done[0]:
        # act0, act_probs, atn, feature_indices, value = ld.forward(observation)
        if i % 50 == 0:
            print(f"Frame count: {i}")
        act = ld.generate_action(observation)

        observation, reward, done, info = env.step(np.array([act, 0]))
        frame_count += 0
        ep_length += 1
        if done[0]:
            r = reward[0][done[0]].item()
            print(f"Random Actions:{ld.number_zero_actions_clingo}/{ep_length}\tReward:{r}")
            rand_acts += ld.number_zero_actions_clingo
            all_acts += ep_length
            rewards.append(r)
            ep_length = 0
            ld.number_zero_actions_clingo = 0
            if 'prev_level_seed' in info[0].keys():
                seed = info[0]["prev_level_seed"]
                rew = info[0]["env_reward"]
                if seed not in performance_track.keys():
                    performance_track[seed] = deque(maxlen=10)
                performance_track[seed].append(rew)
            if len(rewards) > checkpoints[i]:
                i += 1
                all_rewards = list(performance_track.values())
                true_average_reward = np.mean([rew for rew_list in all_rewards for rew in rew_list])
                mean_episode_reward = np.mean(rewards[-40:])
                print(
                    f"Episode:{len(rewards)}\tRewards - balanced:{true_average_reward:.2f}\tmean:{mean_episode_reward:.2f}")
                return true_average_reward, mean_episode_reward, rand_acts / all_acts


def train_logic_program(logdir="logs/train/coinrun/coinrun/2024-03-24__20-31-46__seed_6033"):
    # load model

    # 500: 8x8_8,5,5,5:
    # logdir = "logs/train/coinrun/coinrun/2024-02-12__09-20-18__seed_6033/"
    # logdir = "logs/train/coinrun/coinrun/2024-02-12__09-20-09__seed_6033/"
    # 500: 4x4_10,10:
    # logdir = "logs/train/coinrun/coinrun/2024-02-20__18-02-16__seed_6033"

    # sparsity
    # logdir = "logs/train/coinrun/coinrun/2024-03-24__20-31-46__seed_6033"

    env, ld, n_envs = load_logic_distiller(logdir, atn_threshold=.15)
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

    # action entropy is how unsure is the action
    act_entropy = -np.sum(act_probs * np.log(act_probs), 1)
    all_acts = np.sum(act_probs, 0) / np.sum(act_probs)
    # action kl is how different is the action from average
    act_kl = -np.sum(act_probs * np.log(all_acts), 1)
    # action signal is confident, unusual actions
    action_signal = act_kl - act_entropy

    # # q_diffs no longer used
    # q_diffs = np.diff(value)
    # q_diffs = np.append(q_diffs, 0)

    signal = action_signal

    top_n = new_frames[np.argsort(signal)]
    ld.extract_example(top_n)
    for n in [5, 10, 25, 50, 100]:
        ld.top_n = n
        for i in range(3, 5):
            act_thr = i / 10
            for atn_threshold in [.15, .2, .1]:
                ld.reset_example_strings()
                ld.action_threshold = act_thr
                ld.atn_threshold = atn_threshold
                if not ld.write_examples_to_strings():
                    continue
                ld.write_strings_to_file()
                duration = ld.generate_hypothesis()
                data = {"top_n": [n], "action_threshold": [act_thr], "atn_threshold": [atn_threshold],
                        "calculation_time(s)": [duration],
                        "hypothesis": [ld.hypothesis], "logdir": logdir}
                df = pd.DataFrame(data)
                print(f"top {n}\taction_threshold:{act_thr}\tatn_threshold:{atn_threshold}\n{ld.hypothesis}")
                append_to_csv_if_exists(df, "ilp/logic_examples/results.csv")


def load_logic_distiller(logdir, atn_threshold):
    device = torch.device('cpu')
    cfg = get_config(logdir)
    n_envs = 2
    action_names, done, env, hidden_state, obs, policy, storage = load_policy(False, logdir, n_envs=n_envs,
                                                                              hparams="hard-500-impalafsqmha",
                                                                              start_level=cfg["start_level"],
                                                                              num_levels=cfg["num_levels"])
    # create_logicdistiller
    ld = LogicDistiller(policy, device, probabilistic=False, atn_threshold=atn_threshold)
    return env, ld, n_envs


if __name__ == "__main__":
    # # sparsity
    # logdir = "logs/train/coinrun/coinrun/2024-03-24__20-31-46__seed_6033"

    # very sparse (0.2) 1bn fine-tune:
    logdir = "logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"
    # train_logic_program(logdir)
    test_ld(logdir)
