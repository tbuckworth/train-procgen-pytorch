import copy
import os
import re

import numpy as np
import torch

from common.storage import Storage
from helper import get_hyperparams, initialize_model, print_values_actions, get_action_names, save_gif
from common.env.procgen_wrappers import create_env


def predict(policy, obs, hidden_state, done):
    with torch.no_grad():
        obs = torch.FloatTensor(obs).to(device=policy.device)
        hidden_state = torch.FloatTensor(hidden_state).to(device=policy.device)
        mask = torch.FloatTensor(1 - done).to(device=policy.device)
        dist, value, hidden_state = policy(obs, hidden_state, mask)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

    pi = torch.nn.functional.softmax(dist.logits, dim=1)

    return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy(), pi.cpu().numpy()


def main(logdir, render=True):
    action_names, done, env, hidden_state, obs, policy = load_policy(render, logdir, n_envs=2)
    rewards = np.array([])
    while True:
        act, log_prob_act, value, next_hidden_state, pi = predict(policy, obs, hidden_state, done)
        print_values_actions(action_names, pi, value, rewards=rewards)
        next_obs, rew, done, info = env.step(act)
        rewards = np.append(rewards, rew[done])
        obs = next_obs
        # frames = np.append(frames, obs, axis=0)
        # np.save("logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/go_left.npy", frames)
        hidden_state = next_hidden_state
        if done[0]:
            print(f"Level seed: {info[0]['level_seed']}")


def load_policy(render, logdir, n_envs=None, decoding_info={}, start_level=0, repeat_level=False):
    # logdir = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033"
    # df = pd.read_csv(os.path.join(logdir, "log-append.csv"))
    last_model = latest_model_path(logdir)
    device = torch.device('cpu')
    hyperparameters = get_hyperparams("hard-500-impala")
    hp_file = os.path.join(logdir, "hyperparameters.npy")
    if os.path.exists(hp_file):
        hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    if n_envs is not None:
        hyperparameters["n_envs"] = n_envs
    env_args = {"num": hyperparameters["n_envs"],
                "env_name": "coinrun",
                "start_level": start_level,#325
                "num_levels": 1 if repeat_level else 500,
                "paint_vel_info": True,
                "distribution_mode": "hard"}
    normalize_rew = hyperparameters.get('normalize_rew', True)
    env = create_env(env_args, render, normalize_rew, mirror_some=True, decoding_info=decoding_info)
    model, observation_shape, policy = initialize_model(device, env, hyperparameters)
    policy.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])
    # Test if necessary:
    policy.device = device
    storage = Storage(observation_shape, model.output_dim, 256, n_envs, device)
    action_names = get_action_names(env)
    obs = env.reset()
    hidden_state = np.zeros((n_envs, storage.hidden_state_size))
    done = np.zeros(n_envs)
    # frames = obs
    policy.eval()
    return action_names, done, env, hidden_state, obs, policy


def latest_model_path(logdir):
    files = os.listdir(logdir)
    pattern = r"model_(\d*)\.pth"
    checkpoints = [int(re.search(pattern, x).group(1)) for x in files if re.search(pattern, x)]
    last_model = os.path.join(logdir, f"model_{max(checkpoints)}.pth")
    return last_model


def inspect_frames():
    folder = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/"
    import matplotlib.pyplot as plt
    frames = np.load(f"{folder}go_left.npy", allow_pickle='TRUE')
    action_names, done, env, hidden_state, obs, policy = load_policy(render=False)

    save_gif(frames[:234, ], f"{folder}normal_play.gif")
    save_gif(frames[234:, ], f"{folder}broken_play.gif")

    # STARTS AT -95 = 234

    # i = 263 #LEFT
    # i = 258 #RIGHT
    for i in [258, 263]:
        obs = frames[i:(i + 1), ]
        plt.imshow(frames[i].transpose(1, 2, 0))
        plt.savefig(f"{folder}frame_{i}.png")
        act, log_prob_act, value, next_hidden_state, pi = predict(policy, obs, hidden_state, done)
        print_values_actions(action_names, pi, value)

    right_frame = frames[258:259]
    left_frame = frames[263:264]

    # # Testing interpolation of images - wasn't helpful
    # int_frame = (right_frame + left_frame)/2.0
    # plt.imshow(int_frame[0].transpose(1, 2, 0))
    # plt.show()
    # act, log_prob_act, value, next_hidden_state, pi = predict(policy, int_frame, hidden_state, done)
    # print_values_actions(action_names, pi, value)

    # Just swapping the first velocity information portion of the screen - made no difference
    np_slice = np.index_exp[0, :, 0:13, 0:13]

    swap_indexed_values_and_print(action_names, done, hidden_state, left_frame, [np_slice], policy, right_frame, plt)

    # Here I cycle through segments of the images, swapping them out and monitoring the agent action probabilities
    n = 4
    m = n * n
    k = 64 // n
    slices = []
    for i in range(m):
        row = (i % n)
        col = (i // n)
        np_slice = np.index_exp[0, :, (row * k):((row + 1) * k), (col * k):((col + 1) * k)]
        slices.append(np_slice)
        n_frame = swap_indexed_values_and_print(action_names, done, hidden_state, left_frame, slices, policy,
                                                right_frame, plt, i)
        if i == 9:
            still_right = n_frame
        # I found that the phase shift in action probabilities occurs on the 10th swapping
        if i == 10:
            now_left = n_frame

    plt.imshow(still_right[0].transpose(1, 2, 0))
    plt.savefig(f"{folder}still_right_frame.png")
    plt.imshow(now_left[0].transpose(1, 2, 0))
    plt.savefig(f"{folder}now_left_frame.png")

    # Looking at the embeddings, they are somewhat different
    xr = policy.embedder(torch.Tensor(still_right))
    x = policy.embedder(torch.Tensor(still_right))
    xl = policy.embedder(torch.Tensor(now_left))

    _, ind = torch.where(xr != xl)

    # Here I cycle through each embedding and see if swapping it out significantly shifts the logits
    for i in ind:
        x[0, i] = xl[0, i]
        policy.fc_policy(xr)
        policy.fc_policy(x)


def swap_indexed_values_and_print(action_names, done, hidden_state, left_frame, slices, policy, right_frame, plt, i=""):
    n_frame = copy.deepcopy(right_frame)
    for np_slice in slices:
        n_frame[np_slice] = left_frame[np_slice]
    plt.imshow(n_frame[0].transpose(1, 2, 0))
    plt.show()
    act, log_prob_act, value, next_hidden_state, pi = predict(policy, n_frame, hidden_state, done)
    print_values_actions(action_names, pi, value, i)
    return n_frame


if __name__ == "__main__":
    # IMPALAFSQMHA:
    # main(logdir="logs/train/coinrun/coinrun/2023-12-08__17-11-08__seed_6033")


    # main(logdir="logs/train/coinrun/coinrun/2023-11-23__10-31-05__seed_6033/")

    #impalavqmha
    # main(logdir="logs/train/coinrun/coinrun/2023-11-28__11-37-25__seed_6033/")

    # #impala:
    # main(logdir="logs/train/coinrun/coinrun/2023-11-28__10-59-15__seed_6033/")
    # Strong Impala:
    main(logdir="logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/")

    # #impalavqmha - Using low x_entropy coefficient (with old entropy metric)
    # main(logdir="logs/train/coinrun/coinrun/2023-11-30__13-33-16__seed_6033")

    # #impalavqmha - No x-entropy - mirror env. only
    # main(logdir="logs/train/coinrun/coinrun/2023-11-30__17-47-52__seed_6033")


    # #mut_info alpha=1/3 : (ignores inputs)
    # main(logdir="logs/train/coinrun/coinrun/2023-12-01__10-47-13__seed_6033")

    #mut_info alpha=2/3
    # main(logdir="logs/train/coinrun/coinrun/2023-12-01__10-47-20__seed_6033")
