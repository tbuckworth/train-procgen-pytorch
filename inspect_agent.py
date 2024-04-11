import copy
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from helper_local import print_values_actions, save_gif, last_folder, print_action_entropy, coords_to_image, get_config, balanced_reward, load_storage_and_policy, \
    load_hparams_for_model
from common.env.procgen_wrappers import create_env


def inspect_latents(policy, obs, device):
    obs = torch.FloatTensor(obs).to(device)
    l = policy.embedder.encoder(obs)
    x, indices = policy.embedder.flatten_and_append_coor(l, True)
    ind, count = indices.unique(return_counts=True)
    print(torch.concatenate((ind.unsqueeze(0), count.unsqueeze(0)), dim=0).cpu().numpy())
    #
    #
    # ind, count = np.unique(indices.cpu().numpy(), return_counts=True)
    # np.concatenate((ind.unsqueeze(),count.unsqueeze()),axis=1)
    # dist, value = policy.hidden_to_output(x)
    #
    # act = dist.sample().cpu().numpy()
    # act_probs = dist.probs.detach().cpu().numpy()
    # value = value.detach().cpu().numpy()
    # feature_indices = feature_indices.detach().cpu().numpy()
    # atn = atn.detach().cpu().numpy()
    # return act, act_probs, atn, feature_indices, value


def predict(policy, obs, hidden_state, done, return_dist=False):
    with torch.no_grad():
        obs = torch.FloatTensor(obs).to(device=policy.device)
        hidden_state = torch.FloatTensor(hidden_state).to(device=policy.device)
        mask = torch.FloatTensor(1 - done).to(device=policy.device)
        dist, value, hidden_state = policy(obs, hidden_state, mask)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

    pi = torch.nn.functional.softmax(dist.logits, dim=1)

    if return_dist:
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy(), pi.cpu().numpy(), dist
    return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy(), pi.cpu().numpy()


def plot_atn_arrows(policy, observation, logdir):
    obs = torch.FloatTensor(observation).to(policy.device)
    x, atn_list, feature_indices, _ = policy.embedder.forward_with_attn_indices(obs)
    atn = atn_list[-1]
    # high_atn = atn[0] > 0.4
    found = False
    for i in range(6):
        atn_threshold = (9 - i) / 10
        high_atn = atn[0] > atn_threshold
        if high_atn.sum() > 0:
            found = True
            break
    if not found:
        return
    arrows = high_atn.argwhere().detach().numpy()
    plt.imshow(observation.transpose((0, 2, 3, 1))[0])
    cmap = plt.cm.jet
    cNorm = colors.Normalize(vmin=0, vmax=atn[0].shape[0])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    for arr, weight in zip(arrows, atn[0][high_atn].detach().numpy()):
        colour = scalarMap.to_rgba(arr[0])
        r, c = coords_to_image(arr[1], atn.shape[-1], observation.shape[-1])
        rd, cd = coords_to_image(arr[2], atn.shape[-1], observation.shape[-1])
        plt.arrow(c, r, cd - c, rd - r, width=0.05, head_width=1.5, alpha=weight, color=colour, label=f"r{arr[0]}")
    plt.show()
    print("yes")
    if False:
        plt.savefig(os.path.join(logdir, "lots of attention to top right at beginning.png"))


def main(logdir, render=True, print_entropy=False, draw_atn_arrows=False):
    print(logdir)
    cfg = get_config(logdir)
    action_names, done, env, hidden_state, obs, policy, storage = load_policy(render, logdir, n_envs=2, start_level=0,
                                                                              num_levels=500)
    rewards = np.array([])
    performance_track = {}
    while True:
        act, log_prob_act, value, next_hidden_state, pi = predict(policy, obs, hidden_state, done)
        if print_entropy:
            print_action_entropy(action_names, pi)
            inspect_latents(policy, obs, policy.device)
        else:
            print_values_actions(action_names, pi, value, rewards=rewards)
        next_obs, rew, done, info = env.step(act)
        obs_tensor = torch.FloatTensor(next_obs).to(policy.device)
        x, atn_list, feature_indices, _ = policy.embedder.forward_with_attn_indices(obs_tensor)
        atn = atn_list[-1]
        if draw_atn_arrows:
            if (atn[0] > 0.2).any():
                plot_atn_arrows(policy, next_obs, logdir)
        storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
        rewards = np.append(rewards, rew[done])
        obs = next_obs
        # frames = np.append(frames, obs, axis=0)
        # np.save("logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/go_left.npy", frames)
        hidden_state = next_hidden_state
        if done[0]:
            print(f"Level seed: {info[0]['level_seed']}")
        true_average_reward = balanced_reward(done, info, performance_track)
        print(true_average_reward)


def load_policy(render, logdir, n_envs=None, decoding_info={}, start_level=0, repeat_level=False, num_levels=500,
                hparams="hard-500-impala"):
    # logdir = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033"
    # df = pd.read_csv(os.path.join(logdir, "log-append.csv"))
    device = torch.device('cpu')
    hyperparameters, last_model = load_hparams_for_model(hparams, logdir, n_envs)
    env_args = {"num": hyperparameters["n_envs"],
                "env_name": "coinrun",
                "start_level": start_level,  # 325
                "num_levels": 1 if repeat_level else num_levels,
                "paint_vel_info": True,
                "distribution_mode": "hard"}
    normalize_rew = hyperparameters.get('normalize_rew', True)
    try:
        cfg = get_config(logdir)
        mirror_some = cfg["mirror_env"]
        reduce_duplicate_actions = cfg["reduce_duplicate_actions"]
    except Exception:
        mirror_some = True
        reduce_duplicate_actions = False
    env = create_env(env_args, render, normalize_rew, mirror_some, decoding_info=decoding_info,
                     reduce_duplicate_actions=reduce_duplicate_actions)
    action_names, done, hidden_state, obs, policy, storage = load_storage_and_policy(device, env, hyperparameters,
                                                                                     last_model, logdir, n_envs)
    return action_names, done, env, hidden_state, obs, policy, storage


def inspect_frames():
    folder = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/"
    import matplotlib.pyplot as plt
    frames = np.load(f"{folder}go_left.npy", allow_pickle='TRUE')
    action_names, done, env, hidden_state, obs, policy, storage = load_policy(render=False)

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
    # 338 is a cool, complex level

    # TODO: fix with config.npy
    main(last_folder("logs/train/coinrun/coinrun", 1),
         render=True,
         print_entropy=False,
         draw_atn_arrows=False)

    # IMPALAFSQMHA:
    # main(logdir="logs/train/coinrun/coinrun/2023-12-08__17-11-08__seed_6033")

    # main(logdir="logs/train/coinrun/coinrun/2023-11-23__10-31-05__seed_6033/")

    # impalavqmha
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

    # mut_info alpha=2/3
    # main(logdir="logs/train/coinrun/coinrun/2023-12-01__10-47-20__seed_6033")
