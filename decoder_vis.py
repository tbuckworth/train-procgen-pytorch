import os
import re

import numpy as np
from matplotlib import pyplot as plt

from helper import print_values_actions, add_encoder_to_env, last_folder, save_gif
from inspect_agent import load_policy, latest_model_path, predict
import torch
from common.model import Decoder
from train_decoder import retrieve_coinrun_data, send_reconstruction_update, get_decoder_details


def main(send_reconstructions=False):
    device = torch.device('cpu')
    encoder_path = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/"

    decoder_3, impala_latents_3 = load_decoder_at(device, "block3")
    decoder_2, impala_latents_2 = load_decoder_at(device, "block2")
    decoder_fc, impala_latents_fc = load_decoder_at(device, "fc")

    decoding_info = {"decoder_3": decoder_3,
                     "decoder_2": decoder_2,
                     "decoder_fc": decoder_fc,
                     }
    # load encoder
    action_names, done, env, hidden_state, obs, policy = load_policy(render=True, logdir=encoder_path, n_envs=2,
                                                                     decoding_info=decoding_info, start_level=431,
                                                                     repeat_level=False, num_levels=10)
    encoder = policy.embedder
    add_encoder_to_env(env, encoder)

    # if send_reconstructions:
    #     epoch = int(re.search(r"model_(\d*)\.pth", last_model).group(1))
    #     train_data, valid_data, _ = retrieve_coinrun_data()
    #     send_reconstruction_update(decoder, encoder, epoch, "data/plots", train_data, valid_data, device, impala_latents)

    frames = np.expand_dims(obs[0], 0)
    while True:
        act, log_prob_act, value, next_hidden_state, pi = predict(policy, obs, hidden_state, done)
        # print_values_actions(action_names, pi, value)
        next_obs, rew, done, info = env.step(act)
        frames = np.concatenate((frames[-100:], np.expand_dims(next_obs[0], 0)), 0)
        obs = next_obs
        hidden_state = next_hidden_state
        if done[0]:
            print(f"Level seed: {info[0]['level_seed']}")
    # obs = np.expand_dims(frames[-15],0)
    # np.save(os.path.join(encoder_path, "right_jump_obs.npy"), obs)


def value_saliency_gif():
    device = torch.device('cpu')
    encoder_path = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/"
    action_names, done, env, hidden_state, obs, policy = load_policy(render=True, logdir=encoder_path, n_envs=2,
                                                                     decoding_info={}, start_level=0,
                                                                     repeat_level=False)
    from agents.ppo import PPO as AGENT
    agent = AGENT(env, policy, None, None, device, 0)

    ep_len = 0
    frames = np.expand_dims(obs[0], 0)
    while True:
        obs = np.expand_dims(obs[0], 0)
        act, log_prob_act, value, next_hidden_state, vs_obs = agent.predict_w_value_saliency(obs,
                                                                                             hidden_state,
                                                                                             done)
        sds = (vs_obs - np.mean(vs_obs)) / np.std(vs_obs)
        flt = np.abs(sds) > 1
        flt = np.logical_or.reduce(flt, 1)
        flt = np.tile(flt, (1, 3, 1, 1))
        obs[np.bitwise_not(flt)] -= .2
        frames = np.concatenate((frames[-1000:], np.expand_dims(obs[0], 0)), 0)

        obs, rew, done, info = env.step(np.tile(act, 2))
        ep_len += 1
        hidden_state = next_hidden_state
        if done[0]:
            old_seed = info[0]['level_seed']
            print(f"Level seed: {info[0]['level_seed']}")
            save_gif(frames[-ep_len:], os.path.join(encoder_path, f"saliency_{old_seed}.gif"))
            ep_len = 0


def extreme_action_probs_inspection():
    device = torch.device('cpu')
    encoder_path = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/"
    image_path = os.path.join(encoder_path, "right_jump_obs.npy")

    obs = np.load(image_path, allow_pickle='TRUE')
    action_names, done, env, hidden_state, _, policy = load_policy(render=True, logdir=encoder_path, n_envs=2,
                                                                   decoding_info={}, start_level=0,
                                                                   repeat_level=False)
    from agents.ppo import PPO as AGENT
    agent = AGENT(env, policy, None, None, device, 0)

    plt.imshow(obs[0].transpose(1, 2, 0))
    plt.show()
    act, log_prob_act, value, next_hidden_state, pi = predict(policy, obs, hidden_state, done)
    print_values_actions(action_names, pi, value)

    n_obs = obs.copy()

    background = obs[0, :, 30, 20]

    n_obs[0, :, 0:28, 0:28] = np.tile(background, (28, 28, 1)).transpose(2, 0, 1)
    plt.imshow(n_obs[0].transpose(1, 2, 0))
    plt.show()
    act, log_prob_act, value, next_hidden_state, pi = predict(policy, n_obs, hidden_state, done)
    print_values_actions(action_names, pi, value)

    n_obs[:, :, 0:28, :] = 0  # np.tile(background,(28,64,1)).transpose(2,0,1)
    plt.imshow(n_obs[0].transpose(1, 2, 0))
    plt.show()
    act, log_prob_act, value, next_hidden_state, pi = predict(policy, n_obs, hidden_state, done)
    print_values_actions(action_names, pi, value)

    n_obs[:, :, 38:, :] = 0
    plt.imshow(n_obs[0].transpose(1, 2, 0))
    plt.show()
    act, log_prob_act, value, next_hidden_state, pi = predict(policy, n_obs, hidden_state, done)
    print_values_actions(action_names, pi, value)

    act, log_prob_act, value, next_hidden_state, value_saliency_obs = agent.predict_w_value_saliency(obs, hidden_state,
                                                                                                     done)
    # n_obs[:, :, 38:, :] = 0

    min_v = np.min(value_saliency_obs)
    max_v = np.max(value_saliency_obs)
    vs_obs = (value_saliency_obs - min_v) / (max_v - min_v)

    plt.imshow(vs_obs[0].transpose(1, 2, 0))
    plt.show()
    act, log_prob_act, value, next_hidden_state, pi = predict(policy, vs_obs, hidden_state, done)
    print_values_actions(action_names, pi, value)

    sds = (vs_obs - np.mean(vs_obs)) / np.std(vs_obs)
    flt = np.abs(sds) > 1
    flt = np.logical_or.reduce(flt, 1)
    flt = np.tile(flt, (1, 3, 1, 1))
    vs_obs[flt] = 1
    plt.imshow(vs_obs[0].transpose(1, 2, 0))
    plt.show()

    n_obs = obs.copy()
    n_obs[np.bitwise_not(flt)] -= 0.25
    plt.imshow(n_obs[0].transpose(1, 2, 0))
    plt.savefig(os.path.join(encoder_path, "right_jump_w_saliency.png"))
    act, log_prob_act, value, next_hidden_state, pi = predict(policy, n_obs, hidden_state, done)
    print_values_actions(action_names, pi, value)

    # l = impala_latents_2(policy.embedder, torch.Tensor(obs))
    # obs2 = decoder_2(l).detach().cpu().numpy()
    # plt.imshow(obs2[0].transpose(1, 2, 0))
    # plt.show()
    #
    # l = impala_latents_3(policy.embedder, torch.Tensor(obs))
    # obs3 = decoder_3(l).detach().cpu().numpy()
    # plt.imshow(obs3[0].transpose(1, 2, 0))
    # plt.show()


def load_decoder_at(device, latent_layer):
    decoder_path = last_folder(f"logs/decode/coinrun/decode_{latent_layer}", 1)
    decoder_params, impala_latents = get_decoder_details(latent_layer)
    decoder = Decoder(**decoder_params)
    last_model = latest_model_path(decoder_path)
    print(last_model)
    decoder.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])
    return decoder, impala_latents


if __name__ == "__main__":
    main()

    # #This one shows crazy blue smudge behaviour - the blue smudge seems to correlate with enemies. (just MSE loss)
    # "logs/decode/coinrun/decode/2024-01-17__15-04-40__seed_6033"
    # # level 315 (or 352?) has intense blue code behaviour
    # # level 152 (or 87?) it gets stuck and blue code seems to be the cause.
    # also 49! 366! 64

    # logs/decode/coinrun/decode/2024-01-18__11-44-53__seed_6033
    # 404 - mistook background for enemy - colours indicate enemy?
    # 295 - a lot of colour when landing on first obstacle - indicate enemy/danger?
    # action was DOWN mostly, during colour, but then changed to right as colour went away
    #

    # Tri-decode: 498 - gets stuck and there's loads of intense colour in the image
    # 210: same as 498

    # Tri-decode fully trained:
    # 299 - gets stuck, but not for long
    # 59 - jumps over background image as if enemy
    # 97 - goes back left and then gets stuck
