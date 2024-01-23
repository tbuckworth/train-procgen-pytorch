import re

from helper import print_values_actions, add_encoder_to_env, last_folder
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
                                                                     decoding_info=decoding_info, start_level=0,
                                                                     repeat_level=False)
    encoder = policy.embedder
    add_encoder_to_env(env, encoder)

    # if send_reconstructions:
    #     epoch = int(re.search(r"model_(\d*)\.pth", last_model).group(1))
    #     train_data, valid_data, _ = retrieve_coinrun_data()
    #     send_reconstruction_update(decoder, encoder, epoch, "data/plots", train_data, valid_data, device, impala_latents)

    # env.env.
    # env = DecodedViewerWrapper(env, encoder, decoder, info_key="rgb")
    while True:
        act, log_prob_act, value, next_hidden_state, pi = predict(policy, obs, hidden_state, done)
        print_values_actions(action_names, pi, value)
        next_obs, rew, done, info = env.step(act)
        # rewards = np.append(rewards, rew[done])
        obs = next_obs
        hidden_state = next_hidden_state
        if done[0]:
            print(f"Level seed: {info[0]['level_seed']}")
    # load data
    # process example
    # print example


def load_decoder_at(device, latent_layer):
    decoder_path = last_folder(f"logs/decode/coinrun/decode_{latent_layer}", 1)
    decoder_params, impala_latents = get_decoder_details(latent_layer)
    decoder = Decoder(**decoder_params)
    last_model = latest_model_path(decoder_path)
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
