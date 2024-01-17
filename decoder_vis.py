import re

from helper import print_values_actions, add_encoder_to_env
from inspect_agent import load_policy, latest_model_path, predict
import torch
from common.model import Decoder
from train_decoder import retrieve_coinrun_data, send_reconstruction_update


def main(send_reconstructions=False):

    device = torch.device('cpu')
    encoder_path = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/"
    decoder_path = "logs/decode/coinrun/decode/2024-01-17__09-16-24__seed_6033"

    # load decoder
    decoder = Decoder(
        embedding_dim=32,
        num_hiddens=64,
        num_upsampling_layers=3,
        num_residual_layers=2,
        num_residual_hiddens=32,
    )
    last_model = latest_model_path(decoder_path)
    decoder.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])

    decoding_info = {"decoder": decoder}
    # load encoder
    action_names, done, env, hidden_state, obs, policy = load_policy(render=True, logdir=encoder_path, n_envs=2, decoding_info=decoding_info)
    encoder = policy.embedder
    add_encoder_to_env(env, encoder)

    if send_reconstructions:
        epoch = int(re.search(r"model_(\d*)\.pth", last_model).group(1))
        train_data, valid_data, _ = retrieve_coinrun_data()
        send_reconstruction_update(decoder, encoder, epoch, "data/plots", train_data, valid_data, device)

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



if __name__ == "__main__":
    main()
