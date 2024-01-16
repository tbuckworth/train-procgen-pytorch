from common.env.custom_viewer_wrapper import DecodedViewerWrapper
from helper import print_values_actions
from inspect_agent import load_policy, latest_model_path, predict
import torch
from common.model import Decoder

def main(args):
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    encoder_path = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/"
    decoder_path = "logs/decode/coinrun/decode/2024-01-16__11-47-52__seed_6033"

    # load encoder
    action_names, done, env, hidden_state, obs, policy = load_policy(render=False, logdir=encoder_path, n_envs=2)
    encoder = policy.embedder
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

    env = DecodedViewerWrapper(env, encoder, decoder, info_key="rgb")
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
