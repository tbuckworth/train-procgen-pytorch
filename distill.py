import argparse
import csv
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch

from common.env.procgen_wrappers import create_env
from common.storage import Storage
from helper import initialize_model, get_hyperparams
from contextlib import nullcontext

try:
    import wandb
except ImportError:
    pass


def predict(policy, obs, hidden_state, done, with_grad=False):
    with torch.no_grad() if policy.training or not with_grad else nullcontext():
        obs = torch.Tensor(obs).to(device=policy.device)
        # hidden_state = torch.FloatTensor(hidden_state).to(device=policy.device)
        # mask = torch.FloatTensor(1 - done).to(device=policy.device)
        if policy.has_vq:
            dist, value, hidden_state, commit_loss = policy(obs, hidden_state, hidden_state)
        else:
            dist, value, hidden_state = policy(obs, hidden_state, hidden_state)
        act = dist.sample()
    return dist.logits, act.cpu().numpy()


def load_policy(render, logdir, device, args):
    files = os.listdir(logdir)
    pattern = r"model_(\d*)\.pth"
    checkpoints = [int(re.search(pattern, x).group(1)) for x in files if re.search(pattern, x)]
    last_model = os.path.join(logdir, f"model_{max(checkpoints)}.pth")
    # device = torch.device('cpu')
    hyperparameters = get_hyperparams("hard-500-impala")
    hp_file = os.path.join(logdir, "hyperparameters.npy")
    if os.path.exists(hp_file):
        hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    n_envs = args.n_envs
    env_args = {"num": n_envs,
                "env_name": "coinrun",
                "start_level": args.start_level,
                "num_levels": args.num_levels,
                "paint_vel_info": True,
                "distribution_mode": "hard"}
    normalize_rew = hyperparameters.get('normalize_rew', True)
    env = create_env(env_args, render, normalize_rew, mirror_some=False)
    model, observation_shape, policy = initialize_model(device, env, hyperparameters)
    policy.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])
    # Test if necessary:
    policy.device = device
    storage = Storage(observation_shape, model.output_dim, 256, n_envs, device)
    # action_names = get_action_names(env)
    obs = env.reset()
    hidden_state = np.zeros((n_envs, storage.hidden_state_size))
    done = np.zeros(n_envs)
    # frames = obs
    policy.eval()
    return done, env, hidden_state, obs, policy


def collect_obs(new_policy, obs, hidden_state, done, env, n_states, return_logits=False):
    observations = obs
    rewards = np.array([])
    dones = done
    if return_logits:
        logits_store, _ = predict(new_policy, obs, hidden_state, done)
    while len(observations) <= n_states:
        # Predict:
        logits, act = predict(new_policy, obs, hidden_state, done)
        # Store:
        observations = np.append(observations, obs, axis=0)
        if return_logits:
            logits_store = torch.cat((logits_store, logits), dim=0)
        # Act:
        next_obs, rew, done, info = env.step(act)
        rewards = np.append(rewards, rew[done])
        dones = np.append(dones, done, axis=0)
        obs = next_obs
    if return_logits:
        return observations[1:], rewards, dones[1:], logits_store[1:]
    return observations[1:], rewards, dones[1:]


def collect_validation_data(policy, device, args, hyperparameters, hidden_state, done, n_states):
    env_args = {"num": args.n_envs,
                "env_name": "coinrun",
                "start_level": args.start_level + args.num_levels + 1,
                "num_levels": 0,
                "paint_vel_info": True,
                "distribution_mode": "hard"}
    normalize_rew = hyperparameters.get('normalize_rew', True)
    valid_env = create_env(env_args, False, normalize_rew, mirror_some=False)

    first_obs = valid_env.reset()
    obs, rew, done, logits = collect_obs(policy, first_obs, hidden_state, done, valid_env, n_states, return_logits=True)
    return torch.Tensor(obs).to(device), logits


def validate(new_policy, valid_X, valid_Y_gold, criterion, hidden_state):
    if new_policy.has_vq:
        dist_batch, value_batch, _, commit_loss = new_policy(valid_X, hidden_state, hidden_state)
    else:
        dist_batch, value_batch, _ = new_policy(valid_X, hidden_state, hidden_state)
    Y_pred = dist_batch.logits
    loss = criterion(Y_pred.squeeze(), valid_Y_gold.squeeze())
    return loss.item() * len(valid_Y_gold)


def predict_in_batches(policy, obs, hidden_state, done, batch_size=32*8):
    N_batchs = int(len(obs) / batch_size)
    for i in range(N_batchs):
        obs_batch = obs[i * batch_size: (i + 1) * batch_size]
        Y_gold, _ = predict(policy, obs_batch, hidden_state, done)
        if i == 0:
            total = Y_gold
        else:
            total = torch.cat(total, Y_gold, dim=0)
    return total


def distill(args, logdir_trained):
    logdir = os.path.join('logs', 'distill', "coinrun", "distill")
    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{args.seed}'
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    hyperparameters = get_hyperparams('hard-500-impalavqmha')
    batch_size = args.batch_size
    explore_size = args.explore_size
    save_every = args.n_explore // args.num_checkpoints
    checkpoint_cnt = 0

    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    # Load Trained Model
    done, env, hidden_state, first_obs, policy = load_policy(render=False,
                                                             logdir=logdir_trained,
                                                             args=args,
                                                             device=device)
    # Load Blank Model
    model, observation_shape, new_policy = initialize_model(device, env, hyperparameters)
    new_policy.device = device
    log = pd.DataFrame(columns=["Exploration", "Epoch", "Loss", "Valid_Loss", "Mean_Reward"])

    valid_X, valid_Y_gold = collect_validation_data(policy, device, args, hyperparameters, hidden_state, done,
                                                    n_states=explore_size//32)

    if args.use_wandb:
        wandb.login(key="cfc00eee102a1e9647b244a40066bfc5f1a96610")
        cfg = vars(args)
        cfg.update(hyperparameters)
        # cfg.update(hyperparameters)
        name = f"{hyperparameters['architecture']}-distill-{np.random.randint(1e5)}"
        wandb.init(project="Coinrun VQMHA - Distill", config=cfg, sync_tensorboard=True,
                   tags=args.wandb_tags, resume="allow", name=name)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(new_policy.parameters(), lr=args.lr)
    new_policy.train()

    for n_explores in range(args.n_explore):
        obs, rew, done = collect_obs(new_policy, first_obs, hidden_state, done, env, n_states=explore_size)
        shuffle_index = np.random.permutation([x for x in range(len(obs))])
        obs_tensor = torch.Tensor(obs[shuffle_index]).to(device)
        done = done[shuffle_index]
        N_batchs = int(len(obs) / batch_size)
        # Y_gold, _ = predict(policy, obs, hidden_state, done)
        Y_gold = predict_in_batches(policy, obs, hidden_state, done)

        loss_diffs = []
        for epoch in range(args.nb_epoch):
            epoch_loss = 0.0
            for i in range(N_batchs):
                optimizer.zero_grad()
                obs_batch = obs_tensor[i * batch_size: (i + 1) * batch_size]
                Y_batch = Y_gold[i * batch_size: (i + 1) * batch_size]

                # Forward pass
                # Y_pred, _ = predict(new_policy, obs_batch, hidden_state, done, with_grad=True)
                if new_policy.has_vq:
                    dist_batch, value_batch, _, commit_loss = new_policy(obs_batch, hidden_state, hidden_state)
                else:
                    dist_batch, value_batch, _ = new_policy(obs_batch, hidden_state, hidden_state)
                Y_pred = dist_batch.logits
                loss = criterion(Y_pred.squeeze(), Y_batch.squeeze())

                if new_policy.has_vq:
                    total_loss = loss + commit_loss
                    total_loss.backward()
                else:
                    # Backward pass
                    loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(Y_batch)

        valid_loss = validate(new_policy, valid_X, valid_Y_gold, criterion, hidden_state)

            # threshold = valid_loss - epoch_loss
            # loss_diffs.append(threshold)
            # if len(loss_diffs) > 10 and valid_loss - epoch_loss > min(loss_diffs) + np.std(loss_diffs):
            #     break

        print(f'Epoch {epoch}: total train loss: {epoch_loss:.5f}, valid loss: {valid_loss:.5f}')

        if args.use_wandb:
            perf_dict = {"Exploration": n_explores, "Epoch": epoch, "Loss": epoch_loss, "Valid_Loss": valid_loss,
                         "Mean_Reward": float(np.mean(rew))}
            wandb.log(perf_dict)
        log.loc[len(log)] = perf_dict.values()
        with open(logdir + '/log-append.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(log.columns)
            writer.writerow(log)
        # Save the model
        if n_explores > ((checkpoint_cnt + 1) * save_every):
            print("Saving model.")
            torch.save({'model_state_dict': new_policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f"{logdir}/model_{n_explores}.pth")
            checkpoint_cnt += 1
    wandb.finish()
    return epoch_loss, np.mean(rew)


def distill_with_lr(args, lr):
    args.lr = lr
    return distill(args, "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_level', type=int, default=int(0), help='start-level for environment')
    parser.add_argument('--num_levels', type=int, default=int(500), help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='easy-200', help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='cpu', required=False, help='whether to use gpu')
    parser.add_argument('--gpu_device', type=int, default=int(0), required=False, help='visible device in CUDA')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--num_checkpoints', type=int, default=int(1), help='number of checkpoints to store')
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--mirror_env', action="store_true", default=False)
    parser.add_argument('--wandb_tags', type=str, nargs='+')
    parser.add_argument('--lr', type=float, default=float(1e-4), help='learning rate')
    parser.add_argument('--batch_size', type=int, default=int(8192), help='batch size')
    parser.add_argument('--nb_epoch', type=int, default=int(1e3), help='number of epochs per exploration')
    parser.add_argument('--explore_size', type=int, default=int(256 * 256), help='size of each exploration')
    parser.add_argument('--n_explore', type=int, default=int(300), help='number of explorations')

    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    args = parser.parse_args()
    args.n_envs = 256
    # If Windows:
    if os.name == "nt":
        args.device = "cpu"
        args.n_envs = 2
        args.use_wandb = False

    # for lr in [1e-4, 1e-3, 1e-2, 1e-5]:

    low = 1e-3
    high = 1e-4
    l_loss, _ = distill_with_lr(args, low)
    h_loss, _ = distill_with_lr(args, low)

    diff = h_loss - l_loss
    if diff > 0:
        low = (low + high) / 2
        high *= 10
    else:
        high = (low + high) / 2
        low /= 10

    l_loss, _ = distill_with_lr(args, low)
    h_loss, _ = distill_with_lr(args, low)
