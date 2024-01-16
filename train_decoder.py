import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch

from common.model import Decoder
from helper import create_logdir, impala_latents
from inspect_agent import load_policy
from torch import nn as nn

try:
    import wandb
except ImportError:
    pass

def retrieve_coinrun_data(device, frame_file="data/coinrun_random_frames_500.npy"):
    data = np.load(frame_file)
    data = data.transpose(0, 3, 1, 2)
    np.random.shuffle(data)
    train = 9 * len(data) // 10

    train_data = data[:train]
    valid_data = data[train:]
    return train_data, valid_data, 0.0
    # train_data_variance = np.var(train_data / 255.0)

    data_t = torch.Tensor(train_data).to(device) / 255.0
    data_v = torch.Tensor(valid_data).to(device) / 255.0
    train_data_variance = torch.var(train_data)
    return data_t, data_v, train_data_variance


def validate(data, encoder, decoder, criterion, device, batch_size=2048):
    # #TODO: may need to split this into chunks
    # recon = decoder.forward(x)
    # loss = criterion(recon, data)
    # return loss.item() / len(data)
    total_loss = 0.
    n_evals = 0
    for i in range(len(data)//batch_size+1):
        data_slice = data[i * batch_size:(i + 1) * batch_size]
        obs_batch = torch.Tensor(data_slice/255.0).to(device)
        x_batch = impala_latents(encoder, obs_batch)
        recon = decoder.forward(x_batch)
        loss = criterion(obs_batch, recon)
        total_loss += loss.item()
        n_evals += len(obs_batch)
    return total_loss / n_evals


def train_decoder(args, trained_model_folder):
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    action_names, done, env, hidden_state, obs, policy = load_policy(render=False, logdir=trained_model_folder, n_envs=2)
    train_data, valid_data, train_data_variance = retrieve_coinrun_data(device)

    hyperparameters = {"architecture": "VQ Decoder"}

    encoder = policy.embedder
    encoder.to(device)
    decoder = Decoder(
        embedding_dim=32,
        num_hiddens=64,
        num_upsampling_layers=3,
        num_residual_layers=2,
        num_residual_hiddens=32,
    )
    decoder.to(device)

    # TODO: valid_latents may be too large
    # valid_latents = latents_by_batch(encoder, valid_data, device, batch_size=2048)

    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    lr = args.lr
    checkpoint_cnt = 0
    save_every = nb_epoch // args.num_checkpoints
    n_batch = len(train_data) // batch_size


    logdir = create_logdir(args, 'decode', 'coinrun', 'decode')
    log = pd.DataFrame(columns=["Epoch", "Loss", "Valid_Loss"])


    if args.use_wandb:
        wandb.login(key="cfc00eee102a1e9647b244a40066bfc5f1a96610")
        cfg = vars(args)
        cfg.update(hyperparameters)
        name = f"{hyperparameters['architecture']}-decode-{np.random.randint(1e5)}"
        wandb.init(project="Coinrun - Decode", config=cfg, sync_tensorboard=True,
                   tags=args.wandb_tags, resume="allow", name=name)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)
    decoder.train()

    for epoch in range(nb_epoch):
        epoch_loss = 0.
        n_evals = 0
        for i in range(n_batch):
            optimizer.zero_grad()
            data_slice = train_data[i * batch_size:(i + 1) * batch_size]
            obs_batch = torch.Tensor(data_slice/255.0).to(device)
            x_batch = impala_latents(encoder, obs_batch)
            recon = decoder.forward(x_batch)
            loss = criterion(obs_batch, recon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_evals += len(obs_batch)
        epoch_loss /= n_evals
        valid_loss = validate(valid_data, encoder, decoder, criterion, device)
        print(f"Epoch {epoch}: train loss: {epoch_loss:.5f}, {valid_loss:.5f}")

        if args.use_wandb:
            perf_dict = {"Epoch": epoch, "Loss": epoch_loss, "Valid_Loss": valid_loss}
            wandb.log(perf_dict)
        log.loc[len(log)] = perf_dict.values()
        with open(logdir + '/log-append.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(log.columns)
            writer.writerow(log)
        # Save the model
        if epoch > ((checkpoint_cnt + 1) * save_every):
            print("Saving model.")
            torch.save({'model_state_dict': decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f"{logdir}/model_{epoch}.pth")
            checkpoint_cnt += 1
    wandb.finish()


# def latents_by_batch(encoder, valid_data, device, batch_size):
#     for i in range(len(valid_data) // batch_size + 1):
#         x = torch.Tensor(valid_data[i * batch_size:(i + 1) * batch_size] / 255.0).to(device)
#         l = latents(encoder, x)
#         if valid_latents is None:
#             valid_latents = l
#         else:
#             valid_latents = torch.cat((l, valid_latents), dim=0)
#     return valid_latents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', required=False, help='whether to use gpu')
    parser.add_argument('--gpu_device', type=int, default=int(0), required=False, help='visible device in CUDA')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--num_checkpoints', type=int, default=int(5), help='number of checkpoints to store')
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--wandb_tags', type=str, nargs='+')
    parser.add_argument('--lr', type=float, default=float(1e-5), help='learning rate')
    parser.add_argument('--batch_size', type=int, default=int(256), help='batch size')
    parser.add_argument('--nb_epoch', type=int, default=int(100), help='number of epochs per exploration')
    args = parser.parse_args()
    # If Windows:
    if os.name == "nt":
        args.device = "cpu"
        args.use_wandb = False
    # Strong Impala:
    train_decoder(args, "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/")
