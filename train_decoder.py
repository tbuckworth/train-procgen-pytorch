import argparse
import csv
import random

import numpy as np
import pandas as pd
import torch

from common.model import Decoder
from helper import create_logdir
from inspect_agent import load_policy
from torch import nn as nn

try:
    import wandb
except ImportError:
    pass

def retrieve_coinrun_data(device, frame_file="data/coinrun_frames_500.npy"):
    data = np.load(frame_file)
    data = data.transpose(0, 3, 1, 2)
    np.random.shuffle(data)
    train = 9 * len(data) // 10

    train_data = data[:train]
    valid_data = data[train:]

    train_data_variance = np.var(train_data / 255.0)

    data_t = torch.Tensor(train_data, dtype=torch.float32).to(device) / 255.0
    data_v = torch.Tensor(valid_data, dtype=torch.float32).to(device) / 255.0

    return data_t, data_v, train_data_variance


def validate(data, x, decoder, criterion):
    #TODO: may need to split this into chunks
    recon = decoder.forward(x)
    loss = criterion(recon, data)
    return loss.item() / len(data)


def train_decoder(args, trained_model_folder):
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    action_names, done, env, hidden_state, obs, policy = load_policy(render=False, logdir=trained_model_folder, n_envs=2)
    train_data, valid_data, train_data_variance = retrieve_coinrun_data(device)

    hyperparameters = {"architecture": "VQ Decoder"}

    encoder = policy.embedder
    decoder = Decoder(
        embedding_dim=32,
        num_hiddens=64,
        num_upsampling_layers=3,
        num_residual_layers=2,
        num_residual_hiddens=32,
    )
    decoder.device = device

    # TODO: may need to split this into chunks
    valid_latents = latents(encoder, valid_data)

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
            obs_batch = train_data[i * batch_size:(i + 1) * batch_size]
            x_batch = latents(encoder, obs_batch)
            recon = decoder.forward(x_batch)
            loss = criterion(obs_batch, recon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_evals += len(obs_batch)
        epoch_loss /= n_evals
        valid_loss = validate(valid_data, valid_latents, decoder, criterion)
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




def latents(model, obs):
    x = model.block1(obs)
    x = model.block2(x)
    x = model.block3(x)
    return x


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
    # Strong Impala:
    train_decoder(logdir="logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/")
