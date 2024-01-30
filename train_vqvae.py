import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch

from common import orthogonal_init
from common.model import VQVAE
from email_results import send_image
from helper import create_logdir, plot_reconstructions, get_hyperparams
from torch import nn as nn

try:
    import wandb
except ImportError:
    pass


def retrieve_coinrun_data(frame_file="data/coinrun_random_frames_500.npy"):
    data = np.load(frame_file)
    data = data.transpose(0, 3, 1, 2)
    np.random.shuffle(data)
    train = 9 * len(data) // 10

    train_data = data[:train]
    valid_data = data[train:]
    return train_data, valid_data, 0.0


def validate(data, model, criterion, device, batch_size=2048):
    total_loss = 0.
    n_evals = 0
    for i in range(len(data) // batch_size + 1):
        data_slice = data[i * batch_size:(i + 1) * batch_size]
        obs_batch = torch.Tensor(data_slice / 255.0).to(device)
        outputs = model(obs_batch)
        recon = outputs["x_recon"]
        loss = criterion(obs_batch, recon)
        total_loss += loss.item()
        n_evals += 1
    return total_loss / n_evals


def get_recons(model, train_data, valid_data, device):
    return {"train_reconstructions": encode_and_decode(model, device, train_data),
            "valid_reconstructions": encode_and_decode(model, device, valid_data),
            "train_batch": train_data.transpose(0, 2, 3, 1),
            "valid_batch": valid_data.transpose(0, 2, 3, 1)}


def encode_and_decode(model, device, train_data):
    x = torch.Tensor(train_data / 255.0).to(device)
    outputs = model(x)
    r = outputs["x_recon"]
    return r.detach().cpu().numpy().transpose(0, 2, 3, 1)


def train_vqvae(args):
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    if args.detect_nan:
        torch.autograd.set_detect_anomaly(True)

    train_data, valid_data, _ = retrieve_coinrun_data()
    in_channels = train_data.shape[1]

    hyperparameters = get_hyperparams(args.param_name)

    model = VQVAE(in_channels, **hyperparameters)
    model.to(device)
    model.apply(orthogonal_init)


    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    lr = args.lr
    checkpoint_cnt = 0
    save_every = nb_epoch // args.num_checkpoints
    checkpoints = [50, 100] + [(i+1) * save_every for i in range(args.num_checkpoints)] + [nb_epoch - 1]
    checkpoints.sort()
    n_batch = len(train_data) // batch_size

    logdir = create_logdir(args, 'vqvae', 'coinrun', f'vqvae')
    log = pd.DataFrame(columns=["Epoch", "Loss", "Valid_Loss"])

    if args.use_wandb:
        wandb.login(key="cfc00eee102a1e9647b244a40066bfc5f1a96610")
        cfg = vars(args)
        cfg.update(hyperparameters)
        name = f"VQVAE-decode-{np.random.randint(1e5)}"
        wandb.init(project="Coinrun - VQVAE", config=cfg, sync_tensorboard=True,
                   tags=args.wandb_tags, resume="allow", name=name)

    if args.use_max:
        def criterion(output, target):
            se = output - target ** 2
            loss = torch.mean(se) + torch.max(se)
            return loss
    else:
        criterion = nn.MSELoss()

    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer {args.optim} must be one of 'SGD','Adam'.")
    model.train()

    for epoch in range(nb_epoch):
        epoch_loss = 0.
        n_evals = 0
        for i in range(n_batch):
            optimizer.zero_grad()
            data_slice = train_data[i * batch_size:(i + 1) * batch_size]
            obs_batch = torch.Tensor(data_slice / 255.0).to(device)

            outputs = model(obs_batch)

            recon = outputs["x_recon"]
            loss = criterion(obs_batch, recon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_evals += 1  # len(obs_batch)
        epoch_loss /= n_evals
        valid_loss = validate(valid_data, model, criterion, device)
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
        if epoch == checkpoints[checkpoint_cnt]:
            print("Saving model.")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f"{logdir}/model_{epoch}.pth")
            checkpoint_cnt += 1
            send_reconstruction_update(model, epoch, logdir, train_data, valid_data, device)
    wandb.finish()


def send_reconstruction_update(model, epoch, logdir, train_data, valid_data, device):
    plot_file = os.path.join(logdir, f"reconstructions_{epoch}.png")
    recons = get_recons(model, train_data[:32], valid_data[:32], device)
    plot_reconstructions(recons, plot_file)
    send_image(plot_file, "Coinrun Decoder Reconstructions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu', required=False, help='whether to use gpu')
    parser.add_argument('--gpu_device', type=int, default=int(0), required=False, help='visible device in CUDA')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--num_checkpoints', type=int, default=int(5), help='number of checkpoints to store')
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--wandb_tags', type=str, nargs='+')
    parser.add_argument('--lr', type=float, default=float(1e-4), help='learning rate')
    parser.add_argument('--batch_size', type=int, default=int(256), help='batch size')
    parser.add_argument('--nb_epoch', type=int, default=int(1000), help='number of epochs per exploration')
    parser.add_argument('--optim', type=str, default="SGD", help='Optimizer: "SGD" or "Adam"')
    parser.add_argument('--use_max', action="store_true", default=False, help="Add max squared error to MSE loss?")
    parser.add_argument('--detect_nan', action="store_true", default=False)
    parser.add_argument('--param_name', type=str, default='vq-vae', help='hyper-parameter ID')

    args = parser.parse_args()
    # If Windows:
    if os.name == "nt":
        args.device = "cpu"
        args.use_wandb = False

    print(f"Device:{args.device}")
    train_vqvae(args)
