import numpy as np
import torch

from common.model import Decoder
from inspect_agent import load_policy
from torch import nn as nn


def retrieve_coinrun_data(frame_file="data/coinrun_frames_500.npy"):
    data = np.load(frame_file)
    np.random.shuffle(data)
    train = 4 * len(data) // 5

    train_data = data[:train]
    valid_data = data[train:]

    train_data_variance = np.var(train_data / 255.0)
    return train_data, valid_data, train_data_variance


def main(logdir):
    action_names, done, env, hidden_state, obs, policy = load_policy(render=False, logdir=logdir, n_envs=2)
    train_data, valid_data, train_data_variance = retrieve_coinrun_data()

    model = policy.embedder
    decoder = Decoder(
        embedding_dim=32,
        num_hiddens=64,
        num_upsampling_layers=3,
        num_residual_layers=2,
        num_residual_hiddens=32,
    )


    nb_epoch = 1e3
    batch_size = 256
    lr = 1e-5
    n_batch = len(train_data) // batch_size

    # TODO: normalize and permute train data, so consistent with IMPALA training

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)
    decoder.train()

    for epoch in range(nb_epoch):
        epoch_loss = 0.
        for i in range(n_batch):
            optimizer.zero_grad()
            obs_batch = train_data[i * batch_size:(i + 1) * batch_size]
            x_batch = latents(model, obs_batch)
            recon = decoder.forward(x_batch)
            loss = criterion(obs_batch, recon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        # TODO: valid_loss = validate()
def latents(model, obs):
    x = model.block1(obs)
    x = model.block2(x)
    x = model.block3(x)
    return x


if __name__ == "__main__":
    # Strong Impala:
    main(logdir="logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033/")
