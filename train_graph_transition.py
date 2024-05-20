import csv

import numpy as np
import pandas as pd
import torch
import wandb
from torch import optim
from torch.nn import MSELoss

from common.model import GraphTransitionModel
from helper_local import initialize_model, get_hyperparams, create_logdir, DictToArgs
from symbolic_regression import load_nn_policy

def predict(policy, obs):
    with torch.no_grad():
        obs = torch.FloatTensor(obs).to(device=policy.device)
        p, v = policy.hidden_to_output(policy.embedder(obs))
        act = p.sample()
        return act.cpu().numpy(), v.cpu().numpy()


def collect_data(policy, env, n):
    observation = env.reset()
    act, v = predict(policy, observation)
    X = observation
    A = act
    V = v
    D = np.zeros(len(X))
    while len(X) < n:
        observation, rew, done, info = env.step(act)
        act, v = predict(policy, observation)
        X = np.append(X, observation, axis=0)
        A = np.append(A, act, axis=0)
        V = np.append(V, v, axis=0)
        D = np.append(D, done, axis=0)
    X = torch.FloatTensor(X).to(device=policy.device)
    A = torch.FloatTensor(A).to(device=policy.device)
    V = torch.FloatTensor(V).to(device=policy.device)
    D = torch.FloatTensor(D).to(device=policy.device)
    return X, A, V, D

if __name__ == "__main__":
    n_envs = 32
    learning_rate = 5e-4
    rounds_per_epoch = 1000
    epochs = 100
    num_checkpoints = 10
    seed = 6033
    use_wandb = True
    args = DictToArgs({
        ""
    })
    anchor_logdir = "logs/train/cartpole/test/2024-05-01__11-17-16__seed_6033"
    policy, env, _, test_env = load_nn_policy(anchor_logdir, n_envs)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    _, observation_shape, temp_policy = initialize_model(device, env, get_hyperparams("graph-transition"))
    transition_model = temp_policy.transition_model
    optimizer = optim.Adam(transition_model.parameters(), lr=learning_rate, eps=1e-5)


    checkpoint_cnt = 0
    save_every = epochs // num_checkpoints
    checkpoints = [50, 100] + [(i+1) * save_every for i in range(num_checkpoints)] + [epochs - 1]
    checkpoints.sort()

    logdir = create_logdir(DictToArgs({"seed": seed}), 'train-transition', 'cartpole', f'graph-transition')
    log = pd.DataFrame(columns=["Epoch", "Loss"])

    if use_wandb:
        wandb.login(key="cfc00eee102a1e9647b244a40066bfc5f1a96610")
        cfg = vars(args)
        cfg.update(hyperparameters)
        name = f"{hyperparameters['architecture']}{np.random.randint(1e5)}"
        wandb.init(project=project_name, config=cfg, sync_tensorboard=True,
                   tags=args.wandb_tags, resume="allow", name=name)


    for epoch in range(epochs):
        obs_batch, act_batch, val_batch, done_batch = collect_data(policy, env, rounds_per_epoch)

        flt = done_batch[:-n_envs] == 0

        nobs_guess = transition_model(obs_batch[:-n_envs][flt], act_batch[:-n_envs][flt])
        t_loss = MSELoss()(nobs_guess, obs_batch[n_envs:][flt])
        t_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.use_wandb:
            perf_dict = {"Epoch": epoch, "Loss": t_loss.item()}
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
            torch.save({'model_state_dict': transition_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f"{logdir}/model_{epoch}.pth")
            checkpoint_cnt += 1
    wandb.finish()



