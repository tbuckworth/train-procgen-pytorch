import numpy as np
from pysr import PySRRegressor
import torch

from helper import get_config
from inspect_agent import load_policy


def find_model(X, Y):
    model = PySRRegressor(
        niterations=40,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )

    model.fit(X, Y)

    print(model)


def load_nn_policy(logdir):
    cfg = get_config(logdir)
    n_envs = 2
    action_names, done, env, hidden_state, obs, policy, storage = load_policy(False, logdir, n_envs=n_envs,
                                                                              hparams="hard-500-impalafsqmha",
                                                                              start_level=cfg["start_level"],
                                                                              num_levels=cfg["num_levels"])
    return policy, env, obs, storage


def drop_first_dim(arr):
    shp = np.array(arr.shape)
    new_shape = tuple(np.concatenate(([np.prod(shp[:2])], shp[2:])))
    return arr.reshape(new_shape)

def generate_data(policy, env, observation, n):
    x, y, act = sample_latent_output(policy, observation)
    X = np.expand_dims(x, 0)
    Y = np.expand_dims(y, 0)
    while len(x) < n:
        observation, rew, done, info = env.step(act)
        x, y, act = sample_latent_output(policy, observation)
        X = np.append(X, np.expand_dims(x, 0), axis=0)
        Y = np.append(Y, np.expand_dims(y, 0), axis=0)
    return drop_first_dim(X), drop_first_dim(Y)


def sample_latent_output(policy, observation):
    obs = torch.FloatTensor(observation).to(policy.device)
    x = policy.embedder.forward_to_pool(obs)
    dist, value = policy.hidden_to_output(x)
    y = dist.logits.detach().cpu().numpy()
    act = dist.sample()
    return x.cpu().numpy(), y, act.cpu().numpy()


if __name__ == "__main__":
    logdir = "logs/train/coinrun/coinrun/2024-02-20__18-02-16__seed_6033"
    policy, env, obs, storage = load_nn_policy(logdir)
    X, Y = generate_data(policy, env, obs, n=int(1e5))
    find_model(X, Y)
