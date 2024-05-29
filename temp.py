import os
import re

import numpy as np
import pandas as pd
from numpy import cos,sin,pi

from common.env.env_constructor import get_env_constructor
from create_sh_files import train_hparams, symbreg_hparams
from helper_local import free_gpu, get_config
from cartpole.cartpole_pre_vec import CartPoleVecEnv
import gymnasium
from discrete_env.acrobot_pre_vec import rk4 as rk4_pre_vec
# from discrete_env.acrobot import rk4 as rk4_gymnasium

def compute_pairwise_affinities(X, perplexity=30.0, epsilon=1e-8):
    # Compute pairwise Euclidean distances
    distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

    # Compute joint probabilities (affinities)
    affinities = np.zeros_like(distances)
    beta = 1.0 / (2.0 * perplexity)

    for i in range(X.shape[0]):
        # Use binary search to find sigma for each point
        sigma_low, sigma_high = 0.0, np.inf
        target_entropy = np.log2(perplexity)

        for _ in range(50):
            sigma = (sigma_low + sigma_high) / 2.0
            p_i = np.exp(-distances[i] ** 2 * beta)
            sum_p_i = np.sum(p_i) - p_i[i]
            entropy = -np.sum(p_i / sum_p_i * np.log2(p_i / sum_p_i + epsilon))

            if np.abs(entropy - target_entropy) < 1e-5:
                break
            elif entropy > target_entropy:
                sigma_high = sigma
            else:
                sigma_low = sigma

        affinities[i] = p_i / sum_p_i

    return affinities


def t_sne(X, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
    # Initialize low-dimensional representation randomly
    Y = np.random.randn(X.shape[0], n_components)

    # Perform the optimization using gradient descent
    for _ in range(n_iter):
        q_ij = 1.0 / (1.0 + np.linalg.norm(Y[:, np.newaxis] - Y, axis=2) ** 2)
        q_ij /= np.sum(q_ij)

        grad = 4.0 * np.dot((compute_pairwise_affinities(Y, perplexity) - q_ij).T, Y - Y[:, np.newaxis])

        Y -= learning_rate * grad

    return Y


def tsne_thing():
    # Example usage
    np.random.seed(42)
    X = np.random.rand(100, 10)

    # Apply t-SNE
    X_tsne_custom = t_sne(X)

    # Visualize the results
    import matplotlib.pyplot as plt

    plt.scatter(X_tsne_custom[:, 0], X_tsne_custom[:, 1])
    plt.title('Custom t-SNE Visualization')
    plt.show()


def symbolic_regression_function(obs):
    x0, x1, x2, x3 = obs[0]
    if x0 > -1.62 * (3 * x2 + x3 + x1):
        return np.ones(len(obs))
    return np.zeros(len(obs))


def some_function():
    is_valid = False
    n_envs = 2
    env_args = {"n_envs": n_envs,
                "env_name": "CartPole-v1",
                "degrees": 12,
                "h_range": 2.4,
                }
    if is_valid:
        env_args["degrees"] = 9
        env_args["h_range"] = 1.8
    # env = create_cartpole_env_pre_vec(env_args, render=True, normalize_rew=False)

    env = CartPoleVecEnv(n_envs, degrees=9, h_range=1.8, max_steps=500, render_mode="human")
    obs = env.reset()
    while True:
        act = symbolic_regression_function(obs)
        obs, rew, done, info = env.step(act)


def flappy_bird():
    # import flappy_bird_gymnasium
    import gymnasium

    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)

    obs, _ = env.reset()
    while True:
        # Next action:
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        # Processing:
        obs, reward, terminated, _, info = env.step(action)

        # Checking if the player is still alive
        if terminated:
            break

    env.close()


def call_wandb():
    import pandas as pd
    import wandb

    api = wandb.Api()
    entity, project = "ic-ai-safety", "Symb Reg"
    runs = api.runs(entity + "/" + project,
                    # filters={"$and": [{"summary.problem_name": "acrobot"}]}
                    )

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )

    runs_df.to_csv("project.csv")

    all_dicts = []
    for s, c, n in zip(summary_list, config_list, name_list):
        s_dict = {f"summary.{k}": v for k, v in s.items()}
        s_dict.update({f"config.{k}": v for k, v in c.items()})
        s_dict["name"] = n
        all_dicts.append(s_dict)

    df = pd.DataFrame.from_dict(all_dicts)

    logdirs = np.unique([cfg["logdir"] for cfg in config_list])
    logdir = 'logs/train/acrobot/test/2024-04-25__10-03-20__seed_6033'
    flt = np.array([cfg["logdir"] == logdir for cfg in config_list])

    [bool(re.search("acrobot", cfg["logdir"])) for cfg in config_list]
    [summary.get("problem_name", "") == "acrobot" for summary in summary_list]

    machines = list(filter(lambda summary: summary.get("problem_name", "") == "acrobot", summary_list))


def dsdt_gymnasium(self, s_augmented):
    m1 = self.LINK_MASS_1
    m2 = self.LINK_MASS_2
    l1 = self.LINK_LENGTH_1
    lc1 = self.LINK_COM_POS_1
    lc2 = self.LINK_COM_POS_2
    I1 = self.LINK_MOI
    I2 = self.LINK_MOI
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]
    d1 = (
            m1 * lc1 ** 2
            + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
    )
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
    phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
    phi1 = (
            -m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
    )
    if self.book_or_nips == "nips":
        # the following line is consistent with the description in the
        # paper
        ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    else:
        # the following line is consistent with the java implementation and the
        # book
        ddtheta2 = (
                           a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2
                   ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

def dsdt_pre_vec(self, s_augmented):
    theta1, theta2, dtheta1, dtheta2, g, l1, l2, m1, m2, lc1, lc2, link_moi = self.state.T

    I1 = link_moi
    I2 = link_moi
    a = s_augmented[:, -1]
    theta1 = s_augmented[:, 0]
    theta2 = s_augmented[:, 1]
    dtheta1 = s_augmented[:, 2]
    dtheta2 = s_augmented[:, 3]
    d1 = (
            m1 * lc1 ** 2
            + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
    )
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
    phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
    phi1 = (
            -m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
    )
    if self.book_or_nips == "nips":
        # the following line is consistent with the description in the
        # paper
        ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    else:
        # the following line is consistent with the java implementation and the
        # book
        ddtheta2 = (
                           a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2
                   ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return dtheta1, dtheta2, ddtheta1, ddtheta2, np.zeros_like(dtheta1)


def acrobot_gymnasium():
    env = gymnasium.make("Acrobot-v1", render_mode="human")
    env.reset()
    # env.render()
    for i in range(100):
        # actions = [env.action_space.sample() for _ in range(self.env.n_envs)]
        obs, rew, done, trunc, info = env.step(env.action_space.sample())
        if done:
            print(i)

    state = env.state

    s_augmented = np.append(state, -1)
    y_gold = np.array(dsdt_gymnasium(env, s_augmented))

    env_pv = get_env_constructor("acrobot")(None, {})

    #s_aug_v = np.stack((s_augmented, s_augmented))

    env_pv.state[:, :env_pv.i_g] = state

    s_aug_v = np.tile(s_augmented, env_pv.n_envs).reshape(env_pv.n_envs, len(s_augmented))

    np.stack(dsdt_pre_vec(env_pv, s_aug_v)).T

    rk4_gymnasium(lambda x: dsdt_gymnasium(env,x), s_augmented, [0, 0.2])

    rk4_pre_vec(lambda x: dsdt_pre_vec(env_pv, x), s_aug_v, [0, 0.2])

    test_func = lambda y0: np.allclose(np.asarray(dsdt_pre_vec(env_pv,y0)).T[0],
                                       dsdt_gymnasium(env, y0[0]))

def get_n_create_sh_files():
    hparams = train_hparams()
    n_experiments = np.prod([len(hparams[x]) for x in hparams.keys()])
    print(f"Train experiments: {n_experiments}.")
    hparams = symbreg_hparams()
    n_experiments = np.prod([len(hparams[x]) for x in hparams.keys()])
    print(f"Symbreg experiments: {n_experiments}.")


def extract_hyperparams_symbreg():
    dirs = [
        "logs/train/mountain_car/test/2024-05-22__20-29-39__seed_30/symbreg/2024-05-23__01-17-39",
        "logs/train/cartpole_swing/test/2024-05-01__14-19-53__seed_6033/symbreg/2024-05-13__10-27-13",
        # "logs/train/mountain_car/test/2024-05-03__15-46-58__seed_6033/symbreg/2024-05-14__00-45-59",
        # "logs/train/acrobot/test/2024-05-01__12-22-24__seed_6033/symbreg/2024-05-08__13-36-18",
        "logs/train/cartpole/test/2024-05-01__11-17-16__seed_6033/symbreg/2024-05-03__11-14-03",
        # "logs/train/acrobot/test/2024-05-01__12-22-24__seed_6033/symbreg/2024-05-02__02-06-38",
        "logs/train/cartpole/test/2024-05-01__11-17-16__seed_6033/symbreg/2024-05-02__13-37-11",
    ]
    cfgs = [get_config(symbdir) for symbdir in dirs]
    df = pd.DataFrame.from_dict(cfgs)

    out = {}
    for column in df:
        try:
            vals = np.unique(df[column])
            v_str = [f"{v}" for v in list(vals)]
            out[column] = ','.join(v_str)
        except Exception as e:
            continue

    df2 = pd.DataFrame(out, index=[0])
    latex = df2.T.to_latex()
    print(latex)
    print("pass")


if __name__ == "__main__":
    get_n_create_sh_files()
