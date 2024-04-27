import re

import numpy as np
from helper_local import free_gpu
from cartpole.cartpole_pre_vec import CartPoleVecEnv


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


if __name__ == "__main__":
    call_wandb()
