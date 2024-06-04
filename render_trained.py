import numpy as np
import torch

from symbolic_regression import load_nn_policy

if __name__ == "__main__":
    # logdir = "logs/train/cartpole/test/2024-05-21__13-16-36__seed_6033"
    # logdir = "logs/train/cartpole/test/2024-05-23__07-11-27__seed_6033"
    # logdir = "logs/train/cartpole/test/2024-05-23__09-39-33__seed_6033"
    # trained 200m:
    logdir = "logs/train/cartpole/test/2024-05-24__09-40-11__seed_6033"
    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs=32)
    # env.render_mode = "human"
    # policy.n_rollouts = 3
    n = 32
    for t in range(10):
        policy.temperature = 10**(-t)
        obs = env.reset()
        episodes = 0
        cum_rew = np.zeros(env.n_envs)
        episode_rewards = []
        while episodes < n:
            with torch.no_grad():
                p, v = policy.forward(torch.FloatTensor(obs).to(device=policy.device))
                act = p.sample().cpu().numpy()
            obs, rew, done, info = env.step(act)
            cum_rew += rew
            if np.any(done):
                episodes += np.sum(done)
                episode_rewards += list(cum_rew[done])
        print(f"Temperature: {policy.temperature}, Episodes: {episodes}, Rewards: {np.mean(episode_rewards):.1f}")
