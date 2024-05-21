import numpy as np
import torch

from symbolic_regression import load_nn_policy

if __name__ == "__main__":
    logdir = "logs/train/cartpole/test/2024-05-21__13-16-36__seed_6033"
    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs=8)
    # env.render_mode = "human"
    n = 40
    for i in range(6):
        policy.n_rollouts = i
        obs = env.reset()
        episodes = []
        cum_rew = np.zeros(len(env.action_space.n))
        episode_rewards = []
        while len(episodes < n):
            p, v = policy.forward(torch.FloatTensor(obs).to(device=policy.device))
            act = p.sample()
            obs, rew, done, info = env.step(act)
            cum_rew += rew
            if np.any(done):
                episodes += np.sum(done)
                episode_rewards += list(cum_rew[done])
        print(f"Rollouts: {i}, Episodes: {episodes}, Rewards: {np.mean(episode_rewards):.1f}")
