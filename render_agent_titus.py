import torch

from symbolic_regression import load_nn_policy

# This is cartpole continuous graph optimal - was tested incorrectly:
# todo: test this and then sr it:
# "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033"

def render_agent():
    # This was mlp that was incorrectly tested on training env, so i thought was generalizing perfectly lol
    logdir = "logs/train/cartpole_continuous/test/2024-09-06__17-00-29__seed_6033"
    # first Ant:
    logdir = "logs/train/Ant-v4/pure-graph/2024-09-08__12-46-54__seed_6033"
    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs=2, render=True)

    obs = env.reset()
    for i in range(2000):
        obs = torch.FloatTensor(obs).to(device=policy.device)
        with torch.no_grad():
            dist, value, _ = policy(obs, None, None)
            act = dist.sample().cpu().numpy()
        obs, rew, done, info = env.step(act)

    # tr_obs = env.reset()
    # tr_obs = torch.FloatTensor(tr_obs).to(device=policy.device)
    # tr_dist, tr_value, _ = policy(tr_obs, None, None)

    print("done")

if __name__ == "__main__":
    render_agent()

