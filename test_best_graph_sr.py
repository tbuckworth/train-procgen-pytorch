import os

import pandas as pd
import torch

import wandb
from graph_sr import load_sr_graph_agent, test_agent_mean_reward

from helper_local import wandb_login, get_project, get_latest_file_matching, scp
from symbolic_regression import load_nn_policy


def get_best_wandb_models(project="Cartpole", id_tag="sa_rew", entity="ic-ai-safety"):
    wandb_login()
    api = wandb.Api()
    entity, project = entity, project
    runs = api.runs(entity + "/" + project,
                    filters={"$and": [{"tags": id_tag, "state": "finished"}]}
                    )

    summary_list, config_list, name_list, state_list = [], [], [], []
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

    all_dicts = []
    for s, c, n in zip(summary_list, config_list, name_list):
        s_dict = {f"summary.{k}": v for k, v in s.items()}
        s_dict.update({f"config.{k}": v for k, v in c.items()})
        s_dict["name"] = n
        all_dicts.append(s_dict)

    df = pd.DataFrame.from_dict(all_dicts)

    try:
        y = df["summary.mean_episode_rewards"]
    except KeyError:
        return None, None

    flt = pd.notna(y)
    df = df[flt]
    y = y[flt]
    if len(df) == 0:
        return None, None

    logdir, msg_torch, up_torch, _, _, _ = load_best(df, "summary.loss_transition")
    logdir, _, _, _, r_torch, _ = load_best(df, "summary.loss_reward")
    logdir, _, _, _, _, done_torch = load_best(df, "summary.loss_continuation")
    logdir, _, _, v_torch, _, _ = load_best(df, "summary.mean_episode_rewards", max=True)


    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs=100)
    new_agent = symbolic_agent_constructor(policy, msg_torch, up_torch, v_torch, r_torch, done_torch)

    ns_score_train = test_agent_mean_reward(new_agent, env, "NeuroSymb Train", rounds=100, seed=0)
    ns_score_test = test_agent_mean_reward(new_agent, test_env, "NeuroSymb  Test", rounds=100, seed=0)

    return


def load_best(df, metric, max=False):
    symbdir = df.loc[df[metric].argmin()]["config.symbdir"]
    if max:
        symbdir = df.loc[df[metric].argmax()]["config.symbdir"]
    if not os.path.exists(symbdir):
        print("need to load symbdir:")
        print(symbdir)
        print("continue")
    logdir, ns_agent = load_sr_graph_agent(symbdir)
    ftdir = os.path.join(symbdir, "fine_tune")
    subdir = get_latest_file_matching(r"\d*-\d*-\d*__", 1, ftdir)
    model_path = get_latest_file_matching(r"model_\d*.pth", 1, subdir)
    ns_agent.policy.load_state_dict(torch.load(model_path, map_location=ns_agent.policy.device)["model_state_dict"])
    msg_torch = ns_agent.policy.transition_model.messenger
    up_torch = ns_agent.policy.transition_model.updater
    v_torch = ns_agent.policy.value
    r_torch = ns_agent.policy.r_model
    done_torch = ns_agent.policy.done_model
    return logdir, msg_torch, up_torch, v_torch, r_torch, done_torch


if __name__ == "__main__":
    fixed = {
        "env_name": 'cartpole',
        "exp_name": 'symbreg',  # IMPORTANT!
        "wandb_tags": ["ft034", "graph-transition"],
        "logdir": "logs/train/cartpole/test/2024-06-08__00-54-02__seed_6033",
    }
    project = get_project(fixed["env_name"], fixed["exp_name"])
    id_tag = fixed["wandb_tags"][0]
    get_best_wandb_models(project, id_tag)






