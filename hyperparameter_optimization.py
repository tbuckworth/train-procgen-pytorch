import argparse
import re

import pandas as pd
import wandb
from helper_local import wandb_login


def get_wandb_performance():
    entity = "ic-ai-safety"
    project = "Cartpole"
    id_tag = "sa_rew"
    wandb_login()
    api = wandb.Api()
    entity, project = entity, project
    runs = api.runs(entity + "/" + project,
                    filters={"$and": [{"tags": id_tag}]}
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

    df["summary.mean_episode_rewards"]

    hp = [x for x in df.columns if re.search("config",x)]


    print("x")
    # logdirs = np.unique([cfg["logdir"] for cfg in config_list])
    # logdir = 'logs/train/acrobot/test/2024-04-25__10-03-20__seed_6033'
    # flt = np.array([cfg["logdir"] == logdir for cfg in config_list])
    #
    # [bool(re.search("acrobot", cfg["logdir"])) for cfg in config_list]
    # [summary.get("problem_name", "") == "acrobot" for summary in summary_list]
    #
    # machines = list(filter(lambda summary: summary.get("problem_name", "") == "acrobot", summary_list))


def select_next_hyperparameters():
    pass


def run_next_hyperparameters():
    pass


def main():
    get_wandb_performance()
    select_next_hyperparameters()
    run_next_hyperparameters()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    while True:
        main()
