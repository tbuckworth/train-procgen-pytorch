import argparse
import copy
import os
import re
import time

import numpy as np
import pandas as pd
import sympy

import wandb
from common.model import NBatchPySRTorch

from email_results import send_images_first_last
from graph_sr import fine_tune
from symbolic_regression import load_nn_policy
from symbreg.extra_mappings import get_extra_torch_mappings

from pysr import PySRRegressor
# Important! keep torch after pysr
# import torch
from common.env.procgen_wrappers import create_env
from helper_local import get_config, get_path, balanced_reward, \
    floats_to_dp, dict_to_html_table, wandb_login, add_symbreg_args, \
    sigmoid, get_saved_hyperparams, n_params, create_symb_dir_if_exists

from matplotlib import pyplot as plt

# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\PycharmProjects\train-procgen-pytorch\venv\julia_env\pyjuliapkg\install\bin"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\AppData\Local\Microsoft\WindowsApps"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\.julia\juliaup\julia-1.10.0+0.x64.w64.mingw32\bin"

pysr_loss_functions = {
    "sigmoid": "SigmoidLoss()",
    "exp": "ExpLoss()",
    "l2marg": "L2MarginLoss()",
    "logitmarg": "LogitMarginLoss()",
    "perceptron": "PerceptronLoss()",
    "logitdist": "LogitDistLoss()",
    "mse": "loss(prediction, target) = (prediction - target)^2",
    "mce": "loss(prediction, target) = abs(prediction - target)^3",
    "capped_sigmoid": "loss(y_hat, y) = 1 - tanh(y*y_hat) + abs(y_hat-y)",
    "bce": "loss(y_hat, y) = -y * log(y_hat + 0.00001) - (1 - y) * log(1 - y_hat + 0.00001)",
    "lbce": "loss(z, y) = z - z*y + log(1 + exp(-z))"
}


def find_model(X, Y, symbdir, save_file, weights, args):
    model = PySRRegressor(
        verbosity=0,
        equation_file=get_path(symbdir, save_file),
        maxsize=args.maxsize,
        niterations=args.iterations,  # < Increase me for better results
        binary_operators=args.binary_operators,
        unary_operators=args.unary_operators,
        weights=weights,
        denoise=args.denoise,
        extra_sympy_mappings={"greater": lambda x, y: sympy.Piecewise((1.0, x > y), (0.0, True)),
                              },
        elementwise_loss=pysr_loss_functions[args.loss_function],
        timeout_in_seconds=args.timeout_in_seconds,
        populations=args.populations,
        procs=args.procs,
        batching=args.data_size > 1000,
        weight_optimize=0.001 if args.ncycles_per_iteration > 550 else 0.0,
        ncycles_per_iteration=args.ncycles_per_iteration,
        bumper=args.bumper,
        model_selection=args.model_selection,
        extra_torch_mappings=get_extra_torch_mappings(),
        nested_constraints={"relu": {"relu": 0},
                            "exp": {"exp": 0, "square": 1},
                            "square": {"square": 0, "exp": 1},
                            },
    )
    print("fitting model")
    start = time.time()
    model.fit(X, Y)
    elapsed = time.time() - start
    eqn_str = get_best_str(model)
    print(eqn_str)
    # model.extra_torch_mappings = {sympy.Piecewise: lambda x, y: torch.where(x > y, 1.0, 0.0),
    #                               sympy.functions.elementary.piecewise.ExprCondPair: None,
    #                               sympy.logic.boolalg.BooleanTrue: None}
    return model, elapsed


def get_best_str(model, split='\n'):
    best = model.get_best()
    if type(best) == list:
        return split.join([x.equation for x in model.get_best()])
    return model.get_best().equation


def drop_first_dim(arr):
    shp = np.array(arr.shape)
    new_shape = tuple(np.concatenate(([np.prod(shp[:2])], shp[2:])))
    return arr.reshape(new_shape)


def generate_data(agent, env, n):
    Obs = env.reset()
    M_in, M_out, U_in, U_out, Vm_in, Vm_out, Vu_in, Vu_out = agent.sample(Obs)
    act = agent.forward(Obs)
    act[::2] = np.random.randint(0, env.action_space.n, len(act))[::2]
    while len(M_in) < n:
        observation, rew, done, info = env.step(act)
        m_in, m_out, u_in, u_out, vm_in, vm_out, vu_in, vu_out = agent.sample(observation)
        act = agent.forward(observation)
        act[::2] = np.random.randint(0, env.action_space.n, len(act))[::2]

        M_in = np.append(M_in, m_in, axis=0)
        M_out = np.append(M_out, m_out, axis=0)
        U_in = np.append(U_in, u_in, axis=0)
        U_out = np.append(U_out, u_out, axis=0)
        Vm_in = np.append(Vm_in, vm_in, axis=0)
        Vm_out = np.append(Vm_out, vm_out, axis=0)
        Vu_in = np.append(Vu_in, vu_in, axis=0)
        Vu_out = np.append(Vu_out, vu_out, axis=0)

    return M_in, M_out, U_in, U_out, Vm_in, Vm_out, Vu_in, Vu_out


def test_agent_balanced_reward(agent, env, print_name, n=40):
    performance_track = {}
    episodes = 0
    obs = env.reset()
    act = agent.forward(obs)
    while episodes < n:
        obs, rew, done, info = env.step(act)
        act = agent.forward(obs)
        true_average_reward = balanced_reward(done, info, performance_track)
        if np.any(done):
            episodes += np.sum(done)
    print(f"{print_name}:\tEpisode:{episodes}\tBalanced Reward:{true_average_reward:.2f}")
    return true_average_reward


def trial_agent_mean_reward(agent, env, print_name, n=40, return_values=False, seed=0, print_results=True, reset=True):
    print(print_name)
    episodes = 0
    if reset:
        obs = env.reset(seed=seed)
    else:
        obs = env.reset()
    act = agent.forward(obs)
    cum_rew = np.zeros(len(act))
    episode_rewards = []
    while episodes < n:
        obs, rew, done, info = env.step(act)
        cum_rew += rew
        act = agent.forward(obs)
        if np.any(done):
            episodes += np.sum(done)
            episode_rewards += list(cum_rew[done])
            cum_rew[done] = 0
    if print_results:
        print(f"{print_name}:\tEpisode:{episodes}\tMean Reward:{np.mean(episode_rewards):.2f}")
    if return_values:
        return episode_rewards
        # return {"mean":np.mean(episode_rewards), "std":np.std(episode_rewards)}
    return np.mean(episode_rewards)


# def sample_policy_with_symb_model(model, policy, observation):
#     with torch.no_grad():
#         obs = torch.FloatTensor(observation).to(policy.device)
#         x = policy.embedder.forward_to_pool(obs)
#         h = model(x)
#         dist, value = policy.hidden_to_output(h)
#         # y = dist.logits.detach().cpu().numpy()
#         act = dist.sample()
#     return act.cpu().numpy()


def get_coinrun_test_env(logdir, n_envs):
    cfg = get_config(logdir)
    hyperparameters = get_saved_hyperparams(logdir)
    hyperparameters["n_envs"] = n_envs
    env_args = {"num": hyperparameters["n_envs"],
                "env_name": "coinrun",
                "start_level": cfg["start_level"] + cfg["num_levels"] + 1,  # 325
                "num_levels": 0,
                "paint_vel_info": True,
                "distribution_mode": "hard"}
    normalize_rew = hyperparameters.get('normalize_rew', True)
    try:
        cfg = get_config(logdir)
        mirror_some = cfg["mirror_env"]
    except Exception:
        mirror_some = True
    env = create_env(env_args, False, normalize_rew, mirror_some)
    return env


def send_full_report(df, logdir, symbdir, model, args, dfs):
    # load log csv
    dfl = pd.read_csv(os.path.join(logdir, "log-append.csv"))
    cfg = get_config(logdir)
    fine_tune_moment = None
    if cfg["model_file"] is not None:
        model_dir = re.search(f".*(logs.*)model_\d*.pth", cfg["model_file"]).group(1)
        if os.path.exists(model_dir):
            df0 = pd.read_csv(os.path.join(model_dir, "log-append.csv"))
            fine_tune_moment = df0.timesteps.max()
            dfl.timesteps += fine_tune_moment
            # join df0 and dfl:
            dfl = pd.concat([df0, dfl], ignore_index=True)

    # create graph
    roll_window = 5
    dfl2 = pd.DataFrame(
        {"Neural Train": dfl["mean_episode_rewards"].rolling(window=roll_window).mean(),
         "Neural Test": dfl["val_mean_episode_rewards"].rolling(window=roll_window).mean()
         })  # ,
    dfl2.index = dfl["timesteps"]

    ax = dfl2.plot()
    ns_train = df.NeuroSymb_score_Train[0]
    ns_test = df.NeuroSymb_score_Test[0]
    rn_train = df.Random_score_Train[0]
    rn_test = df.Random_score_Test[0]

    hline_dict = {"NeuroSymb Train": [ns_train, "blue", "dashed"],
                  "NeuroSymb Test": [ns_test, "orange", "dashed"],
                  "Random Train": [rn_train, "blue", "dotted"],
                  "Random Test": [rn_test, "orange", "dotted"], }
    for key in hline_dict.keys():
        plt_hline(dfl2, key, hline_dict[key][0], hline_dict[key][1], hline_dict[key][2])

    min_y = min(0, dfl["mean_episode_rewards"].min(), dfl["val_mean_episode_rewards"].min())
    max_y = max(0, dfl["mean_episode_rewards"].max(), dfl["val_mean_episode_rewards"].max())
    plt.ylim(ymin=min_y)
    plt.title("Rolling Average Reward")

    if fine_tune_moment is not None:
        plt.vlines(x=fine_tune_moment, ymin=min_y, ymax=max_y, linestyles='--')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                     box.width, box.height * 0.85])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3)

    plot_file = os.path.join(symbdir, "training_plot.png")
    plt.savefig(plot_file)

    params = dict_to_html_table(args.__dict__)

    eqn_str = get_best_str(model, split="<br/>")
    # send email
    eqn_str = floats_to_dp(eqn_str)

    nn_test = df.Neural_score_Test[0]
    nn_train = df.Neural_score_Train[0]

    tab_code = pd.DataFrame({"Train": [nn_train, ns_train, rn_train],
                             "Test": [nn_test, ns_test, rn_test]},
                            index=["Neural", "NeuroSymbolic", "Random"], ).round(2).to_html()

    test_improved = (ns_test > nn_test) and (ns_test > rn_test)
    train_improved = ns_train > nn_train and (ns_train > rn_train)

    statement = "Failed"
    if test_improved and train_improved:
        statement = "Complete Success"
    if not test_improved and train_improved:
        statement = "Improved Training Reward"
    if test_improved and not train_improved:
        statement = "Improved Generalization"

    lims = [
        min(dfs.logit.min(), dfs.logit_estimate.min()),
        max(dfs.logit.max(), dfs.logit_estimate.max())
    ]
    dfs.plot.scatter(x="logit", y="logit_estimate", c="value")
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    scatter_file = os.path.join(symbdir, "training_scatter.png")
    plt.savefig(scatter_file)

    body_text = f"<br><b>{statement}</b><br>{tab_code}<br><b>Learned Formula:</b><br><p>{eqn_str}</p><br>{params}"
    send_images_first_last([plot_file, scatter_file], "PySR Results", body_text=body_text)
    print("")


def split_df_by_index_and_pivot(df):
    dfv = df.filter(regex="_score_").T
    dfv2 = pd.Series(dfv.index).str.split("_score_", expand=True)
    dfv2.columns = ["Model", "Environment"]
    dfv2["mean_reward"] = dfv[0].values.round(decimals=2)
    dfw = dfv2.pivot(columns="Environment", index="Model", values="mean_reward")
    tab_code = dfw.to_html(index=True)


def plt_hline(dfl2, label, ns_train, colour, style):
    plt.hlines(y=ns_train,
               xmin=0,
               xmax=dfl2.index.max(),
               label=label,
               linestyles=style,
               color=colour)


def temp_func():
    iterations = 10
    data_size = 1000
    rounds = 300
    n_envs = 32
    # Coinrun:
    # logdir = "logs/train/coinrun/coinrun/2024-02-20__18-02-16__seed_6033"

    # Very Sparse Coinrun:
    # logdir = "logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"

    # # cartpole that didn't work:
    # logdir = "logs/train/cartpole/cartpole/2024-03-28__11-27-12__seed_6033"
    #
    # # 10bn timestep cartpole (works!):
    logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    run_double_graph_neurosymbolic_search(data_size, iterations, logdir, n_envs, rounds)


def get_entropy(Y):
    # only legit for single action (cartpole)
    if len(Y.shape) == 1:
        p = sigmoid(Y)
        q = 1 - p
        return -p * np.log(p) - q * np.log(q)
    ents = -(np.exp(Y) * Y).sum(-1)
    return ents


def one_hot(targets, nb_classes):
    return np.eye(nb_classes)[targets]


# def compare_outputs(sympol, nn_pol, obs):
#     obs = torch.FloatTensor(obs).to(device=sympol.device)
#     sympol.transition_model.message_model


def run_double_graph_neurosymbolic_search(args):
    data_size = args.data_size
    logdir = args.logdir
    n_envs = args.n_envs
    hp_override = {
        "device": args.device,
        "seed": args.seed,
        "val_epochs": args.val_epochs,
        "dyn_epochs": args.dyn_epochs,
        "learning_rate": args.learning_rate,
        "t_learning_rate": args.t_learning_rate,
    }
    if n_envs < 2:
        raise Exception("n_envs must be at least 2")
    symbdir, save_file = create_symb_dir_if_exists(logdir)
    print(f"symbdir: '{symbdir}'")
    cfg = vars(args)
    np.save(os.path.join(symbdir, "config.npy"), cfg)

    wandb_name = args.wandb_name
    if args.wandb_name is None:
        wandb_name = f"graph-{np.random.randint(1e5)}"

    if args.use_wandb:
        wandb_login()
        name = wandb_name
        wb_resume = "allow"  # if args.model_file is None else "must"
        project = "Graph Symb Reg"
        cfg["symbdir"] = symbdir
        if args.wandb_group is not None:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name, group=args.wandb_group)
        else:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name)

    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs)
    nn_agent = symbolic_agent_constructor(policy)
    m_in, m_out, u_in, u_out, vm_in, vm_out, vu_in, vu_out = generate_data(nn_agent, env, int(data_size))

    print("data generated")
    weights = None
    msgdir, _ = create_symb_dir_if_exists(symbdir, "msg")
    updir, _ = create_symb_dir_if_exists(symbdir, "upd")
    vmdir, _ = create_symb_dir_if_exists(symbdir, "vm")
    vudir, _ = create_symb_dir_if_exists(symbdir, "vu")

    print("\nTransition Messenger:")
    msg_model, _ = find_model(m_in, m_out, msgdir, save_file, weights, args)
    print("\nTransition Updater:")
    up_model, _ = find_model(u_in, u_out, updir, save_file, weights, args)

    print("\nValue Messsenger:")
    v_msg_model, _ = find_model(vm_in, vm_out, vmdir, save_file, weights, args)
    print("\nValue Updater:")
    v_up_model, _ = find_model(vu_in, vu_out, vudir, save_file, weights, args)

    msg_torch = NBatchPySRTorch(msg_model.pytorch())
    up_torch = NBatchPySRTorch(up_model.pytorch())

    v_msg_torch = NBatchPySRTorch(v_msg_model.pytorch())
    v_up_torch = NBatchPySRTorch(v_up_model.pytorch())

    ns_agent = symbolic_agent_constructor(copy.deepcopy(policy), msg_torch, up_torch, v_msg_torch, v_up_torch)

    print(f"Neural Parameters: {n_params(nn_agent.policy)}")
    print(f"Symbol Parameters: {n_params(ns_agent.policy)}")

    _, env, _, test_env = load_nn_policy(logdir, 100)

    fine_tuned_policy = fine_tune(ns_agent.policy, logdir, symbdir, hp_override)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()

    print(f"\nBinary Ops:\n{' '.join(args.binary_operators)}\n")

    args.logdir = "logs/train/cartpole/2024-07-11__04-48-25__seed_6033"

    if args.logdir is None:
        raise Exception("No oracle provided. Please provide logdir argument")
        print(f"No oracle provided.\nUsing Logdir: {args.logdir}")
    run_double_graph_neurosymbolic_search(args)



