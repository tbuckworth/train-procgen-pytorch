import argparse
import copy
import inspect
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import sympy
import pickle as pkl

# from torch import cuda

import wandb
# from agents.ppo_model import PPOModel
from common.logger import Logger
from common.model import NBatchPySRTorch, NBatchPySRTorchMult
from common.storage import BasicStorage
# from discrete_env.mountain_car_pre_vec import create_mountain_car

from email_results import send_images_first_last
from symbolic_regression import load_nn_policy
from symbreg.agents import SymbolicAgent, NeuralAgent, RandomAgent, NeuroSymbolicAgent
from symbreg.extra_mappings import get_extra_torch_mappings
from windows_dll_setup import windows_dll_setup_for_pysr

windows_dll_setup_for_pysr()
from pysr import PySRRegressor
from pysr.utils import _csv_filename_to_pkl_filename
# Important! keep torch after pysr
import torch
from common.env.procgen_wrappers import create_env, create_procgen_env
from helper_local import get_config, get_path, balanced_reward, load_storage_and_policy, \
    load_hparams_for_model, floats_to_dp, dict_to_html_table, wandb_login, add_symbreg_args, DictToArgs, \
    inverse_sigmoid, sigmoid, sample_from_sigmoid, map_actions_to_values, get_actions_from_all, \
    entropy_from_binary_prob, get_saved_hyperparams, softmax, sample_numpy_probs, n_params, get_logdir_from_symbdir, \
    get_latest_file_matching, get_agent_constructor, get_model_with_largest_checkpoint, initialize_storage
from common.env.env_constructor import get_env_constructor
# from cartpole.create_cartpole import create_cartpole
# from boxworld.create_box_world import create_bw_env
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
        equation_file=get_path(symbdir, save_file),
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
    M_in, M_out, U_in, U_out, Sa, Dones, Rew, V = agent.sample(Obs)
    act = agent.forward(Obs)
    act[::2] = np.random.randint(0, env.action_space.n, len(act))[::2]
    while len(Obs) < n:
        observation, rew, done, info = env.step(act)
        m_in, m_out, u_in, u_out, sa, dones, rew, v = agent.sample(observation)
        act = agent.forward(observation)
        act[::2] = np.random.randint(0, env.action_space.n, len(act))[::2]

        M_in = np.append(M_in, m_in, axis=0)
        M_out = np.append(M_out, m_out, axis=0)
        U_in = np.append(U_in, u_in, axis=0)
        U_out = np.append(U_out, u_out, axis=0)
        Sa = np.append(Sa, sa, axis=0)
        Dones = np.append(Dones, dones, axis=0)
        Rew = np.append(Rew, rew, axis=0)
        V = np.append(V, v, axis=0)
        Obs = np.append(Obs, observation, axis=0)
    return M_in, M_out, U_in, U_out, Sa, Dones, Rew, V, Obs


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


def test_agent_mean_reward(agent, env, print_name, n=40, return_values=False, seed=0):
    print(print_name)
    episodes = 0
    obs = env.reset(seed=seed)
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


def create_symb_dir_if_exists(logdir, dir_name="symbreg"):
    save_file = "symb_reg.csv"
    symbdir = os.path.join(logdir, dir_name)
    if not os.path.exists(symbdir):
        os.mkdir(symbdir)
    symbdir = os.path.join(symbdir, time.strftime("%Y-%m-%d__%H-%M-%S"))
    if not os.path.exists(symbdir):
        os.mkdir(symbdir)
    return symbdir, save_file


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
    run_graph_neurosymbolic_search(data_size, iterations, logdir, n_envs, rounds)


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


def load_learning_objects(logdir, ftdir, device):
    hyperparameters = get_saved_hyperparams(logdir)
    cfg = get_config(logdir)
    args = DictToArgs(cfg)
    create_venv = get_env_constructor(args.env_name)

    env = create_venv(args, hyperparameters)
    env_valid = create_venv(args, hyperparameters, is_valid=True) if args.use_valid_env else None

    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{args.seed}'
    newdir = os.path.join(ftdir, run_name)
    if not os.path.exists(newdir):
        os.mkdir(newdir)

    logger = Logger(args.n_envs, newdir, use_wandb=args.use_wandb, transition_model=args.algo == "ppo-model",
                    double_graph=args.algo == "double-graph-agent", ppo_pure=args.algo=="ppo-pure")
    logger.max_steps = hyperparameters.get("max_steps", 10 ** 3)

    observation_shape = env.observation_space.shape

    algo = hyperparameters.get('algo', 'ppo')
    model_based = algo in ['ppo-model', 'graph-agent']
    double_graph = algo in ['double-graph-agent']
    ppo_pure = algo in ['ppo-pure']
    hidden_state_dim = 1

    storage, storage_valid, _ = initialize_storage(args, device, double_graph, hidden_state_dim, model_based,
                                                args.n_envs, args.n_steps, observation_shape)

    agent_cons = get_agent_constructor(args.algo)

    return env, env_valid, logger, storage, storage_valid, hyperparameters, args, agent_cons


def fine_tune(policy, logdir, symbdir, hp_override, cont=False):
    ftdir = os.path.join(symbdir, "fine_tune")
    if not os.path.exists(ftdir):
        os.mkdir(ftdir)
    if cont:
        model_dir = get_pysr_dir(symbdir, "fine_tune")

    env, env_valid, logger, storage, storage_valid, hyperparameters, args, AGENT = load_learning_objects(logdir, ftdir,
                                                                                                         policy.device)
    hyperparameters.update(hp_override)
    del hyperparameters["device"]
    agent = AGENT(env, policy, logger, storage, policy.device,
                  hyperparameters.get("num_checkpoints", args.num_checkpoints),
                  env_valid=env_valid,
                  storage_valid=storage_valid,
                  **hyperparameters)

    if cont:
        model_file = get_model_with_largest_checkpoint(model_dir)
        checkpoint = torch.load(model_file, map_location=policy.device)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.v_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    agent.train(args.num_timesteps)
    wandb.finish()


def run_graph_neurosymbolic_search(args):
    fixed_val = "value" in args.fixed_nn
    data_size = args.data_size
    logdir = args.logdir
    n_envs = args.n_envs
    hp_override = {
        "device": args.device,
        "seed": args.seed,
        "val_epochs": args.val_epochs,
        "dyn_epochs": args.dyn_epochs,
        "dr_epochs": args.dr_epochs,
        "learning_rate": args.learning_rate,
        "t_learning_rate": args.t_learning_rate,
        "dr_learning_rate": args.dr_learning_rate,
        "rew_coef": args.rew_coef,
        "done_coef": args.done_coef,
    }
    if fixed_val:
        hp_override["val_epochs"] = 0
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
    m_in, m_out, u_in, u_out, sa, dones, rew, v, obs = generate_data(nn_agent, env, int(data_size))

    print("data generated")
    if os.name != "nt":
        weights = None
        msgdir, _ = create_symb_dir_if_exists(symbdir, "msg")
        updir, _ = create_symb_dir_if_exists(symbdir, "upd")
        vdir, _ = create_symb_dir_if_exists(symbdir, "v")
        rdir, _ = create_symb_dir_if_exists(symbdir, "r")
        ddir, _ = create_symb_dir_if_exists(symbdir, "done")

        print("\nTransition Messenger:")
        msg_model, elapsed_m = find_model(m_in, m_out, msgdir, save_file, weights, args)
        print("\nTransition Updater:")
        up_model, elapsed_u = find_model(u_in, u_out, updir, save_file, weights, args)

        if not fixed_val:
            print("\nValue Model:")
            v_model, elapsed_v = find_model(obs, v, vdir, save_file, weights, args)
        print("\nReward Model:")
        r_model, elapsed_r = find_model(sa, rew, rdir, save_file, weights, args)
        print("\nDone Model:")
        args.loss_function = "lbce"
        done_model, elapsed_dones = find_model(sa, dones, ddir, save_file, weights, args)

        # mi = torch.FloatTensor(m_in).to(device=policy.device)
        # ui = torch.FloatTensor(u_in).to(device=policy.device)
        # oi = torch.FloatTensor(obs).to(device=policy.device)
        # sai = torch.FloatTensor(sa).to(device=policy.device)

        msg_torch = NBatchPySRTorch(msg_model.pytorch())
        up_torch = NBatchPySRTorch(up_model.pytorch())
        if not fixed_val:
            v_torch = NBatchPySRTorch(v_model.pytorch())
        else:
            v_torch = None
        r_torch = NBatchPySRTorch(r_model.pytorch())
        done_torch = NBatchPySRTorch(done_model.pytorch())

        ns_agent = symbolic_agent_constructor(copy.deepcopy(policy), msg_torch, up_torch, v_torch, r_torch, done_torch)

        rn_agent = RandomAgent(env.action_space.n)

        ###################################

        # msg_torch(mi).shape == policy.transition_model.messenger(mi).shape
        # up_torch(ui).shape == policy.transition_model.updater(ui).shape
        # v_torch(oi).shape == policy.value(oi).shape
        # r_torch(sai).shape == policy.dr(sai)[0].shape
        # done_torch(sai).shape == policy.dr(sai)[1].shape
        # # compare_outputs(ns_agent.policy, policy, obs)
        ###################################

        print(f"Neural Parameters: {n_params(nn_agent.policy)}")
        print(f"Symbol Parameters: {n_params(ns_agent.policy)}")

        _, env, _, test_env = load_nn_policy(logdir, 100)

        fine_tuned_policy = fine_tune(ns_agent.policy, logdir, symbdir, hp_override)
        return
        ns_score_train = test_agent_mean_reward(ns_agent, env, "NeuroSymb Train", rounds, seed)
        nn_score_train = test_agent_mean_reward(nn_agent, env, "Neural    Train", rounds, seed)
        rn_score_train = test_agent_mean_reward(rn_agent, env, "Random    Train", rounds, seed)

        ns_score_test = test_agent_mean_reward(ns_agent, test_env, "NeuroSymb  Test", rounds, seed)
        nn_score_test = test_agent_mean_reward(nn_agent, test_env, "Neural     Test", rounds, seed)
        rn_score_test = test_agent_mean_reward(rn_agent, test_env, "Random     Test", rounds, seed)
        return

        best = msg_model.get_best()
        if type(best) != list:
            best = [best]
        best_loss = np.mean([x.loss for x in best])
        best_complexity = np.mean([x.complexity for x in best])
        problem_name = re.search("logs/train/([^/]*)/", logdir).group(1)

        df_values = {
            "Random_score_Train": [rn_score_train],
            "Neural_score_Train": [nn_score_train],
            "NeuroSymb_score_Train": [ns_score_train],
            "Neural_score_Test": [nn_score_test],
            "Random_score_Test": [rn_score_test],
            "NeuroSymb_score_Test": [ns_score_test],
            "Elapsed_Seconds": [elapsed],
            "Mean_Best_Loss": [best_loss],
            "Mean_Complexity_of_Best": [best_complexity],
            "Problem_name": [problem_name]
        }

        Y_hat = pysr_model.predict(X)
        # TODO: Use agent methods to clean this all up:
        if ns_agent.single_output:
            if args.stochastic:
                p = sigmoid(Y)
                Y_act = sample_from_sigmoid(p)
                p_hat = sigmoid(Y_hat)
                Y_hat_act = sample_from_sigmoid(p)
                ent = entropy_from_binary_prob(p)
                ent_hat = entropy_from_binary_prob(p_hat)
            else:
                Y_act = Y
                p = Y
                p_hat = Y_hat
                Y_hat_act = Y_hat
                ent = np.zeros_like(p)
                ent_hat = np.zeros_like(p_hat)
        else:
            if not args.stochastic:
                p = softmax(save_Y)
                ent = -(p * np.log(p)).sum(1)
                Y_act = sample_numpy_probs(p)
                p = p[np.arange(len(p)), Y_act]
                Y_hat_act = ns_agent.pred_to_action(Y_hat)
                p_hat = np.ones_like(Y)  # one_hot(Y_hat_act, save_Y.shape[-1])
                ent_hat = np.zeros_like(Y_hat)
            else:
                p = softmax(Y)
                p_hat = softmax(Y_hat)
                Y_hat_act = sample_numpy_probs(p_hat)
                ent_hat = -(p_hat * np.log(p_hat)).sum(1)
                Y_act = sample_numpy_probs(p)
                ent = -(p * np.log(p)).sum(1)

        df_values["Entropy_Pred"] = [ent_hat.mean()]
        df_values["Entropy"] = [ent.mean()]

        shp = Y.shape
        if len(shp) == 1:
            shp = (shp[0], 1)

        if not args.stochastic:
            action_vector = actions[Y_act]
        elif len(actions) == 2:
            action_vector = np.repeat(actions[-1], shp[0])
        else:
            action_vector = np.repeat(actions, shp[0])

        all_metrics = np.vstack(
            (action_vector,
             Y.reshape(np.prod(shp)),
             np.tile(V, shp[-1]),
             Y_hat.reshape(np.prod(shp)),
             p.reshape(np.prod(shp)),
             p_hat.reshape(np.prod(shp)),
             np.tile(Y_act, shp[-1]),
             np.tile(Y_hat_act, shp[-1]),
             np.tile(ent, shp[-1]),
             np.tile(ent_hat, shp[-1])
             )
        ).T
        # all_metrics = np.vstack((Y, V, Y_hat, p, p_hat, Y_act, Y_hat_act)).T
        columns = ["action", "logit", "value", "logit_estimate", "prob", "prob_estimate",
                   "sampled_action", "sampled_action_estimate", "entropy", "entropy_estimate"]
        if problem_name == "cartpole":
            all_metrics = np.hstack((X, all_metrics))
            state_features = [
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
                "gravity",
                "pole_length",
                "cart_mass",
                "pole_mass",
                "force_magnitude",
            ]
            columns = state_features + columns
        dfs = pd.DataFrame(all_metrics, columns=columns)
        dfs.loc[:, dfs.columns != "action"] = dfs.loc[:, dfs.columns != "action"].astype(float)

        if args.use_wandb:
            wandb.log({k: df_values[k][0] for k in df_values.keys()})
            # wandb_table = wandb.Table(
            #     # make this work for multiple equations:
            #     dataframe=pysr_model.equations_[["equation", "score", "loss", "complexity"]]
            # )
            wandb_metrics = wandb.Table(dataframe=dfs)
            wandb.log(
                # {#f"equations": wandb_table,
                {f"metrics": wandb_metrics},
            )

        df = pd.DataFrame(df_values)
        df.to_csv(os.path.join(symbdir, "results.csv"), mode="w", header=True, index=False)
        send_full_report(df, logdir, symbdir, pysr_model, args, dfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()

    print(f"\nBinary Ops:\n{' '.join(args.binary_operators)}\n")

    if args.logdir is None:
        raise Exception("No oracle provided. Please provide logdir argument")
        print(f"No oracle provided.\nUsing Logdir: {args.logdir}")
    run_graph_neurosymbolic_search(args)


def load_double_graph_agent(symbdir, load_weights=False):
    # symbdir = "logs/train/cartpole/2024-07-11__04-48-25__seed_6033/symbreg/2024-07-22__04-36-52"
    logdir = get_logdir_from_symbdir(symbdir)
    policy, _, symbolic_agent_constructor, _ = load_nn_policy(logdir)

    msgdir = get_pysr_dir(symbdir, "msg")
    updir = get_pysr_dir(symbdir, "upd")
    vmdir = get_pysr_dir(symbdir, "vm")
    vudir = get_pysr_dir(symbdir, "vu")

    msg_torch = load_pysr_to_torch(msgdir)
    up_torch = load_pysr_to_torch(updir)
    vm_torch = load_pysr_to_torch(vmdir)
    vu_torch = load_pysr_to_torch(vudir)

    ns_agent = symbolic_agent_constructor(policy, msg_torch, up_torch, vm_torch, vu_torch)

    if load_weights:
        model_dir = get_pysr_dir(symbdir, "fine_tune")
        model_file = get_model_with_largest_checkpoint(model_dir)
        checkpoint = torch.load(model_file, map_location=policy.device)
        ns_agent.policy.load_state_dict(checkpoint["model_state_dict"])
        ns_agent.v_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return logdir, ns_agent


def load_sr_graph_agent(symbdir):
    logdir = get_logdir_from_symbdir(symbdir)
    policy, _, symbolic_agent_constructor, _ = load_nn_policy(logdir)

    msgdir = get_pysr_dir(symbdir, "msg")
    updir = get_pysr_dir(symbdir, "upd")
    vdir = get_pysr_dir(symbdir, "v")
    rdir = get_pysr_dir(symbdir, "r")
    ddir = get_pysr_dir(symbdir, "done")

    msg_torch = load_pysr_to_torch(msgdir)
    up_torch = load_pysr_to_torch(updir)
    v_torch = load_pysr_to_torch(vdir)
    r_torch = load_pysr_to_torch(rdir)
    done_torch = load_pysr_to_torch(ddir)

    ns_agent = symbolic_agent_constructor(policy, msg_torch, up_torch, v_torch, r_torch, done_torch)
    return logdir, ns_agent


def get_pysr_dir(symbdir, sub_folder):
    return get_latest_file_matching(r"\d*-\d", 1, folder=os.path.join(symbdir, sub_folder))


def load_pysr_to_torch(msgdir):
    try:
        pickle_filename = os.path.join(msgdir, "symb_reg.pkl")
        # msg_model = PySRRegressor.from_file(pickle_filename, extra_torch_mappings=get_extra_torch_mappings())
        msg_model = pysr_from_file(pickle_filename, extra_torch_mappings=get_extra_torch_mappings())
        msg_torch = NBatchPySRTorch(msg_model.pytorch())
        return msg_torch
    except Exception:
        return None

def load_all_pysr(msgdir, device):
    pickle_filename = os.path.join(msgdir, "symb_reg.pkl")
    msg_model = pysr_from_file(pickle_filename, extra_torch_mappings=get_extra_torch_mappings(),
                               extra_sympy_mappings={"greater": lambda x, y: sympy.Piecewise((1.0, x > y), (0.0, True)),
                                                     }
                               )
    msg_torch = all_pysr_pytorch(msg_model, device)
    return msg_torch


def all_pysr_pytorch(msg_model, device):
    idx = [i for i in range(len(msg_model.equations_))]
    in_model = msg_model.pytorch(idx)
    equations = msg_model.equations.equation[idx]
    # TODO: sort this out:
    if isinstance(in_model, list):
        in_model = msg_model.equations[0].torch_format
        #TODO: not sure about this:
        equations = [msg_model.equations[0].equation]
        # models = [NBatchPySRTorchMult(m.torch_format.tolist(), cat_dim=0, device=device) for m in msg_model.equations]
        # return NBatchPySRTorchMult(models, cat_dim=-1, device=device)
    return NBatchPySRTorchMult(in_model.tolist(), equations, cat_dim=0, device=device)


def pysr_from_file(
        equation_file,
        *,
        binary_operators=None,
        unary_operators=None,
        n_features_in=None,
        feature_names_in=None,
        selection_mask=None,
        nout=1,
        **pysr_kwargs,
):
    """
    Create a model from a saved model checkpoint or equation file.

    Parameters
    ----------
    equation_file : str
        Path to a pickle file containing a saved model, or a csv file
        containing equations.
    binary_operators : list[str]
        The same binary operators used when creating the model.
        Not needed if loading from a pickle file.
    unary_operators : list[str]
        The same unary operators used when creating the model.
        Not needed if loading from a pickle file.
    n_features_in : int
        Number of features passed to the model.
        Not needed if loading from a pickle file.
    feature_names_in : list[str]
        Names of the features passed to the model.
        Not needed if loading from a pickle file.
    selection_mask : list[bool]
        If using select_k_features, you must pass `model.selection_mask_` here.
        Not needed if loading from a pickle file.
    nout : int
        Number of outputs of the model.
        Not needed if loading from a pickle file.
        Default is `1`.
    **pysr_kwargs : dict
        Any other keyword arguments to initialize the PySRRegressor object.
        These will overwrite those stored in the pickle file.
        Not needed if loading from a pickle file.

    Returns
    -------
    model : PySRRegressor
        The model with fitted equations.
    """

    pkl_filename = _csv_filename_to_pkl_filename(equation_file)

    # Try to load model from <equation_file>.pkl
    print(f"Checking if {pkl_filename} exists...")
    if os.path.exists(pkl_filename):
        print(f"Loading model from {pkl_filename}")
        assert binary_operators is None
        assert unary_operators is None
        assert n_features_in is None
        with open(pkl_filename, "rb") as f:
            model = pkl.load(f)
        # Change equation_file_ to be in the same dir as the pickle file
        base_dir = os.path.dirname(pkl_filename)
        base_equation_file = os.path.basename(model.equation_file_)
        model.equation_file_ = os.path.join(base_dir, base_equation_file)

        # Get constructor parameters and default values
        params = inspect.signature(model.__init__).parameters

        # Filter for missing parameters excluding kwargs
        missing_params = {k: v for k, v in params.items() if
                          k not in model.__dict__.keys() and v.name != "self" and v.kind != v.VAR_KEYWORD}

        if len(missing_params) > 0:
            warnings.warn(
                "The following missing parameters will be assigned with default values:"
                f"{', '.join(missing_params.keys())}"
                "This may be due to the model being saved under an old package version."
            )

        # Assign missing attributes
        for k, v in missing_params.items():
            setattr(model, k, v)

        # Update any parameters if necessary, such as
        # extra_sympy_mappings:
        model.set_params(**pysr_kwargs)
        if "equations_" not in model.__dict__ or model.equations_ is None:
            model.refresh()

        return model
