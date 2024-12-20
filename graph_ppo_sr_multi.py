import argparse
import copy
import os
import traceback

import numpy as np
import torch
from torch import nn

import wandb

from double_graph_sr import find_model, trial_agent_mean_reward
from graph_sr import get_pysr_dir, load_all_pysr, all_pysr_pytorch
from helper_local import add_symbreg_args, wandb_login, n_params, get_project, print_dict, \
    get_model_with_largest_checkpoint, create_symb_dir_if_exists
from symbolic_regression import load_nn_policy


def generate_data(agent, env, n):
    Obs = env.reset()
    M_in, M_out, A_in, A_out = agent.sample(Obs)
    act = agent.forward(Obs)
    randomize_nth(Obs, act, env)
    while len(M_in) < n:
        observation, rew, done, info = env.step(act)
        m_in, m_out, a_in, a_out = agent.sample(observation)
        act = agent.forward(observation)
        randomize_nth(Obs, act, env)

        M_in = np.append(M_in, m_in, axis=0)
        M_out = np.append(M_out, m_out, axis=0)
        A_in = np.append(A_in, a_in, axis=0)
        A_out = np.append(A_out, a_out, axis=0)

    return M_in, M_out, A_in, A_out


def randomize_nth(Obs, act, env, n=2):
    act[::n] = np.array([env.action_space.sample().squeeze() for _ in Obs])[::n]

def extract_target_from_dist(dist, deterministic):
    if isinstance(dist, torch.distributions.Normal):
        x = dist.loc
        if not deterministic:
            x = torch.concat([dist.loc.unsqueeze(-1), dist.scale.unsqueeze(-1)], dim=-1)
    elif isinstance(dist, torch.distributions.Categorical):
        x = dist.logits
        # # TODO: check this
        # if deterministic:
        #     x = x.argmax(dim=-1)
    else:
        raise NotImplementedError(f"dist must be one of (Normal,Categorical), not {type(dist)}")
    return x

def generate_data_supervised(agent, env, n):
    def predict(Obs):
        with torch.no_grad():
            Obs = torch.FloatTensor(Obs).to(agent.policy.device)
            dist, a_out, m_out = agent.policy.forward_fine_tune(Obs)
            act = dist.sample().detach().cpu().numpy()
            if agent.deterministic:
                act = dist.loc.detach().cpu().numpy()
        y = extract_target_from_dist(dist, agent.deterministic)
        randomize_nth(Obs, act, env)
        return y, act, Obs, a_out, m_out

    Obs = env.reset()
    Y, act, Obs, A_out, M_out = predict(Obs)
    while len(Y) < n:
        obs, rew, done, info = env.step(act)
        y, act, obs, a_out, m_out = predict(obs)
        Y = torch.cat([Y, y], axis=0)
        Obs = torch.cat([Obs, obs], axis=0)
        A_out = torch.cat([A_out, a_out], axis=0)
        M_out = torch.cat([M_out, m_out], axis=0)

    return Obs, Y, A_out, M_out


def fine_tune_supervised(student, teacher, env, test_env, args, ftdir, ensemble="messenger", a_coef=1., m_coef=1000., start=0, target_reward=np.inf):
    stop_early = False
    nc = args.num_checkpoints
    save_every = args.num_timesteps // nc
    checkpoints = [((i + 1) * save_every) + start for i in range(nc)] + [args.num_timesteps - 2 + start]
    checkpoints.sort()
    t = start
    i = 0

    with torch.no_grad():
        x, y, a_out, m_out = generate_data_supervised(teacher, env, args.batch_size)
        loss, l_loss, m_loss, a_loss, a_out_hat, m_out_hat = calc_losses(x, y, a_out, m_out, student, a_coef, m_coef,
                                                                         args)
    mean_rewards, val_mean_rewards = set_elites_trial_agent(a_out, a_out_hat, args, ensemble, env, m_out, m_out_hat,
                                                            student, test_env)
    param_count = n_params(student.policy.graph.messenger) + n_params(student.policy.graph.actor)

    if args.use_wandb:
        wandb.log({
            'timesteps': t,
            'loss': loss.item(),
            'l_loss': l_loss.item(),
            'a_loss': a_loss.item(),
            'm_loss': m_loss.item(),
            'mean_reward': mean_rewards,
            'val_mean_reward': val_mean_rewards,
            'n_params': param_count,
        })
    optimizer = torch.optim.Adam(student.policy.parameters(), lr=args.learning_rate)
    for _ in range(args.num_timesteps // args.batch_size):
        x, y, a_out, m_out = generate_data_supervised(teacher, env, args.batch_size)
        for _ in range(args.epoch):
            loss, l_loss, m_loss, a_loss, a_out_hat, m_out_hat = calc_losses(x, y, a_out, m_out, student, a_coef,
                                                                             m_coef, args)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_rewards, val_mean_rewards = set_elites_trial_agent(a_out, a_out_hat, args, ensemble, env, m_out, m_out_hat,
                                                                student, test_env)
        if mean_rewards > 0.95 * target_reward:
            stop_early = True
        t += len(x)
        log = {
            'timesteps': t,
            'loss': loss.item(),
            'l_loss': l_loss.item(),
            'a_loss': a_loss.item(),
            'm_loss': m_loss.item(),
            'mean_reward': mean_rewards,
            'val_mean_reward': val_mean_rewards,
            'n_params': param_count,
        }
        print_dict(log)
        if args.use_wandb:
            wandb.log(log)
        if t > checkpoints[i] or stop_early:
            print("Saving model.")
            torch.save({'model_state_dict': student.policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       ftdir + '/model_' + str(t) + '.pth')
            i += 1
            if i == len(checkpoints):
                break
        if stop_early:
            break
    set_elites(a_out, a_out_hat, ensemble, m_out, m_out_hat, student)
    return t


def calc_losses(x, y, a_out, m_out, ns_agent, a_coef, m_coef, a):
    dist_hat, a_out_hat, m_out_hat = ns_agent.policy.forward_fine_tune(x)
    y_hat = extract_target_from_dist(dist_hat, ns_agent.deterministic)
    offset = 0
    if a_out.shape[-1] != a_out_hat.shape[-1]:
        # This is where a_out_hat is symbolic deterministic (just mean, no var)
        # and a_out is stochastic, (still has the var)
        a_out = a_out[..., :-1]
        a_out_hat = a_out_hat.unsqueeze(-1)
        offset = -1
    if not a.min_mse:
        l_loss = nn.MSELoss()(y, y_hat)
        m_loss = nn.MSELoss()(m_out, m_out_hat)
        a_loss = nn.MSELoss()(a_out, a_out_hat)
    else:
        m2 = ((m_out - m_out_hat) ** 2).mean(dim=(-1, -2, -3))
        m_loss = m2.mean()
        a2 = ((a_out - a_out_hat[..., m2.argmin(), :, :, :]) ** 2).mean(dim=(-1, -2, -3))
        a_loss = a2.mean()

        y_hat_min = y_hat
        if y_hat.ndim <= 3 + offset:
            y_hat_min = y_hat[m2.argmin()]
        if y_hat.ndim == 4 + offset:
            y_hat_min = y_hat[a2.argmin(), m2.argmin()]

        l_loss = ((y - y_hat_min) ** 2).mean()

    # plot messenger vs truth
    # plt.scatter(m_out.reshape(-1).detach().cpu().numpy(), m_out_hat[6].reshape(-1).detach().cpu().numpy())

    loss = l_loss + a_loss * a_coef + m_loss * m_coef
    return loss, l_loss, m_loss, a_loss, a_out_hat, m_out_hat


def set_elites_trial_agent(a_out, a_out_hat, args, ensemble, env, m_out, m_out_hat, ns_agent, test_env):
    set_elites(a_out, a_out_hat, ensemble, m_out, m_out_hat, ns_agent)
    mean_rewards = trial_agent_mean_reward(ns_agent, env, "", n=args.n_tests, seed=args.seed, print_results=False,
                                           reset=False)
    val_mean_rewards = trial_agent_mean_reward(ns_agent, test_env, "", n=args.n_tests,
                                               seed=args.seed, print_results=False, reset=False)
    if ensemble in ["messenger", "both"]:
        ns_agent.policy.graph.messenger.set_elite(None)
    if ensemble in ["actor", "both"]:
        ns_agent.policy.graph.actor.set_elite(None)
    return mean_rewards, val_mean_rewards


def set_elites(a_out, a_out_hat, ensemble, m_out, m_out_hat, ns_agent):
    m_elite = None
    if ensemble in ["messenger", "both"]:
        m_elite = elite_index(m_out, m_out_hat)
        ns_agent.policy.graph.messenger.set_elite(m_elite)
    if ensemble in ["actor", "both"]:
        x = a_out_hat
        if m_elite is not None:
            x = a_out_hat[:, m_elite]
        a_elite = elite_index(a_out, x)
        ns_agent.policy.graph.actor.set_elite(a_elite)


def elite_index(m_out, m_out_hat):
    # TODO: is the dim correct for all?
    return ((m_out - m_out_hat) ** 2).mean(dim=(1, 2, 3)).argmin()


def run_graph_ppo_multi_sr(args):
    logdir = args.logdir
    n_envs = args.n_envs
    data_size = args.data_size
    if args.load_pysr:
        symbdir = args.symbdir
        save_file = "symb_reg.csv"
    else:
        symbdir, save_file = create_symb_dir_if_exists(logdir)
    cfg = vars(args)
    np.save(os.path.join(symbdir, "config.npy"), cfg)

    wandb_name = args.wandb_name
    if args.wandb_name is None:
        wandb_name = f"graph-ppo-sr{np.random.randint(1e5)}"

    if args.use_wandb:
        wandb_login()
        name = wandb_name
        wb_resume = "allow"  # if args.model_file is None else "must"
        project = get_project("cartpole", "symbreg")
        cfg["symbdir"] = symbdir
        if args.wandb_group is not None:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name, group=args.wandb_group)
        else:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name)

    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs)
    nn_agent = symbolic_agent_constructor(policy)
    if not args.stochastic:
        nn_agent.set_deterministic(True)

    m_in, m_out, a_in, a_out = generate_data(nn_agent, env, int(data_size))

    print("data generated")
    act_torch = None
    weights = None
    eq_log = {}
    if args.load_pysr:
        msgdir = get_pysr_dir(symbdir, "msg")
        actdir = get_pysr_dir(symbdir, "act")
        msg_torch = load_all_pysr(msgdir, device=policy.device)
        if not args.sequential:
            act_torch = load_all_pysr(actdir, device=policy.device)
    else:
        msgdir, _ = create_symb_dir_if_exists(symbdir, "msg")
        actdir, _ = create_symb_dir_if_exists(symbdir, "act")

        print("\nMessenger:")
        msg_model, _ = find_model(m_in, m_out, msgdir, save_file, weights, args)
        msg_torch = all_pysr_pytorch(msg_model, policy.device)
        eq_log["messenger"] = msg_model.get_best().equation
        if not args.sequential:
            print("\nActor:")
            act_model, _ = find_model(a_in, a_out, actdir, save_file, weights, args)
            act_torch = all_pysr_pytorch(act_model, policy.device)
            eq_log["actor"] = act_model.get_best().equation
        if args.use_wandb:
            wandb.log(eq_log)

    # upload msgdir to wandb

    ns_agent = symbolic_agent_constructor(copy.deepcopy(policy), msg_torch, act_torch)
    if not args.stochastic:
        ns_agent.set_deterministic(True)
        if not args.sequential:
            ns_agent.policy.set_no_var(True)
    print(f"Neural Parameters: {n_params(nn_agent.policy)}")
    param_count = n_params(ns_agent.policy.graph.messenger) + n_params(ns_agent.policy.graph.actor)
    print(f"Symbol Parameters: {param_count}")

    # supervised learning:
    _, env, _, test_env = load_nn_policy(logdir, n_envs=100)
    neural_train = trial_agent_mean_reward(nn_agent, env, "Neural Train")
    neural_test = trial_agent_mean_reward(nn_agent, test_env, "Neural Test")
    if args.use_wandb:
        wandb.log({"neural_train_score": neural_train,
                   "neural_test_score": neural_test
                   })

    ftdir = os.path.join(symbdir, "fine_tune")
    if not os.path.exists(ftdir):
        os.mkdir(ftdir)
    if not args.sequential:
        fine_tune_supervised(ns_agent, nn_agent, env, test_env, args, ftdir, ensemble="both")
    else:
        if args.load_ft:
            model_file = get_model_with_largest_checkpoint(ftdir)
            checkpoint = torch.load(model_file, map_location=ns_agent.policy.device)
            ns_agent.policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            t = fine_tune_supervised(ns_agent, nn_agent, env, test_env, args, ftdir, ensemble="messenger", target_reward=neural_train)
        # freeze messenger
        s_agent = copy.deepcopy(ns_agent)
        s_agent.policy.to(device=ns_agent.policy.device)
        for param in s_agent.policy.graph.messenger.parameters():
            param.requires_grad = False
        if args.use_wandb:
            wandb.log({"switch_timestep": t})
        find_actor = True
        error_count = 0
        while find_actor:
            find_actor = False
            try:
                _, _, a_in, a_out = generate_data(ns_agent, env, int(data_size))
                print("\nActor:")
                act_model, _ = find_model(a_in, a_out, actdir, save_file, weights, args)
                act_torch = all_pysr_pytorch(act_model, policy.device)
                eq_log["actor"] = act_model.get_best().equation
                s_agent.policy.graph.actor = act_torch.to(device=ns_agent.policy.device)
                if not args.stochastic:
                    s_agent.policy.set_no_var(True)

                # upload act_dir to wandb

                fine_tune_supervised(s_agent, ns_agent, env, test_env, args, ftdir, ensemble="actor", start=t, target_reward=neural_train)
            except Exception as e:
                print(traceback.format_exc())
                if t < args.num_timesteps:
                    # implies early stopping, which implies good performance, so worth repeating.
                    find_actor = True
                    error_count += 1
                if error_count >= 10 or t >= args.num_timesteps:
                    wandb.finish(exit_code=-1)
                    return

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()
    # args.logdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033"
    args.logdir = "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033"
    args.logdir = "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033"
    args.iterations = 32
    args.stochastic = False

    args.load_pysr = True
    # args.symbdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033/symbreg/2024-08-27__10-39-50"
    # args.symbdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033/symbreg/2024-08-27__19-55-01"
    # args.symbdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033/symbreg/2024-08-28__17-46-04"
    # args.symbdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033/symbreg/2024-09-04__10-16-46"
    # args.symbdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033/symbreg/2024-09-04__10-36-16"
    # args.symbdir = "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033/symbreg/2024-09-10__11-52-31"
    # args.symbdir = "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033/symbreg/2024-09-11__11-15-26"
    # args.symbdir = "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033/symbreg/2024-09-18__14-41-39"
    args.symbdir = "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033/symbreg/2024-09-14__22-06-42"
    args.sequential = True
    args.min_mse = True
    args.model_selection = "accuracy"
    args.maxsize = 7
    args.use_wandb = False
    args.binary_operators = ["+", "-", "*", "greater", "/"]
    args.unary_operators = ["sin", "relu", "log", "exp", "sign", "sqrt", "square"]
    args.device = "gpu" if torch.cuda.is_available() else "cpu"
    args.learning_rate = 1e-1
    args.ncycles_per_iteration = 4000
    args.n_tests = 1
    args.batch_size = 10
    args.num_checkpoints = 10
    args.num_timesteps = int(1e2)
    args.epoch = 1
    args.load_ft = False


    # replicating perfect gen:
    args.use_wandb = True
    args.batch_size = 413
    args.bumper = False
    args.data_size = 4610
    args.denoise = False
    args.device = "gpu"
    args.env_name = "cartpole"
    args.epoch = 696
    args.exp_name = "symbreg"
    args.iterations = 32
    args.learning_rate = 0.00835
    args.load_pysr = True
    args.logdir = "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033"
    args.loss_function = "mse"
    args.maxsize = 57
    args.min_mse = True
    args.model_selection = "accuracy"
    args.n_envs = 2
    args.n_tests = 40
    args.ncycles_per_iteration = 4000
    args.num_checkpoints = 2
    args.num_timesteps = 10900
    args.original_start = "Thu Sep 12 args.13:56 =58 2024"
    args.param_name = "graph"
    args.populations = 15
    args.procs = 15
    args.rounds = 10
    args.seed = 6033
    args.sequential = True
    args.stochastic = False
    args.symbdir = "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033/symbreg/2024-09-14__22-06-42"
    args.timeout_in_seconds = 36000
    args.use_wandb = True
    args.wandb_tags = ["gpp-cont1", "p-gen"]
    while True:
        try:
            run_graph_ppo_multi_sr(args)
        except Exception as e:
            print(traceback.format_exc())
            if args.use_wandb:
                wandb.finish(exit_code=-1)
