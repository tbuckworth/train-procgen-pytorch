import inspect
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf

import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import wandb
from sqlalchemy.ext.asyncio import AsyncTransaction

from common.env.env_constructor import get_pets_env_constructor
from helper_local import create_logdir, wandb_login, get_project

mpl.rcParams.update({"font.size": 16})


def get_env_hyperparameters(args, env_cons):
    params = inspect.signature(env_cons(args, {}).env.env.__init__).parameters.keys()
    params = [f"{p}_v" for p in params] + list(params)
    return {k: v for k, v in args.__dict__.items() if k in params}


def run_pets(args):
    # possible dynamics model:
    # pets.pets_models.GraphTransitionPets
    # mbrl.models.GaussianMLP

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = args.seed
    # env = cartpole_env.CartPoleEnv(render_mode="rgb_array")
    env_cons = get_pets_env_constructor(args.env_name)
    env_hyperparameters = get_env_hyperparameters(args, env_cons)
    env = env_cons(args, env_hyperparameters)
    env_valid = env_cons(args, env_hyperparameters, is_valid=True)
    obs, _ = env.reset(seed)
    vobs, _ = env_valid.reset(seed)
    assert obs.shape == vobs.shape
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fns.cartpole
    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.cartpole
    if args.use_custom_reward_fn:
        reward_fn = env.rew_func
        term_fn = env.done_func

    # configuration

    trial_length = args.trial_length  # 200
    num_trials = args.num_trials  # 100
    ensemble_size = args.ensemble_size  # 5

    # Everything with "???" indicates an option with a missing value.
    # Our utility functions will fill in these details using the
    # environment information
    cfg = generate_pets_cfg_dict(args, device)

    # Setup Wandb:
    wandb_name = args.wandb_name
    if args.wandb_name is None:
        model_name = re.split(r'\.', args.dyn_model)[-1]
        wandb_name = f"{model_name}_{np.random.randint(1e5)}"
    for key, value in args.__dict__.items():
        print(key, ':', value)
    if args.detect_nan:
        torch.autograd.set_detect_anomaly(True)

    logdir = create_logdir(args, "pets", args.env_name, "")
    np.save(os.path.join(logdir, "config.npy"), vars(args))
    print(f'Logging to {logdir}')

    if args.use_wandb:
        # if upload_env_params:
        #     cfg.update(env_params)
        #     cfg.update(env_params_v)
        wb_cfg = vars(args)
        wb_cfg["logdir"] = logdir
        wandb_login()
        name = f"pets-{wandb_name}"
        wb_resume = "allow"  # if args.model_file is None else "must"
        project = get_project(args.env_name, args.exp_name)
        if args.wandb_group is not None:
            wandb.init(project=project, config=wb_cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name, group=args.wandb_group)
        else:
            wandb.init(project=project, config=wb_cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name)

    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    # Create a gym-like environment to encapsulate the model
    model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)

    replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)

    common_util.rollout_agent_trajectories(
        env,
        trial_length,  # initial exploration steps
        planning.RandomAgent(env),
        {},  # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer,
        trial_length=trial_length
    )

    print("# samples stored", replay_buffer.num_stored)

    agent = load_pets_agent(args, device, model_env)

    train_losses = []
    val_scores = []
    train_step = [0]
    trial_step = 0

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("trial/step")
    wandb.define_metric("trial/*", step_metric="trial/step")

    def train_callback_tr(trial):
        def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
            train_losses.append(tr_loss)
            val_scores.append(val_score.mean().item())  # this returns val score per ensemble model
            train_step[-1] += 1
            log_names = ["train/step", "train/trial", "train/epoch", "train/train_loss", "train/val_score"]
            log = [train_step[-1], trial, _epoch, tr_loss, val_score.mean().item()]
            if args.use_wandb:
                wandb.log({k: v for k, v in zip(log_names, log)})

        return train_callback

    # Create a trainer for the model
    model_trainer = models.ModelTrainer(dynamics_model,
                                        optim_lr=args.learning_rate,  # 1e-3,
                                        weight_decay=args.weight_decay,  # 5e-5,
                                        )

    # # Create visualization objects
    # fig, axs = plt.subplots(1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]})
    # ax_text = axs[0].text(300, 50, "")

    # Main PETS loop
    save_every = num_trials // args.num_checkpoints
    checkpoint_cnt = 0
    checkpoints = [(i + 1) * save_every for i in range(args.num_checkpoints)] + [num_trials - 2]
    checkpoints.sort()
    all_rewards = [0]
    val_rewards = [0]
    for trial in range(num_trials):
        obs, _ = env.reset(None)
        agent.reset()

        terminated = False
        total_reward = 0.0
        steps_trial = 0
        save = False
        while not terminated:
            # --------------- Model Training -----------------
            if steps_trial == 0:
                dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats

                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )

                if args.overfit:
                    dataset_val = None

                model_trainer.train(
                    dataset_train,
                    dataset_val=dataset_val,
                    num_epochs=args.num_epochs,  # 50,
                    patience=args.patience,  # 50,
                    callback=train_callback_tr(trial),
                    silent=True)
            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, terminated, truncated, _ = common_util.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer)

            obs = next_obs
            total_reward += reward
            steps_trial += 1

            if steps_trial == trial_length:
                if total_reward < max(all_rewards[:-1]):
                    save = True
                break

        # Rollout in val env:
        total_val_reward = run_agent_in_env(agent, env_valid, trial_length)

        val_rewards.append(total_val_reward)
        all_rewards.append(total_reward)

        trial_step += 1
        log_names = ["trial/step",
                     "trial/trial",
                     "trial/total_reward",
                     "trial/total_val_reward",
                     "trial/train_loss",
                     "trial/val_score",
                     "trial/cum_max_total_reward"]
        log = [trial_step, trial, total_reward, total_val_reward, train_losses[-1], val_scores[-1],
               max(all_rewards)]
        print(f"Trial:\t\t{trial}")
        print(f"Reward:\t\t{total_reward}")
        print(f"Val.Reward:\t{total_val_reward}")
        print(f"Max:\t\t{max(all_rewards)}")

        if args.use_wandb:
            wandb.log({k: v for k, v in zip(log_names, log)})
        if trial > checkpoints[checkpoint_cnt] or save:
            print("Saving model.")
            save_dir = f"{logdir}/model_{trial}"
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            dynamics_model.save(save_dir)
            checkpoint_cnt += 1
    wandb.finish()

    # fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    # ax[0].plot(train_losses)
    # ax[0].set_xlabel("Total training epochs")
    # ax[0].set_ylabel("Training loss (avg. NLL)")
    # ax[1].plot(val_scores)
    # ax[1].set_xlabel("Total training epochs")
    # ax[1].set_ylabel("Validation score (avg. MSE)")
    # ax[2].plot(all_rewards)
    # ax[2].set_xlabel("Trials")
    # ax[2].set_ylabel("Rewards")
    #
    # plt.show()
    #
    # print("nothing")

def load_pets_agent(args, device, model_env):
    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": args.planning_horizon,  # 15,
        "replan_freq": args.replan_freq,  # 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": args.num_iterations,  # 5,
            "elite_ratio": args.elite_ratio,  # 0.1,
            "population_size": args.population_size,  # 500,
            "alpha": args.alpha,  # 0.1,
            "device": device,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "clipped_normal": False
        }
    })
    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=args.num_particles,  # 20
    )
    return agent

def generate_pets_cfg_dict(args, device):
    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            # "_target_": "mbrl.models.GaussianMLP",
            "_target_": args.dyn_model,
            "device": device,
            "num_layers": args.num_layers,  # 4,
            "ensemble_size": args.ensemble_size,
            "hid_size": args.hid_size,  # 200,
            "in_size": "???",
            "out_size": "???",
            "deterministic": args.deterministic,  # False,
            "propagation_method": "fixed_model",
            # can also configure activation function for GaussianMLP
            "activation_fn_cfg": {
                "_target_": "torch.nn.LeakyReLU",
                "negative_slope": 0.01
            }
        },
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": args.trial_length,
            "num_steps": args.num_trials * args.trial_length,
            "model_batch_size": args.model_batch_size,  # 32,
            "validation_ratio": args.validation_ratio,  # 0.05
        }
    }
    try:
        cfg_dict["dynamics_model"]["residual"] = args.residual
    except AttributeError:
        pass

    cfg = omegaconf.OmegaConf.create(cfg_dict)
    return cfg

def run_agent_in_env(agent, env, trial_length):
    obs, _ = env.reset(None)
    agent.reset()

    terminated = False
    total_reward = 0.0
    steps_trial = 0
    while not terminated:
        # --- Doing env step using the agent and adding to model dataset ---
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        obs = next_obs
        total_reward += reward
        steps_trial += 1
        if steps_trial == trial_length:
            break
    return total_reward
