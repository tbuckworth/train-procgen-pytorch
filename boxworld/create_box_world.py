import numpy as np
from gym3 import vectorize_gym, ViewerWrapper, ToBaselinesVecEnv
from gym3.interop import _space2vt

# from stable_baselines3.common.vec_env import VecExtractDictObs

from boxworld.box_world_env import BoxWorld
from boxworld.box_world_env_vec import BoxWorldVec
from common.env.procgen_wrappers import VecExtractDictObs, VecNormalize, TransposeFrame, ScaledFloatFrame


def vectorizeBoxWorld(env_args, n_envs, seed, use_subproc=True):
    # boxWorldId = 'BoxWorld-v0'  # It is best practice to have a space name and version number.

    # gym.envs.registration.register(
    #     id=boxWorldId,
    #     entry_point="box_world_env:BoxWorld",
    #     max_episode_steps=env_args["max_steps"],  # Customize to your needs.
    #     reward_threshold=10  # Customize to your needs.
    # )
    # env_args["id"] = boxWorldId
    env = vectorize_gym(num=n_envs, env_kwargs=env_args, seed=seed, env_fn=make_boxworld, use_subproc=use_subproc)
    return env


def make_boxworld(n, goal_length, num_distractor, distractor_length, max_steps=10 ** 6, collect_key=True, world=None):
    return BoxWorld(n, goal_length, num_distractor, distractor_length, max_steps, collect_key, world)


def create_box_world_env(env_args_in, render, normalize_rew=True):
    # if render:
    #     env_args["render_mode"] = "rgb_array"
    env_args = env_args_in.copy()
    n_envs = env_args["n_envs"]
    del env_args["n_envs"]
    seed = env_args["seed"]
    del env_args["seed"]
    # use_subproc?
    venv = vectorizeBoxWorld(env_args, n_envs=n_envs, seed=seed)
    if render:
        venv = ViewerWrapper(venv, info_key="rgb")
    venv = ToBaselinesVecEnv(venv)
    # venv = VecExtractDictObs(venv, "rgb")
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
        # the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


def create_box_world_env_pre_vec(env_args_in, render, normalize_rew=True):
    # if render:
    #     env_args["render_mode"] = "rgb_array"
    env_args = env_args_in.copy()
    n_envs = env_args["n_envs"]
    # use_subproc?
    venv = make_boxworld_vec(env_args, n_envs)
    if render:
        venv = ViewerWrapper(venv, info_key="rgb")
    venv = ToBaselinesVecEnv(venv)
    # venv = VecExtractDictObs(venv, "rgb")
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
        # the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


def make_boxworld_vec(env_args, n_envs):
    venv = BoxWorldVec(n_envs, env_args["n"], env_args["goal_length"], env_args["num_distractor"],
                       env_args["distractor_length"], max_steps=env_args["max_steps"], start_seed=env_args["seed"],
                       n_levels=env_args["n_levels"])
    venv.ob_space = _space2vt(venv.observation_space)
    venv.ac_space = _space2vt(venv.action_space)

    return venv


if __name__ == "__main__":
    hyperparameters = {}
    env_args = {"n_envs": 512,
                "n": hyperparameters.get('grid_size', 12),
                "goal_length": hyperparameters.get('goal_length', 5),
                "num_distractor": hyperparameters.get('num_distractor', 0),
                "distractor_length": hyperparameters.get('distractor_length', 0),
                "max_steps": 10 ** 2,
                "seed": 0,
                "n_levels": 0
                }
    normalize_rew = hyperparameters.get('normalize_rew', True)
    env = create_box_world_env_pre_vec(env_args, render=True, normalize_rew=normalize_rew)
    num_actions = env.action_space.n

    while True:
        actions = np.random.randint(0, num_actions, size=env_args["n_envs"])
        # actions = np.array([np.random.choice(num_actions) for _ in range(env_args["n_envs"])])
        next_obs, rew, done, info = env.step(actions)
        if np.any(done):
            print("done")


def create_bw_env(args, hyperparameters, is_valid=False):
    n_envs = hyperparameters.get('n_envs', 32)
    max_steps = hyperparameters.get("max_steps", 10 ** 3)
    env_args = {"n_envs": n_envs,
                "n": hyperparameters.get('grid_size', 12),
                "goal_length": hyperparameters.get('goal_length', 5),
                "num_distractor": hyperparameters.get('num_distractor', 0),
                "distractor_length": hyperparameters.get('distractor_length', 0),
                "max_steps": max_steps,
                "n_levels": args.num_levels,
                "seed": args.seed,
                }
    normalize_rew = hyperparameters.get('normalize_rew', True)
    if is_valid:
        env_args["n"] = hyperparameters.get('grid_size_v', 12)
        env_args["goal_length"] = hyperparameters.get('goal_length_v', 5)
        env_args["num_distractor"] = hyperparameters.get('num_distractor_v', 0)
        env_args["distractor_length"] = hyperparameters.get('distractor_length_v', 0)
        env_args["seed"] = args.seed + np.random.randint(1e6, 1e7) if env_args["n_levels"] == 0 else env_args[
                                                                                                         "n_levels"] + 1
        env_args["n_levels"] = 0
    return create_box_world_env_pre_vec(env_args, render=False, normalize_rew=normalize_rew)
