from gym3 import vectorize_gym, ViewerWrapper, ToBaselinesVecEnv
# from stable_baselines3.common.vec_env import VecExtractDictObs

from boxworld.box_world_env import BoxWorld
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

def make_boxworld(n, goal_length, num_distractor, distractor_length, max_steps=10**6, collect_key=True, world=None):
    return BoxWorld(n, goal_length, num_distractor, distractor_length, max_steps, collect_key, world)



def create_box_world_env(env_args, render, normalize_rew=True):
    if render:
        env_args["render_mode"] = "rgb_array"
    n_envs = env_args["n_envs"]
    del env_args["n_envs"]
    seed = env_args["seed"]
    del env_args["seed"]
    #use_subproc?
    venv = vectorizeBoxWorld(env_args, n_envs=n_envs, seed=seed)
    if render:
        # could create a mirrorFrame wrapper that goes on gym3 envs, and put before the viewer...
        venv = ViewerWrapper(venv, info_key="rgb")
    venv = ToBaselinesVecEnv(venv)
    # venv = VecExtractDictObs(venv, "rgb")
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
        # the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv