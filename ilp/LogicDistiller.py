# Deep RL Policy Subsumption and Extrapolation with Inductive Logic Programming
import copy
import os.path
import re

import gym
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from helper import coords_to_image
# from matplotlib import pyplot as plt

# from VQMHA import flatten_features
# from coinrun import save_gif
# from coinrun_ppo import input_to_state, sample_action
from ilp.ilp_helper import create_cmd, run_subprocess, append_to_csv_if_exists, write_string_to_file, \
    extract_clingo_solution
from inspect_agent import load_policy


class LogicDistiller:
    def __init__(self, policy, device, probabilistic=True, atn_threshold=0.6, action_threshold=0.6):
        self.device = device
        self.clingo_file_contents = None
        self.number_zero_actions_clingo = 0
        self.clingo_file = "logic_examples/clingo_learning.lp"
        self.top_n = int(1e9)
        self.learning_file = "logic_examples/fastLAS_learning.las"
        self.hypothesis = None
        self.e_facts = np.ndarray([], dtype=np.int32)
        self.r_facts = None
        self.coords = None
        self.f_facts = None
        self.a_facts = None
        self.probabilistic = probabilistic
        self.atn_threshold = atn_threshold
        self.action_threshold = action_threshold
        self.policy = policy
        self.example_list = []
        self.example_strings = []

    def reset_example_strings(self):
        self.a_facts = None
        self.example_strings = []

    def extract_example(self, observation):
        act, act_probs, atn, feature_indices, value = self.forward(observation)
        self.example_list.append((feature_indices, atn, act_probs, value))

    def forward(self, observation):
        obs = torch.FloatTensor(observation).to(self.device)
        # Investigate mha_layer 2 attention:
        x, atn, feature_indices = self.policy.embedder.forward_with_attn_indices(obs, 2)
        dist, value = self.policy.hidden_to_output(x)
        act = dist.sample().cpu().numpy()
        act_probs = dist.probs.detach().cpu().numpy()
        value = value.detach().cpu().numpy()
        feature_indices = feature_indices.detach().cpu().numpy()
        atn = atn.detach().cpu().numpy()
        return act, act_probs, atn, feature_indices, value

    def draw_attention(self, observation, atn):
        high_atn = atn[0] > 0.2
        arrows = high_atn.argwhere().detach().numpy()
        plt.imshow(observation.transpose((0, 2, 3, 1))[0])
        for arr, weight in zip(arrows, atn[0][high_atn].detach().numpy()):
            r, c = coords_to_image(arr[1], atn.shape[-1], observation.shape[-1])
            rd, cd = coords_to_image(arr[2], atn.shape[-1], observation.shape[-1])
            plt.arrow(c, r, cd-c, rd-r, width=0.05, head_width=1.5, alpha=weight)
        plt.show()

    def write_examples_to_strings(self, training=True):
        for example in self.example_list:
            feature_indices, atn, actions, q_value = example
            # Filter by top_n
            feature_indices = feature_indices[-self.top_n:]
            atn = atn[-self.top_n:]
            actions = actions[-self.top_n:]
            q_value[-self.top_n:]
            #
            # self.e_facts.append(np.unique(feature_indices))
            self.e_facts = np.append(self.e_facts, np.unique(feature_indices))
            facts = self.features_to_string(feature_indices)
            ind = np.cumsum(np.ones(facts.shape).astype(int), axis=1) - 1
            preds = self.atn_to_string_fast(atn, ind)
            acts, nacts, a_facts = self.actions_to_string(actions)
            if self.a_facts is None:
                self.add_facts(a_facts, facts, ind)
            flt = np.any(acts != "", axis=1)
            if not training:
                flt = np.full(facts.shape[0], True)
            return self.preds_and_facts_to_string(preds[flt], facts[flt], acts[flt], nacts[flt])

    def add_facts(self, a_facts, facts, ind):
        self.a_facts = a_facts[0]
        self.f_facts = concat_np_list(["feature(f", ind[0], ").\n"], ind[0].shape)
        n = int(np.sqrt(len(facts[0])))
        coord_n = np.array([_ for _ in range(n)])
        self.coords = concat_np_list(["coord(", coord_n, ").\n"], coord_n.shape)

    def features_to_string(self, feature_indices):
        fact_strings = feature_indices.astype(str)
        fact_strings = np.core.defchararray.add("e", fact_strings)
        return fact_strings

    def atn_to_string_fast(self, atn, facts):
        atn_str = np.round(atn, 2).astype(str)
        n = facts.shape[0]
        m = facts.shape[1]
        facts_sq = np.repeat(facts, m, axis=1).reshape((n, m, m))
        facts_2 = np.transpose(facts_sq, axes=[0, 2, 1])
        f_str = concat_np_list(["(f", facts_sq, ",f", facts_2, ")"], facts_sq.shape)

        f_str = f_str.reshape(n, 1, m, m)
        nr = atn.shape[1]
        f_str_nr = np.repeat(f_str, nr, axis=1)

        ones = np.ones(atn_str.shape).astype(int)
        ind = np.cumsum(ones, axis=1).astype(str)
        if self.r_facts is None:
            self.r_facts = ind[0, :, 0, 0]
        compound_list = [atn_str, "::r", ind, f_str_nr, ".\n"]
        filt = np.full(atn.shape, False)
        if not self.probabilistic:
            compound_list = ["r", ind, f_str_nr, ".\n"]
            filt = atn < self.atn_threshold
        rels = concat_np_list(compound_list, atn_str.shape)
        rels[filt] = ""
        n_out = np.prod(rels.shape[1:])
        output = rels.reshape((atn.shape[0], n_out))
        return output

    def preds_and_facts_to_string(self, preds, facts, acts, nacts):
        if len(facts) == 0:
            return False
        n = int(np.sqrt(len(facts[0])))
        ind = np.cumsum(np.ones(facts[0].shape).astype(int)) - 1
        # TODO: can it be done for many observations simultaneously?
        for i, _ in enumerate(preds):
            x_str = concat_np_list(["x(f", ind, ",", ind % n, ").\n"], ind.shape)
            y_str = concat_np_list(["y(f", ind, ",", ind // n, ").\n"], ind.shape)
            coords = ''.join(x_str) + ''.join(y_str)
            feats = concat_np_list([facts[i], "(f", ind, ").\n"], shape=facts[i].shape)
            f_string = ''.join(feats)
            a_str = ', '.join(acts[i][acts[i] != ""])
            na_str = ', '.join(nacts[i][nacts[i] != ""])
            p_str = ''.join(preds[i])
            example = ('\n\n'.join([f_string, coords, p_str]), a_str, na_str)
            self.example_strings.append(example)
        return True

    def actions_to_string(self, acts):
        # TODO: make sure this goes one at a time
        acts_str = np.round(acts, 2).astype(str)
        ones = np.ones(acts_str.shape).astype(int)
        ind = np.cumsum(ones, axis=1).astype(str)
        a_facts = concat_np_list(["action(a", ind, ").\n"], ind.shape)

        compound_list = [acts_str, "::take(a", ind, ").\n"]
        filt = np.full(acts.shape, False)
        if not self.probabilistic:
            compound_list = ["take(a", ind, ")"]
            filt = acts < self.action_threshold
        act_names = concat_np_list(compound_list, acts.shape)
        nact_names = copy.deepcopy(act_names)
        nact_names[np.bitwise_not(filt)] = ""
        act_names[filt] = ""
        return act_names, nact_names, a_facts

    def write_strings_to_file(self):
        output = self.generate_mode_bias()
        output += ''.join(self.a_facts) + "\n\n"
        output += ''.join(self.f_facts) + "\n\n"
        for i, example in enumerate(self.example_strings):
            output += f"#pos(eg_{i}@1,{{{example[1]}}},{{{example[2]}}},{{\n"
            output += example[0]
            output += "\n}).\n\n"
        write_string_to_file(output, self.learning_file)

    def generate_hypothesis(self):

        # This assumes my filepath and using WSL
        if os.name == "nt":
            filepath = os.path.join("/mnt/c/Users/titus/PycharmProjects/train-procgen-pytorch/", re.sub("\\\\", "/", self.learning_file))
        else:
            filepath = os.path.join("ilp/logic_examples/", self.learning_file)
        # Now we generate hypotheses
        cmd = create_cmd(["FastLAS", "--nopl", "--force-safety", filepath])
        output = run_subprocess(cmd, "\\n")
        # if output == "b''":
        #     print("FastLAS Error")
        #     return False
        # TODO: how can this be a float?
        self.hypothesis = output
        return True

    def generate_mode_bias(self):
        output = "#modeh(take(const(action))).\n"
        modeb_r = concat_np_list(["#modeb(r", self.r_facts, "(var(feature), var(feature))).\n"], self.r_facts.shape)
        output += ''.join(modeb_r)
        output += "#modeb(x(var(feature), const(coord))).\n"
        output += "#modeb(y(var(feature), const(coord))).\n"
        e_facts = np.unique(self.e_facts)
        modeb_r = concat_np_list(["#modeb(e", e_facts, "(var(feature))).\n"], e_facts.shape)
        output += ''.join(modeb_r)
        output += '\n\n#bias("allow_disjunction.").\n'
        # output += '\n\n:- take(A1), take(A2), A1 != A2.\n'
        return output

    def generate_action(self, observation):
        # ground the observation
        self.clear_examples()
        self.extract_example(observation)
        if not self.write_examples_to_strings(training=False):
            self.number_zero_actions_clingo += 1
            return np.random.randint(len(self.a_facts))
        # add the hypothesis
        self.write_program()
        # generate actions
        result = self.run_clingo()
        actions = extract_clingo_solution(result)
        # sample from actions
        action = np.random.choice(actions)
        if len(action) == 0:
            self.number_zero_actions_clingo += 1
            # TODO: check this number
            return np.random.randint(len(self.a_facts))
        # the reason for -1 is that action names start from 1 instead of 0:
        return int(re.search(r"take\(a(\d*)\)", action).group(1)) - 1

    def run_clingo(self):
        filepath = os.path.join("/mnt/c/Users/titus/PycharmProjects/VAE/", re.sub("\\\\", "/", self.clingo_file))
        cmd = create_cmd(["clingo", filepath])
        # TODO: try to get this working
        # cmd = ["wsl", f'echo "{self.clingo_file_contents}" | clingo /dev/stdin']
        output = run_subprocess(cmd, "\\n")
        return output

    def write_program(self):
        output = ""
        output += ''.join(self.a_facts) + "\n\n"
        output += ''.join(self.f_facts) + "\n\n"
        output += self.example_strings[0][0] + "\n\n"
        output += self.hypothesis
        output += "\n\n#show take/1.\n"
        self.clingo_file_contents = re.sub("\n", " ", output)
        write_string_to_file(output, self.clingo_file)

    def clear_examples(self):
        self.example_list = []
        self.example_strings = []


def concat_np_list(l, shape):
    output = np.repeat("", np.prod(shape)).reshape(shape)
    for arr in l:
        if type(arr) == np.ndarray:
            arr = arr.astype(str)
        output = np.core.defchararray.add(output, arr)
    return output


def main():
    # actor_path = "trained_models/test_VQMHA_tf_functions"
    actor_path = "trained_models/coinrun_ppo_actor_episode_578_running_reward_9.60_n_epchs_10000_n_hdn_128_n_res_hdn_32_n_res_lyrs_2_reduction_factor_16_embedding_dim_64_num_embeddings_512_cc_0.25_vq_use_ema_True_decay_0.99_lr_0.0003_attn_hds_8_dff_256_n_levels_1%2E"
    critic_path = "trained_models/coinrun_ppo_critic_episode_578_running_reward_9.60_n_epchs_10000_n_hdn_128_n_res_hdn_32_n_res_lyrs_2_reduction_factor_16_embedding_dim_64_num_embeddings_512_cc_0.25_vq_use_ema_True_decay_0.99_lr_0.0003_attn_hds_8_dff_256_n_levels_1%2E"
    logicDistiller = LogicDistiller(actor_path, critic_path, False, 0.6, 0.6)
    env_name = "procgen:procgen-coinrun-v0"
    env = gym.make(env_name, start_level=0, num_levels=1)
    observation = input_to_state(env.reset())
    logicDistiller.extract_example(observation)
    logicDistiller.write_examples_to_strings()
    print(logicDistiller.example_strings[0])





def play_logic():
    actor_path = "trained_models/coinrun_ppo_actor_episode_578_running_reward_9.60_n_epchs_10000_n_hdn_128_n_res_hdn_32_n_res_lyrs_2_reduction_factor_16_embedding_dim_64_num_embeddings_512_cc_0.25_vq_use_ema_True_decay_0.99_lr_0.0003_attn_hds_8_dff_256_n_levels_1%2E"
    critic_path = "trained_models/coinrun_ppo_critic_episode_578_running_reward_9.60_n_epchs_10000_n_hdn_128_n_res_hdn_32_n_res_lyrs_2_reduction_factor_16_embedding_dim_64_num_embeddings_512_cc_0.25_vq_use_ema_True_decay_0.99_lr_0.0003_attn_hds_8_dff_256_n_levels_1%2E"
    logicDistiller = LogicDistiller(actor_path, critic_path, probabilistic=False, atn_threshold=.6, action_threshold=.3)

    num_levels = 1
    env_name = "procgen:procgen-coinrun-v0"
    env = gym.make(env_name, start_level=0, num_levels=num_levels)

    df = pd.read_csv("logic_examples/results.csv")
    dft = pd.read_csv("logic_examples/results_timings.csv")
    df_code = df["top_n"].astype(str) + "_" + df["action_threshold"].astype(str)
    dft_code = dft["top_n"].astype(str) + "_" + dft["action_threshold"].astype(str)
    df = df[np.invert(np.isin(df_code, dft_code))]
    df = df[df.hypothesis.notna()]
    for row in df.index:
        logicDistiller.action_threshold = df.action_threshold[row]
        logicDistiller.hypothesis = df.hypothesis[row]

        reward_history = []
        ep_length = []

        for i in range(10):
            observation = input_to_state(env.reset())
            done = False
            length = 0
            while not done:
                action = logicDistiller.generate_action(observation)
                observation, reward, done, _ = env.step(action)
                observation = input_to_state(observation)
                length += 1
            ep_length.append(length)
            reward_history.append(reward)
        data = {"top_n": [df.top_n[row]],
                "action_threshold": [logicDistiller.action_threshold],
                "av_reward": [np.mean(reward_history)],
                "av_ep_length": [np.mean(ep_length)]}
        df2 = pd.DataFrame(data)
        append_to_csv_if_exists(df2, "logic_examples/results_timings.csv")
    return


def play_logic_record_gif():
    actor_path = "trained_models/coinrun_ppo_actor_episode_578_running_reward_9.60_n_epchs_10000_n_hdn_128_n_res_hdn_32_n_res_lyrs_2_reduction_factor_16_embedding_dim_64_num_embeddings_512_cc_0.25_vq_use_ema_True_decay_0.99_lr_0.0003_attn_hds_8_dff_256_n_levels_1%2E"
    critic_path = "trained_models/coinrun_ppo_critic_episode_578_running_reward_9.60_n_epchs_10000_n_hdn_128_n_res_hdn_32_n_res_lyrs_2_reduction_factor_16_embedding_dim_64_num_embeddings_512_cc_0.25_vq_use_ema_True_decay_0.99_lr_0.0003_attn_hds_8_dff_256_n_levels_1%2E"
    logicDistiller = LogicDistiller(actor_path, critic_path, probabilistic=False, atn_threshold=.6,
                                    action_threshold=.5)
    df = pd.read_csv("logic_examples/results.csv")
    row = 6
    logicDistiller.action_threshold = df.action_threshold[row]
    logicDistiller.hypothesis = df.hypothesis[row]

    start_level = 2
    for start_level in range(10):
        num_levels = 1
        env_name = "procgen:procgen-coinrun-v0"
        env = gym.make(env_name, start_level=start_level, num_levels=num_levels)

        for i in range(num_levels):
            observation = input_to_state(env.reset())
            if i > 0:
                frames = np.append(frames, observation, axis=0)
            else:
                frames = observation
            done = False
            while not done and len(frames) < 300:
                action = logicDistiller.generate_action(observation)
                observation, reward, done, _ = env.step(action)
                observation = input_to_state(observation)
                frames = np.append(frames, observation, axis=0)
        # save a gif
        data = (frames.astype("float32") + 0.5) * 255
        random_actions = f"{logicDistiller.number_zero_actions_clingo}_of_{len(frames)}"
        save_gif(data,
                 filename=f"output_images/logic_player_top5_cuttoff_0.5_level{start_level}_random_acts_{random_actions}.gif")
        print(f"Number of zero actions from clingo/frames: {random_actions})")
    return
    for i in range(2):
        observation = input_to_state(env.reset())
        if i > 0:
            frames = np.append(frames, observation, axis=0)
        else:
            frames = observation
        done = False
        while not done and len(frames) < 300:
            _, action = sample_action(logicDistiller.actor, observation)
            observation, reward, done, _ = env.step(action[0].numpy())
            observation = input_to_state(observation)
            frames = np.append(frames, observation, axis=0)
    # save a gif
    data = (frames.astype("float32") + 0.5) * 255
    # if len(data) > 300:
    #     data = data[-300:]
    save_gif(data, filename="output_images/Neural_player2.gif")
    # print(f"Number of zero actions from clingo/frames: {logicDistiller.number_zero_actions_clingo}/{len(frames)})")


# if __name__ == "__main__":
#     train_logic_program()
#     exit(0)
#     # load model
#     device = torch.device('cpu')
#     # logdir = "logs/train/coinrun/coinrun/2024-02-04__17-32-32__seed_6033/"
#     logdir = None
#     logdir = "logs/train/coinrun/coinrun/2024-02-12__09-20-18__seed_6033/"
#     action_names, done, env, hidden_state, obs, policy, storage = load_policy(False, logdir, n_envs=2,
#                                                                               hparams="hard-500-impalafsqmha")
#
#     # create_logicdistiller
#     ld = LogicDistiller(policy, device)
#     ld.extract_example(obs)
#     ld.write_examples_to_strings()
#     ld.write_strings_to_file()
#     ld.run_clingo()

# TODO: x and y coords can be moved to background knowledge
#   add other take(an) as negative examples
