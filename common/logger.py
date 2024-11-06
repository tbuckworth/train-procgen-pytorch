import numpy as np
import pandas as pd
from collections import deque
import time
import csv

try:
    import wandb
except ImportError:
    pass


class Logger(object):

    def __init__(self, n_envs, logdir, use_wandb=False, has_vq=False,
                 algo="ppo",
                 # transition_model=False, double_graph=False, ppo_pure=False, IPL=False,
                 greedy=False):
        self.true_mean_reward = None
        self.true_mean_reward_v = None
        self.true_mean_reward_g = None
        self.greedy = greedy
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir
        self.use_wandb = use_wandb

        # training
        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])

        self.episode_timeout_buffer = deque(maxlen=40)
        self.episode_len_buffer = deque(maxlen=40)
        self.episode_reward_buffer = deque(maxlen=40)

        # validation
        self.episode_rewards_v = []
        for _ in range(n_envs):
            self.episode_rewards_v.append([])

        self.episode_timeout_buffer_v = deque(maxlen=40)
        self.episode_len_buffer_v = deque(maxlen=40)
        self.episode_reward_buffer_v = deque(maxlen=40)

        # greedy
        self.episode_rewards_g = []
        for _ in range(n_envs):
            self.episode_rewards_g.append([])

        self.episode_timeout_buffer_g = deque(maxlen=40)
        self.episode_len_buffer_g = deque(maxlen=40)
        self.episode_reward_buffer_g = deque(maxlen=40)

        time_metrics = ["timesteps", "wall_time", "num_episodes"]  # only collected once
        loss_metrics = ["loss_pi", "loss_v", "loss_entropy", "loss_x_entropy", "atn_entropy", "atn_entropy2",
                        "loss_sparsity", "loss_feature_sparsity", "loss_total"]
        if algo in ['ppo-model', 'graph-agent']:
            loss_metrics = ["loss_v", "loss_transition", "loss_entropy", "loss_x_entropy",
                            "loss_reward", "loss_continuation", "loss_total"]
        if algo in ['double-graph-agent']:
            loss_metrics = ["loss_v", "loss_transition", "loss_entropy", "loss_x_entropy", "loss_total"]
        if has_vq:
            loss_metrics = ["loss_pi", "loss_v", "loss_entropy", "loss_x_entropy", "loss_commit", "loss_total"]
        if algo in ['ppo-pure', 'espo']:
            loss_metrics = ["loss_pi", "loss_v", "loss_entropy", "loss_x_entropy", "loss_total"]
        if algo in ["IPL"]:
            loss_metrics = ["entropy", "mutual_info", "loss_total", "rew_corr", "gamma", "loss_alpha", "alpha",
                            "pred_reward"]
        if algo == "IPL_ICM":
            loss_metrics = ["entropy", "mutual_info", "loss_total", "rew_corr", "gamma", "loss_alpha", "alpha",
                            "pred_reward", "loss_novelty", "loss_reward", "loss_zero_value"]

        if algo == "goal-seeker":
            loss_metrics = ["loss_pi", "loss_v", "loss_entropy", "loss_x_entropy", "loss_total",
                            "loss_action", "loss_forward", "loss_goal", "loss_distance"]

        # Make sure this is consistent with _get_episode_statistics:
        episode_metrics = ["max_episode_rewards", "mean_episode_rewards", "median_episode_rewards",
                           "min_episode_rewards",
                           "max_episode_len", "mean_episode_len", "min_episode_len",
                           "mean_timeouts", "mean_episode_len_pos_reward",
                           "balanced_mean_rewards"]  # collected for both train and val envs
        all_columns = (time_metrics + episode_metrics +
                       ["val_" + m for m in episode_metrics] +
                       ["ema_rewards"] + loss_metrics + ["learning_rate"])
        if greedy:
            all_columns = (time_metrics + episode_metrics +
                           ["val_" + m for m in episode_metrics] +
                           ["greedy_" + m for m in episode_metrics] +
                           ["ema_rewards"] + loss_metrics + ["learning_rate"])

        self.log = pd.DataFrame(columns=all_columns)

        self.timesteps = 0
        self.num_episodes = 0

    def feed(self, rew_batch, done_batch, true_mean_reward,
             rew_batch_v=None, done_batch_v=None, true_mean_reward_v=None,
             rew_batch_g=None, done_batch_g=None, true_mean_reward_g=None
             ):
        self.true_mean_reward = true_mean_reward
        self.true_mean_reward_v = true_mean_reward_v
        self.true_mean_reward_g = true_mean_reward_g
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        valid = rew_batch_v is not None and done_batch_v is not None
        greedy = rew_batch_g is not None and done_batch_g is not None
        if valid:
            rew_batch_v = rew_batch_v.T
            done_batch_v = done_batch_v.T
        if greedy:
            rew_batch_g = rew_batch_g.T
            done_batch_g = done_batch_g.T

        # TODO: Vectorize this if possible
        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])

                if valid:
                    self.episode_rewards_v[i].append(rew_batch_v[i][j])

                if greedy:
                    self.episode_rewards_g[i].append(rew_batch_g[i][j])

                if done_batch[i][j]:
                    ep_length = len(self.episode_rewards[i])
                    self.episode_timeout_buffer.append(1 if ep_length == self.max_steps else 0)
                    self.episode_len_buffer.append(ep_length)
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1
                if valid and done_batch_v[i][j]:
                    ep_length = len(self.episode_rewards_v[i])
                    self.episode_timeout_buffer_v.append(1 if ep_length == self.max_steps else 0)
                    self.episode_len_buffer_v.append(ep_length)
                    self.episode_reward_buffer_v.append(np.sum(self.episode_rewards_v[i]))
                    self.episode_rewards_v[i] = []
                if greedy and done_batch_g[i][j]:
                    ep_length = len(self.episode_rewards_g[i])
                    self.episode_timeout_buffer_g.append(1 if ep_length == self.max_steps else 0)
                    self.episode_len_buffer_g.append(ep_length)
                    self.episode_reward_buffer_g.append(np.sum(self.episode_rewards_g[i]))
                    self.episode_rewards_g[i] = []

        self.timesteps += (self.n_envs * steps)

    def dump(self, summary={}, lr=0.):
        wall_time = time.time() - self.start_time
        episode_statistics = self._get_episode_statistics()  # 14
        episode_statistics_list = list(episode_statistics.values())  # 14
        loss_statistics = list(summary.values())  # 5 (x_ent = #4)
        ema_reward = episode_statistics['Rewards/mean_episodes']
        if len(self.log) > 0:
            smoothing = .99 / (1 + len(self.log))
            prev_ema = self.log["ema_rewards"].loc[len(self.log) - 1]
            ema_reward = ema_reward * smoothing + prev_ema * (1 - smoothing)
        log = [self.timesteps, wall_time, self.num_episodes] + episode_statistics_list + [
            ema_reward] + loss_statistics + [lr]
        self.log.loc[len(self.log)] = log

        with open(self.logdir + '/log-append.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(self.log.columns)
            writer.writerow(log)

        print(self.log.loc[len(self.log) - 1])

        if self.use_wandb:
            wandb.log({k: v for k, v in zip(self.log.columns, log)})

    def _get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes'] = np.max(self.episode_reward_buffer, initial=0)
        episode_statistics['Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer)
        episode_statistics['Rewards/median_episodes'] = np.median(self.episode_reward_buffer)
        episode_statistics['Rewards/min_episodes'] = np.min(self.episode_reward_buffer, initial=0)
        episode_statistics['Len/max_episodes'] = np.max(self.episode_len_buffer, initial=0)
        episode_statistics['Len/mean_episodes'] = np.mean(self.episode_len_buffer)
        episode_statistics['Len/min_episodes'] = np.min(self.episode_len_buffer, initial=0)
        episode_statistics['Len/mean_timeout'] = np.mean(self.episode_timeout_buffer)
        episode_statistics['Len/mean_episodes_pos_reward'] = np.mean(
            np.array(self.episode_len_buffer)[np.array(self.episode_reward_buffer) > 0])
        episode_statistics['Rewards/balanced_mean'] = self.true_mean_reward
        # valid
        episode_statistics['[Valid] Rewards/max_episodes'] = np.max(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer_v)
        episode_statistics['[Valid] Rewards/median_episodes'] = np.median(self.episode_reward_buffer_v)
        episode_statistics['[Valid] Rewards/min_episodes'] = np.min(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Len/max_episodes'] = np.max(self.episode_len_buffer_v, initial=0)
        episode_statistics['[Valid] Len/mean_episodes'] = np.mean(self.episode_len_buffer_v)
        episode_statistics['[Valid] Len/min_episodes'] = np.min(self.episode_len_buffer_v, initial=0)
        episode_statistics['[Valid] Len/mean_timeout'] = np.mean(self.episode_timeout_buffer_v)
        episode_statistics['[Valid] Len/mean_episodes_pos_reward'] = np.mean(
            np.array(self.episode_len_buffer_v)[np.array(self.episode_reward_buffer_v) > 0])
        episode_statistics['[Valid] Rewards/balanced_mean'] = self.true_mean_reward_v
        if self.greedy:
            episode_statistics['[Greedy] Rewards/max_episodes'] = np.max(self.episode_reward_buffer_g, initial=0)
            episode_statistics['[Greedy] Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer_g)
            episode_statistics['[Greedy] Rewards/median_episodes'] = np.median(self.episode_reward_buffer_g)
            episode_statistics['[Greedy] Rewards/min_episodes'] = np.min(self.episode_reward_buffer_g, initial=0)
            episode_statistics['[Greedy] Len/max_episodes'] = np.max(self.episode_len_buffer_g, initial=0)
            episode_statistics['[Greedy] Len/mean_episodes'] = np.mean(self.episode_len_buffer_g)
            episode_statistics['[Greedy] Len/min_episodes'] = np.min(self.episode_len_buffer_g, initial=0)
            episode_statistics['[Greedy] Len/mean_timeout'] = np.mean(self.episode_timeout_buffer_g)
            episode_statistics['[Greedy] Len/mean_episodes_pos_reward'] = np.mean(
                np.array(self.episode_len_buffer_g)[np.array(self.episode_reward_buffer_g) > 0])
            episode_statistics['[Greedy] Rewards/balanced_mean'] = self.true_mean_reward_g

        return episode_statistics
