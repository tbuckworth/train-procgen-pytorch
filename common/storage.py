import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from collections import deque


class Storage():

    def __init__(self, obs_shape, hidden_state_size, num_steps, num_envs, device, continuous_actions=False,
                 act_shape=None):
        self.continuous_actions = continuous_actions
        self.performance_track = {}
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.hidden_state_size = hidden_state_size
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs_batch = torch.zeros(self.num_steps + 1, self.num_envs, *self.obs_shape)
        self.hidden_states_batch = torch.zeros(self.num_steps + 1, self.num_envs, self.hidden_state_size)
        # TODO: if cont, then act_batch shape takes action_shape into consideration
        if self.continuous_actions:
            self.act_batch = torch.zeros(self.num_steps, self.num_envs, self.act_shape).squeeze()
            self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs, self.act_shape).squeeze()
        else:
            self.act_batch = torch.zeros(self.num_steps, self.num_envs)
            self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.value_batch = torch.zeros(self.num_steps + 1, self.num_envs)
        self.return_batch = torch.zeros(self.num_steps, self.num_envs)
        self.adv_batch = torch.zeros(self.num_steps, self.num_envs)
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, hidden_state, act, rew, done, info, log_prob_act, value):
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.hidden_states_batch[self.step] = torch.from_numpy(hidden_state.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.log_prob_act_batch[self.step] = torch.from_numpy(log_prob_act.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)

        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_hidden_state, last_value):
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.hidden_states_batch[-1] = torch.from_numpy(last_hidden_state.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        rew_batch = self.rew_batch
        if use_gae:
            A = 0
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]
                value = self.value_batch[i]
                next_value = self.value_batch[i + 1]

                delta = (rew + gamma * next_value * (1 - done)) - value
                self.adv_batch[i] = A = gamma * lmbda * A * (1 - done) + delta
        else:
            G = self.value_batch[-1]
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]

                G = rew + gamma * G * (1 - done)
                self.return_batch[i] = G

        self.return_batch = self.adv_batch + self.value_batch[:-1]
        if normalize_adv:
            self.adv_batch = (self.adv_batch - torch.mean(self.adv_batch)) / (torch.std(self.adv_batch) + 1e-8)

    def fetch_train_generator(self, mini_batch_size=None, recurrent=False):
        batch_size = self.num_steps * self.num_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size
        # If agent's policy is not recurrent, data could be sampled without considering the time-horizon
        if not recurrent:
            sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                                   mini_batch_size,
                                   drop_last=True)
            for indices in sampler:
                yield self.collate_data(indices)
        # If agent's policy is recurrent, data should be sampled along the time-horizon
        else:
            num_mini_batch_per_epoch = batch_size // mini_batch_size
            num_envs_per_batch = self.num_envs // num_mini_batch_per_epoch
            perm = torch.randperm(self.num_envs)
            for start_ind in range(0, self.num_envs, num_envs_per_batch):
                idxes = perm[start_ind:start_ind + num_envs_per_batch]
                obs_batch = torch.FloatTensor(self.obs_batch[:-1, idxes]).reshape(-1, *self.obs_shape).to(self.device)
                # [0:1] instead of [0] to keep two-dimensional array
                hidden_state_batch = torch.FloatTensor(self.hidden_states_batch[0:1, idxes]).reshape(-1,
                                                                                                     self.hidden_state_size).to(
                    self.device)
                act_batch = torch.FloatTensor(self.act_batch[:, idxes]).reshape(-1).to(self.device)
                done_batch = torch.FloatTensor(self.done_batch[:, idxes]).reshape(-1).to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch[:, idxes]).reshape(-1).to(self.device)
                value_batch = torch.FloatTensor(self.value_batch[:-1, idxes]).reshape(-1).to(self.device)
                return_batch = torch.FloatTensor(self.return_batch[:, idxes]).reshape(-1).to(self.device)
                adv_batch = torch.FloatTensor(self.adv_batch[:, idxes]).reshape(-1).to(self.device)
                yield obs_batch, hidden_state_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch

    def collate_data(self, indices):
        obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
        hidden_state_batch = torch.FloatTensor(self.hidden_states_batch[:-1]).reshape(-1,
                                                                                      self.hidden_state_size).to(
            self.device)
        if self.continuous_actions:
            act_batch = torch.FloatTensor(self.act_batch).reshape(-1, self.act_shape)[indices].to(self.device)
            log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1, self.act_shape)[
                indices].to(self.device)
        else:
            act_batch = torch.FloatTensor(self.act_batch).reshape(-1, )[indices].to(self.device)
            log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1)[indices].to(self.device)
        done_batch = torch.FloatTensor(self.done_batch).reshape(-1)[indices].to(self.device)
        value_batch = torch.FloatTensor(self.value_batch[:-1]).reshape(-1)[indices].to(self.device)
        return_batch = torch.FloatTensor(self.return_batch).reshape(-1)[indices].to(self.device)
        adv_batch = torch.FloatTensor(self.adv_batch).reshape(-1)[indices].to(self.device)
        return obs_batch, hidden_state_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch

    def fetch_log_data(self):
        if 'env_reward' in self.info_batch[0][0]:
            rew_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = self.rew_batch.numpy()
        if 'env_done' in self.info_batch[0][0]:
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = self.done_batch.numpy()
        if 'prev_level_seed' in self.info_batch[0][0]:
            for step in range(self.num_steps):
                infos = np.array(self.info_batch[step])
                completes = infos[done_batch[step] > 0]
                for info in completes:
                    seed = info["prev_level_seed"]
                    rew = info["env_reward"]
                    if seed not in self.performance_track.keys():
                        self.performance_track[seed] = deque(maxlen=10)
                    self.performance_track[seed].append(rew)
        else:
            # TODO: Implement for BoxWorld?
            true_average_reward = np.nan
        all_rewards = list(self.performance_track.values())
        true_average_reward = np.mean([rew for rew_list in all_rewards for rew in rew_list])
        return rew_batch, done_batch, true_average_reward


class BasicStorage():

    def __init__(self, obs_shape, num_steps, num_envs, device):
        self.performance_track = {}
        self.obs_shape = obs_shape
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs_batch = torch.zeros(self.num_steps + 1, self.num_envs, *self.obs_shape)
        self.act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.value_batch = torch.zeros(self.num_steps + 1, self.num_envs)
        self.return_batch = torch.zeros(self.num_steps, self.num_envs)
        self.adv_batch = torch.zeros(self.num_steps, self.num_envs)
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, act, rew, done, info, value):
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)

        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_value):
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        rew_batch = self.rew_batch
        if use_gae:
            A = 0
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]
                value = self.value_batch[i]
                next_value = self.value_batch[i + 1]

                delta = (rew + gamma * next_value * (1 - done)) - value
                self.adv_batch[i] = A = gamma * lmbda * A * (1 - done) + delta
        else:
            G = self.value_batch[-1]
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]

                G = rew + gamma * G * (1 - done)
                self.return_batch[i] = G

        self.return_batch = self.adv_batch + self.value_batch[:-1]
        if normalize_adv:
            self.adv_batch = (self.adv_batch - torch.mean(self.adv_batch)) / (torch.std(self.adv_batch) + 1e-8)

    def fetch_train_generator(self, mini_batch_size=None, recurrent=False):
        batch_size = self.num_steps * self.num_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size
        # If agent's policy is not recurrent, data could be sampled without considering the time-horizon
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)
        for indices in sampler:
            obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
            nobs_batch = torch.FloatTensor(self.obs_batch[1:]).reshape(-1, *self.obs_shape)[indices].to(self.device)
            act_batch = torch.FloatTensor(self.act_batch).reshape(-1)[indices].to(self.device)
            done_batch = torch.FloatTensor(self.done_batch).reshape(-1)[indices].to(self.device)
            value_batch = torch.FloatTensor(self.value_batch[:-1]).reshape(-1)[indices].to(self.device)
            return_batch = torch.FloatTensor(self.return_batch).reshape(-1)[indices].to(self.device)
            adv_batch = torch.FloatTensor(self.adv_batch).reshape(-1)[indices].to(self.device)
            rew_batch = torch.FloatTensor(self.rew_batch).reshape(-1)[indices].to(self.device)
            yield obs_batch, nobs_batch, act_batch, done_batch, value_batch, return_batch, adv_batch, rew_batch

    def fetch_log_data(self):
        if 'env_reward' in self.info_batch[0][0]:
            rew_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = self.rew_batch.numpy()
        if 'env_done' in self.info_batch[0][0]:
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = self.done_batch.numpy()
        if 'prev_level_seed' in self.info_batch[0][0]:
            for step in range(self.num_steps):
                infos = np.array(self.info_batch[step])
                completes = infos[done_batch[step] > 0]
                for info in completes:
                    seed = info["prev_level_seed"]
                    rew = info["env_reward"]
                    if seed not in self.performance_track.keys():
                        self.performance_track[seed] = deque(maxlen=10)
                    self.performance_track[seed].append(rew)
        else:
            # TODO: Implement for BoxWorld?
            true_average_reward = np.nan
        all_rewards = list(self.performance_track.values())
        true_average_reward = np.mean([rew for rew_list in all_rewards for rew in rew_list])
        return rew_batch, done_batch, true_average_reward


class IPLStorage():

    def __init__(self, obs_shape, num_steps, num_envs, device):
        self.performance_track = {}
        self.obs_shape = obs_shape
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs_batch = torch.zeros(self.num_steps + 1, self.num_envs, *self.obs_shape)
        self.act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.value_batch = torch.zeros(self.num_steps + 1, self.num_envs)
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, act, rew, done, info, value):
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)

        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_value):
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def fetch_train_generator(self, mini_batch_size=None, recurrent=False):
        batch_size = self.num_steps * self.num_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size
        # If agent's policy is not recurrent, data could be sampled without considering the time-horizon
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)
        for indices in sampler:
            obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
            nobs_batch = torch.FloatTensor(self.obs_batch[1:]).reshape(-1, *self.obs_shape)[indices].to(self.device)
            act_batch = torch.FloatTensor(self.act_batch).reshape(-1)[indices].to(self.device)
            done_batch = torch.FloatTensor(self.done_batch).reshape(-1)[indices].to(self.device)
            value_batch = torch.FloatTensor(self.value_batch[:-1]).reshape(-1)[indices].to(self.device)
            rew_batch = torch.FloatTensor(self.rew_batch).reshape(-1)[indices].to(self.device)
            yield obs_batch, nobs_batch, act_batch, done_batch, value_batch, rew_batch

    def fetch_log_data(self):
        if 'env_reward' in self.info_batch[0][0]:
            rew_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = self.rew_batch.numpy()
        if 'env_done' in self.info_batch[0][0]:
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = self.done_batch.numpy()
        if 'prev_level_seed' in self.info_batch[0][0]:
            for step in range(self.num_steps):
                infos = np.array(self.info_batch[step])
                completes = infos[done_batch[step] > 0]
                for info in completes:
                    seed = info["prev_level_seed"]
                    rew = info["env_reward"]
                    if seed not in self.performance_track.keys():
                        self.performance_track[seed] = deque(maxlen=10)
                    self.performance_track[seed].append(rew)
        else:
            # TODO: Implement for BoxWorld?
            true_average_reward = np.nan
        all_rewards = list(self.performance_track.values())
        true_average_reward = np.mean([rew for rew_list in all_rewards for rew in rew_list])
        return rew_batch, done_batch, true_average_reward


class GoalSeekerStorage():

    def __init__(self, obs_shape, num_steps, num_envs, device, continuous_actions=False, act_shape=None):
        self.continuous_actions = continuous_actions
        self.performance_track = {}
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs_batch = torch.zeros(self.num_steps + 1, self.num_envs, *self.obs_shape)
        # TODO: if cont, then act_batch shape takes action_shape into consideration
        if self.continuous_actions:
            self.act_batch = torch.zeros(self.num_steps, self.num_envs, self.act_shape).squeeze()
            self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs, self.act_shape).squeeze()
        else:
            self.act_batch = torch.zeros(self.num_steps, self.num_envs)
            self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.value_batch = torch.zeros(self.num_steps + 1, self.num_envs)
        self.return_batch = torch.zeros(self.num_steps, self.num_envs)
        self.adv_batch = torch.zeros(self.num_steps, self.num_envs)
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, act, rew, done, info, log_prob_act, value):
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.log_prob_act_batch[self.step] = torch.from_numpy(log_prob_act.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)

        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_value):
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        rew_batch = self.rew_batch
        if use_gae:
            A = 0
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]
                value = self.value_batch[i]
                next_value = self.value_batch[i + 1]

                delta = (rew + gamma * next_value * (1 - done)) - value
                self.adv_batch[i] = A = gamma * lmbda * A * (1 - done) + delta
        else:
            G = self.value_batch[-1]
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]

                G = rew + gamma * G * (1 - done)
                self.return_batch[i] = G

        self.return_batch = self.adv_batch + self.value_batch[:-1]
        if normalize_adv:
            self.adv_batch = (self.adv_batch - torch.mean(self.adv_batch)) / (torch.std(self.adv_batch) + 1e-8)

    def fetch_train_generator(self, mini_batch_size=None, recurrent=False):
        batch_size = self.num_steps * self.num_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size
        # If agent's policy is not recurrent, data could be sampled without considering the time-horizon
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)

        # goal_obs
        # goal_distance
        goal_obs = torch.zeros_like(self.obs_batch)
        goal_distance = torch.zeros((self.num_steps, self.num_envs))

        done_batch = torch.concat((self.done_batch, torch.tensor([[False for _ in range(self.num_envs)]])))
        for e in range(self.num_envs):
            for row in range(self.num_steps):
                in_range = done_batch[row + 1:, e].cumsum(0) == 0
                if in_range.any():
                    future_vals = self.value_batch[row + 1:, e][in_range]
                    goal_position = future_vals.argmax()
                    goal_obs[row, e] = self.obs_batch[row + goal_position, e]
                    goal_distance[row, e] = goal_position

        for indices in sampler:
            obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
            nobs_batch = torch.FloatTensor(self.obs_batch[1:]).reshape(-1, *self.obs_shape)[indices].to(self.device)
            if self.continuous_actions:
                act_batch = torch.FloatTensor(self.act_batch).reshape(-1, self.act_shape)[indices].to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1, self.act_shape)[indices].to(
                    self.device)
            else:
                act_batch = torch.FloatTensor(self.act_batch).reshape(-1, )[indices].to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1)[indices].to(self.device)
            done_batch = torch.FloatTensor(self.done_batch).reshape(-1)[indices].to(self.device)
            value_batch = torch.FloatTensor(self.value_batch[:-1]).reshape(-1)[indices].to(self.device)
            return_batch = torch.FloatTensor(self.return_batch).reshape(-1)[indices].to(self.device)
            adv_batch = torch.FloatTensor(self.adv_batch).reshape(-1)[indices].to(self.device)
            goal_obs_batch = goal_obs[:-1].reshape(-1, *self.obs_shape)[indices].to(self.device)
            goal_distance_batch = goal_distance.reshape(-1)[indices].to(self.device)
            # goal_obs
            # ` goal_distance
            yield (obs_batch, nobs_batch, act_batch, done_batch, log_prob_act_batch,
                   value_batch, return_batch, adv_batch, goal_obs_batch, goal_distance_batch)

    def fetch_log_data(self):
        if 'env_reward' in self.info_batch[0][0]:
            rew_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = self.rew_batch.numpy()
        if 'env_done' in self.info_batch[0][0]:
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = self.done_batch.numpy()
        if 'prev_level_seed' in self.info_batch[0][0]:
            for step in range(self.num_steps):
                infos = np.array(self.info_batch[step])
                completes = infos[done_batch[step] > 0]
                for info in completes:
                    seed = info["prev_level_seed"]
                    rew = info["env_reward"]
                    if seed not in self.performance_track.keys():
                        self.performance_track[seed] = deque(maxlen=10)
                    self.performance_track[seed].append(rew)
        else:
            # TODO: Implement for BoxWorld?
            true_average_reward = np.nan
        all_rewards = list(self.performance_track.values())
        true_average_reward = np.mean([rew for rew_list in all_rewards for rew in rew_list])
        return rew_batch, done_batch, true_average_reward


class SAEStorage(Storage):
    def __init__(self, obs_shape, hidden_state_size, num_steps, num_envs, device, continuous_actions=False,
                 act_shape=None):
        super().__init__(obs_shape, hidden_state_size, num_steps, num_envs, device, continuous_actions, act_shape)

    def reset(self):
        self.obs_batch = torch.zeros(self.num_steps + 1, self.num_envs, *self.obs_shape)
        self.hidden_batch = torch.zeros(self.num_steps + 1, self.num_envs, self.hidden_state_size)
        # if self.continuous_actions:
        #     self.act_batch = torch.zeros(self.num_steps, self.num_envs, self.act_shape).squeeze()
        #     self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs, self.act_shape).squeeze()
        # else:
        self.act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.logit_batch = torch.zeros(self.num_steps, self.num_envs, self.act_shape).squeeze()
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.value_batch = torch.zeros(self.num_steps + 1, self.num_envs)
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, hidden, act, rew, done, info, logits, value):
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.hidden_batch[self.step] = torch.from_numpy(hidden.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.logit_batch[self.step] = torch.from_numpy(logits.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)

        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_hidden, last_value):
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.hidden_batch[-1] = torch.from_numpy(last_hidden.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def collate_data(self, indices):
        obs_batch = self.reshape_data(self.obs_batch[:-1], self.obs_shape, indices)
        hidden_batch = self.reshape_data(self.hidden_batch[:-1], (self.hidden_state_size,), indices)
        logit_batch = self.reshape_data(self.logit_batch, (self.act_shape,), indices)
        value_batch = self.reshape_data(self.value_batch[:-1], (), indices)
        act_batch = self.reshape_data(self.act_batch, (), indices)
        return obs_batch, hidden_batch, act_batch, logit_batch, value_batch

        # obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
        # hidden_batch = torch.FloatTensor(self.hidden_batch[:-1]).reshape(-1,
        #                                                                               self.hidden_state_size)[indices].to(
        #     self.device)
        # logit_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1, self.act_shape)[
        #         indices].to(self.device)
        # act_batch = torch.FloatTensor(self.act_batch).reshape(-1)[indices].to(self.device)
        # value_batch = torch.FloatTensor(self.value_batch[:-1]).reshape(-1)[indices].to(self.device)
        # return obs_batch, hidden_batch, act_batch, logit_batch, value_batch

    def reshape_data(self, data, shape, indices):
        return torch.FloatTensor(data).reshape(-1, *shape)[indices].to(self.device)
