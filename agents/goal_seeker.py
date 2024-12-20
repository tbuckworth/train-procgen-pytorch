from torch import nn
from torch.distributed import broadcast

from .base_agent import BaseAgent
from common.misc_util import adjust_lr, get_n_params, cross_batch_entropy, attention_entropy, adjust_lr_grok, \
    sparsity_loss
import torch
import torch.optim as optim
import numpy as np


class GoalSeeker(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 env_valid=None,
                 storage_valid=None,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 n_minibatch=8,
                 mini_batch_size=32 * 8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 x_entropy_coef=0.,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 entropy_scaling=None,
                 increasing_lr=False,
                 goal_loss_coef=1.,
                 distance_loss_coef=1.,
                 forward_loss_coef=1.,
                 action_loss_coef=1.,
                 use_planning_to_act=True,
                 **kwargs):
        super(GoalSeeker, self).__init__(env, policy, logger, storage, device,
                                         n_checkpoints, env_valid, storage_valid)
        self.use_planning_to_act = use_planning_to_act
        self.goal_loss_coef = goal_loss_coef
        self.distance_loss_coef = distance_loss_coef
        self.forward_loss_coef = forward_loss_coef
        self.action_loss_coef = action_loss_coef
        self.total_timesteps = 0
        self.entropy_scaling = entropy_scaling
        self.entropy_multiplier = 1.
        self.min_rew = -1.
        self.max_rew = 11.
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.n_minibatch = n_minibatch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.x_entropy_coef = x_entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae
        if increasing_lr:
            self.adjust_lr = adjust_lr_grok
        else:
            self.adjust_lr = adjust_lr

    def predict(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            dist, value, _ = self.policy(obs)
            if self.use_planning_to_act:
                act = self.policy.plan(obs)
            else:
                act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()

    def predict_w_value_saliency(self, obs):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        dist, value, _ = self.policy(obs)
        value.backward(retain_graph=True)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return act.detach().cpu().numpy(), log_prob_act.detach().cpu().numpy(), value.detach().cpu().numpy(), obs.grad.data.detach().cpu().numpy()

    def nll_loss(self, dist, target):
        return nn.GaussianNLLLoss()(dist.loc, target, dist.scale ** 2)

    def optimize(self):
        # Original PPO losses:
        pi_loss_list, value_loss_list, entropy_loss_list, total_loss_list = [], [], [], []
        # Extra losses/metrics:
        x_ent_loss_list, atn_entropy_list, atn_entropy_list2, fs_loss_list, s_loss_list = [], [], [], [], []

        action_loss_list, forward_loss_list, goal_loss_list, distance_loss_list = [], [], [], []

        batch_size = self.n_steps * self.n_envs // self.n_minibatch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = False
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                (obs_batch, nobs_batch, act_batch, done_batch, \
                 old_log_prob_act_batch, old_value_batch,
                 return_batch, adv_batch, goal_obs_batch,
                 goal_distance_batch
                 ) = sample

                dist_batch, value_batch, h_batch = self.policy(obs_batch)

                # Action prediction (for learning good latents)
                # TODO: is flt right here - get next_obs instead?
                flt = done_batch == 0
                act_hat = self.policy.predict_action(obs_batch[flt], nobs_batch[flt])
                if self.policy.continuous_actions:
                    action_loss = self.nll_loss(act_hat, act_batch[flt])
                else:
                    a_hot = torch.nn.functional.one_hot(act_batch[flt].to(torch.int64), self.policy.action_size).to(self.device)
                    action_loss = nn.CrossEntropyLoss()(act_hat.probs, a_hot.float())

                # Forward prediction
                next_hidden = self.policy.predict_next(obs_batch[flt], act_batch[flt])
                next_h_gold = self.policy.embedder(obs_batch[flt])
                forward_loss = self.nll_loss(next_hidden, next_h_gold)

                # Goal prediction
                goal_hidden = self.policy.predict_goal(obs_batch)
                goal_h_gold = self.policy.embedder(goal_obs_batch)
                goal_loss = self.nll_loss(goal_hidden, goal_h_gold)

                # Trajectory prediction
                distance = self.policy.traj_distance_hidden(h_batch, goal_h_gold)
                distance_loss = nn.MSELoss()(distance, goal_distance_batch)

                # TODO: or we do goal-seeking policy update

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                if adv_batch.shape != log_prob_act_batch.shape:
                    adv_batch = torch.tile(adv_batch.unsqueeze(-1), log_prob_act_batch.shape[1:])
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip,
                                                                                              self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                # entropy_loss = dist_batch.entropy().mean()
                x_batch_ent_loss, entropy_loss, = cross_batch_entropy(dist_batch)

                loss = (pi_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy_loss
                        - self.x_entropy_coef * x_batch_ent_loss
                        + self.action_loss_coef * action_loss
                        + self.forward_loss_coef * forward_loss
                        + self.goal_loss_coef * goal_loss
                        + self.distance_loss_coef * distance_loss
                        )
                if loss.isnan():
                    print("nan loss")
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(-pi_loss.item())
                value_loss_list.append(-value_loss.item())
                entropy_loss_list.append(entropy_loss.item())
                x_ent_loss_list.append(x_batch_ent_loss.item())
                total_loss_list.append(loss.item())
                action_loss_list.append(action_loss.item())
                forward_loss_list.append(forward_loss.item())
                goal_loss_list.append(goal_loss.item())
                distance_loss_list.append(distance_loss.item())

        # Adjust common/Logger.__init__ if you add/remove from summary:
        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list),
                   'Loss/x_entropy': np.mean(x_ent_loss_list),
                   'Loss/total': np.mean(total_loss_list),
                   'Loss/action': np.mean(action_loss_list),
                   'Loss/forward': np.mean(forward_loss_list),
                   'Loss/goal': np.mean(goal_loss_list),
                   'Loss/distance': np.mean(distance_loss_list),
                   }

        return summary

    def train(self, num_timesteps):
        self.total_timesteps = num_timesteps
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        checkpoints = [(i + 1) * save_every for i in range(self.num_checkpoints)]
        checkpoints.sort()
        obs = self.env.reset()
        done = np.zeros(self.n_envs)

        if self.env_valid is not None:
            obs_v = self.env_valid.reset()
            hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
            done_v = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value = self.predict(obs)
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(obs, act, rew, done, info, log_prob_act, value)
                obs = next_obs
            value_batch = self.storage.value_batch[:self.n_steps]
            _, _, last_val = self.predict(obs)
            self.storage.store_last(obs, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # valid
            if self.env_valid is not None:
                for _ in range(self.n_steps):
                    act_v, log_prob_act_v, value_v = self.predict(obs_v)
                    next_obs_v, rew_v, done_v, info_v = self.env_valid.step(act_v)
                    self.storage_valid.store(obs_v, act_v,
                                             rew_v, done_v, info_v,
                                             log_prob_act_v, value_v)
                    obs_v = next_obs_v
                _, _, last_val_v = self.predict(obs_v)
                self.storage_valid.store_last(obs_v, last_val_v)
                self.storage_valid.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch, true_average_reward = self.storage.fetch_log_data()
            # print(f"Mean Reward:{np.mean(rew_batch[done_batch > 0]):.2f}")
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v, true_average_reward_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = true_average_reward_v = None
            self.logger.feed(rew_batch, done_batch, true_average_reward, rew_batch_v, done_batch_v,
                             true_average_reward_v)

            self.optimizer, lr = self.adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            self.logger.dump(summary, lr)
            # Save the model
            # if self.t > ((checkpoint_cnt + 1) * save_every):
            if self.t > checkpoints[checkpoint_cnt]:
                print("Saving model.")
                torch.save({'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           self.logger.logdir + '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()
