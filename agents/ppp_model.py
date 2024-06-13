from torch.nn import MSELoss, BCELoss

from .base_agent import BaseAgent
from common.misc_util import adjust_lr, get_n_params, cross_batch_entropy, attention_entropy, adjust_lr_grok, \
    sparsity_loss
import torch
import torch.optim as optim
import numpy as np


def adjust_temp(temperature, timesteps, max_timesteps):
    new_temp = temperature * (1 - (timesteps / max_timesteps))
    return max(new_temp, 1e-20)


class PPPModel(BaseAgent):
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
                 n_minibatch=8,
                 mini_batch_size=32 * 8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 rew_coef=.5,
                 done_coef=5.,
                 trans_coef=1.,
                 epochs=3,
                 normalize_adv=True,
                 normalize_rew=True,
                 clip_value=True,
                 use_gae=True,
                 increasing_lr=False,
                 anneal_temp=False,
                 **kwargs):
        super(PPPModel, self).__init__(env, policy, logger, storage, device,
                                       n_checkpoints, env_valid, storage_valid)

        self.trans_coef = trans_coef
        self.anneal_temperature = anneal_temp
        self.epoch = epochs
        self.done_coef = done_coef
        self.clip_value = clip_value
        self.total_timesteps = 0
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_minibatch = n_minibatch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.rew_coef = rew_coef
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
            dist, value = self.policy(obs)
            act = dist.sample()
        return act.cpu().numpy(), value.cpu().numpy()

    def optimize(self):
        # Losses:
        total_loss_list, rew_loss_list, cont_loss_list = [], [], []
        t_loss_list, value_loss_list, ent_loss_list, x_ent_loss_list = [], [], [], []

        batch_size = self.n_steps * self.n_envs // self.n_minibatch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, nobs_batch, act_batch, done_batch, \
                    old_value_batch, return_batch, adv_batch, rew_batch = sample
                dist_batch, value_batch = self.policy(obs_batch)

                x_batch_ent_loss, entropy_loss, = cross_batch_entropy(dist_batch)

                h_batch = self.policy.encoder(obs_batch)

                flt = done_batch == 0
                nobs_guess = self.policy.sub_policy.transition_model(h_batch[flt], act_batch[flt])
                t_loss = MSELoss()(nobs_guess, nobs_batch[flt])

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip,
                                                                                              self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()
                if not self.clip_value:
                    value_loss = v_surr1.mean()


                done_guess, reward_guess = self.policy.sub_policy.dones_rewards(h_batch, act_batch)
                reward_loss = MSELoss()(reward_guess, rew_batch)
                done_loss = BCELoss()(done_guess, done_batch)

                loss = (reward_loss * self.rew_coef
                        + done_loss * self.done_coef
                        + value_loss * self.value_coef
                        + t_loss + self.trans_coef)
                loss.backward()


                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1

                total_loss_list.append(loss.item())
                value_loss_list.append(value_loss.item())
                rew_loss_list.append(reward_loss.item())
                cont_loss_list.append(done_loss.item())
                t_loss_list.append(t_loss.item())
                ent_loss_list.append(entropy_loss.item())
                x_ent_loss_list.append(x_batch_ent_loss.item())
        # Adjust common/Logger.__init__ if you add/remove from summary:
        summary = {'Loss/v': np.mean(value_loss_list),
                   'Loss/transition': np.mean(t_loss_list),
                   'Loss/entropy': np.mean(ent_loss_list),
                   'Loss/x_entropy': np.mean(x_ent_loss_list),
                   'Loss/reward': np.mean(rew_loss_list),
                   'Loss/continuation': np.mean(cont_loss_list),
                   'Loss/total': np.mean(total_loss_list),
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
            done_v = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, value = self.predict(obs)
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(obs, act, rew, done, info, value)
                obs = next_obs
            value_batch = self.storage.value_batch[:self.n_steps]
            _, last_val = self.predict(obs)
            self.storage.store_last(obs, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # valid
            if self.env_valid is not None:
                for _ in range(self.n_steps):
                    act_v, value_v = self.predict(obs_v)
                    next_obs_v, rew_v, done_v, info_v = self.env_valid.step(act_v)
                    self.storage_valid.store(obs_v, act_v,
                                             rew_v, done_v, info_v,
                                             value_v)
                    obs_v = next_obs_v
                _, last_val_v = self.predict(obs_v)
                self.storage_valid.store_last(obs_v, last_val_v)
                self.storage_valid.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch, true_average_reward = self.storage.fetch_log_data()
            print(f"Mean Reward:{np.mean(rew_batch[done_batch > 0]):.2f}")
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v, true_average_reward_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = true_average_reward_v = None
            self.logger.feed(rew_batch, done_batch, true_average_reward, rew_batch_v, done_batch_v,
                             true_average_reward_v)

            self.v_optimizer, lr = self.adjust_lr(self.v_optimizer, self.v_learning_rate, self.t, num_timesteps)
            self.t_optimizer, _ = self.adjust_lr(self.t_optimizer, self.t_learning_rate, self.t, num_timesteps)
            if self.anneal_temperature:
                self.policy.temperature = adjust_temp(self.policy.temperature, self.t, num_timesteps)

            self.logger.dump(summary, lr)
            # Save the model
            # if self.t > ((checkpoint_cnt + 1) * save_every):
            if self.t > checkpoints[checkpoint_cnt]:
                print("Saving model.")
                torch.save({'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.v_optimizer.state_dict()},
                           self.logger.logdir + '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()
