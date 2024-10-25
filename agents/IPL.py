import wandb

from .base_agent import BaseAgent
from common.misc_util import adjust_lr, cross_batch_entropy, adjust_lr_grok
import torch
import torch.optim as optim
import numpy as np


class IPL(BaseAgent):
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
                 normalize_adv=True,
                 normalize_rew=True,
                 increasing_lr=False,
                 env_greedy=None,
                 storage_greedy=None,
                 learned_gamma=False,
                 accumulate_all_grads=False,
                 **kwargs):
        super(IPL, self).__init__(env, policy, logger, storage, device,
                                  n_checkpoints, env_valid, storage_valid)
        self.accumulate_all_grads = accumulate_all_grads
        self.learned_gamma = learned_gamma
        self.env_greedy = env_greedy
        self.storage_greedy = storage_greedy
        self.total_timesteps = 0
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
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        if increasing_lr:
            self.adjust_lr = adjust_lr_grok
        else:
            self.adjust_lr = adjust_lr

    def predict(self, obs, greedy=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, None, None)
            if greedy:
                if self.policy.continuous_actions:
                    act = dist.loc
                else:
                    act = dist.probs.argmax(dim=-1)
            else:
                act = dist.sample()

        return act.cpu().numpy(), value.cpu().numpy()

    def predict_w_value_saliency(self, obs, hidden_state, done):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
        mask = torch.FloatTensor(1 - done).to(device=self.device)
        dist, value, hidden_state = self.policy(obs, hidden_state, mask)
        value.backward(retain_graph=True)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return act.detach().cpu().numpy(), log_prob_act.detach().cpu().numpy(), value.detach().cpu().numpy(), hidden_state.detach().cpu().numpy(), obs.grad.data.detach().cpu().numpy()

    def optimize(self):
        # Loss and info:
        mutual_info_list, entropy_list, total_loss_list, corr_list, gamma_list = [], [], [], [], []

        batch_size = self.n_steps * self.n_envs // self.n_minibatch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        gamma = self.gamma
        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, nobs_batch, act_batch, done_batch, _, rew_batch = sample

                dist_batch, value_batch, _ = self.policy(obs_batch, None, None)
                _, next_value_batch, _ = self.policy(nobs_batch, None, None)
                next_value_batch[done_batch.bool()] = 0

                if self.learned_gamma:
                    gamma = self.policy.gamma()

                predicted_reward = dist_batch.log_prob(act_batch) + value_batch - gamma * next_value_batch

                loss = torch.nn.functional.mse_loss(predicted_reward, rew_batch)

                corr = torch.corrcoef(torch.stack((predicted_reward, rew_batch)))[0,1]

                mutual_info, entropy, = cross_batch_entropy(dist_batch)

                if loss.isnan():
                    print("nan loss")
                loss.backward()

                # Let model handle the large batch-size with small gpu-memory
                if not self.accumulate_all_grads and grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                entropy_list.append(entropy.item())
                mutual_info_list.append(mutual_info.item())
                total_loss_list.append(loss.item())
                corr_list.append(corr.item())
                gamma_list.append(gamma.item() if self.learned_gamma else gamma)
            if self.accumulate_all_grads:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Adjust common/Logger.__init__ if you add/remove from summary:
        summary = {'Loss/entropy': np.mean(entropy_list),
                   'Loss/mutual_info': np.mean(mutual_info_list),
                   'Loss/total': np.mean(total_loss_list),
                   'Loss/rew_corr': np.mean(corr_list),
                   'Loss/gamma': np.mean(gamma_list),
                   }
        self.last_predicted_reward = predicted_reward.detach().cpu().numpy()
        self.last_reward = rew_batch.detach().cpu().numpy()
        return summary

    def train(self, num_timesteps):
        self.total_timesteps = num_timesteps
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        checkpoints = [(i + 1) * save_every for i in range(self.num_checkpoints)]
        checkpoints.sort()
        obs = self.env.reset()

        if self.env_valid is not None:
            obs_v = self.env_valid.reset()

        if self.env_greedy is not None:
            obs_g = self.env_greedy.reset()

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            obs = self.collect_data(obs, self.storage, self.env)

            # valid
            if self.env_valid is not None:
                obs_v = self.collect_data(obs_v, self.storage_valid, self.env_valid)

            if self.env_greedy is not None:
                obs_g = self.collect_data(obs_g, self.storage_greedy, self.env_greedy, greedy=True)

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

            if self.storage_greedy is not None:
                rew_batch_g, done_batch_g, true_average_reward_g = self.storage_greedy.fetch_log_data()
            else:
                rew_batch_g = done_batch_g = true_average_reward_g = None

            self.logger.feed(rew_batch, done_batch, true_average_reward,
                             rew_batch_v, done_batch_v, true_average_reward_v,
                             rew_batch_g, done_batch_g, true_average_reward_g,
                             )

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
        if self.logger.use_wandb:
            data = [[x, y] for (x, y) in zip(self.last_reward, self.last_predicted_reward)]
            table = wandb.Table(data=data, columns=["Reward", "Predicted Reward"])
            wandb.log({"predicted reward": wandb.plot.scatter(table, "Reward", "Predicted Reward",
                                                             title="Predicted Reward vs Reward")})
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()
        if self.env_greedy is not None:
            self.env_greedy.close()

    def collect_data(self, obs, storage, env, greedy=False):
        for _ in range(self.n_steps):
            act, value = self.predict(obs, greedy)
            next_obs, rew, done, info = env.step(act)
            storage.store(obs, act, rew, done, info, value)
            obs = next_obs
        _, last_val = self.predict(obs, greedy)
        storage.store_last(obs, last_val)
        return obs
