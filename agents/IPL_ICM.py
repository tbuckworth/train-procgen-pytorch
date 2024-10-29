import copy

import matplotlib.pyplot as plt
import wandb
from torch import nn

from .base_agent import BaseAgent
from common.misc_util import adjust_lr, cross_batch_entropy, adjust_lr_grok
import torch
import torch.optim as optim
import numpy as np


class IPL_ICM(BaseAgent):
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
                 learned_temp=False,
                 reward_incentive=False,
                 adv_incentive=False,
                 accumulate_all_grads=False,
                 alpha_learning_rate=2.5e-4,
                 target_entropy_coef=0.5,
                 beta=1,
                 novelty_loss_coef=1,
                 zv_loss_coef=0,
                 upload_obs=False,
                 alpha=1,
                 separate_icm=False,
                 **kwargs):
        super(IPL_ICM, self).__init__(env, policy, logger, storage, device,
                                      n_checkpoints, env_valid, storage_valid)
        self.separate_icm = separate_icm
        self.alpha = alpha
        self.upload_obs = upload_obs
        self.last_obs = []
        self.n_transition_guesses = 3
        self.n_imagined_actions = 2
        self.novelty_loss_coef = novelty_loss_coef
        self.zv_loss_coef = zv_loss_coef
        self.beta = beta
        self.target_entropy = self.policy.target_entropy * target_entropy_coef
        self.adv_incentive = adv_incentive
        self.reward_incentive = reward_incentive
        self.learned_temp = learned_temp
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
        self.nll_loss = nn.GaussianNLLLoss(reduction='none')

        params = [param for name, param in self.policy.named_parameters() if name != "log_alpha"]

        self.optimizer = optim.Adam(params, lr=learning_rate, eps=1e-5)
        if self.learned_temp:
            self.alpha_optimizer = optim.Adam([self.policy.log_alpha], lr=alpha_learning_rate, eps=1e-5)
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
            dist, value, hidden = self.policy(obs)
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
        mutual_info_list, entropy_list, total_loss_list, corr_list, gamma_list, alpha_list, alpha_loss_list = [], [], [], [], [], [], []
        pred_rew_list, novelty_loss_list, loss_list, zv_loss_list = [], [], [], []
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

                dist_batch, value_batch, h_batch = self.policy(obs_batch)

                if self.n_imagined_actions > 0:
                    # Imagined next states
                    # acts_imagined = dist_batch.sample() # sample a lot!
                    acts_imagined = torch.stack([dist_batch.sample() for _ in range(self.n_imagined_actions)])
                    h_batch_imagined = h_batch.tile(self.n_imagined_actions, 1, 1)
                    # TODO: remove duplicate sampled actions
                    nx_dist_imagined = self.policy.next_state(h_batch_imagined, acts_imagined)
                    # pred_h_next_imagined = nx_dist_imagined.sample()  # sample a lot!
                    pred_h_next_imagined = torch.concat([nx_dist_imagined.sample() for _ in range(self.n_transition_guesses)])
                    _, next_value_batch_imagined = self.policy.hidden_to_output(pred_h_next_imagined)

                    # necessary to repeat actions to align with next state samples:
                    act_tile_shape = (self.n_transition_guesses, *[1 for _ in acts_imagined.shape[1:]])
                    acts_imagined = acts_imagined.tile(act_tile_shape)

                # real next value
                _, next_value_batch_real, next_h_batch = self.policy(nobs_batch)
                next_value_batch_real[done_batch.bool()] = rew_batch[done_batch.bool()]

                if self.separate_icm:
                    # must clone, otherwise modified in place error on backwards pass later.
                    next_h_batch_clone = next_h_batch.detach().clone()
                    next_h_batch_clone[done_batch.bool()] = 0
                    # intrinsic reward TODO: maybe we don't want it to update h_batch?
                    nx_dist = self.policy.next_state(h_batch, act_batch)
                    # need to filter out dones!
                    novelty_loss_all = self.nll_loss(nx_dist.loc, next_h_batch_clone, nx_dist.scale**2)
                    novelty_loss = novelty_loss_all.mean()

                    # we want value to be zero when hidden is zero, because this is guessing termination
                    if done_batch.sum()>0:
                        _, term_value = self.policy.hidden_to_output(next_h_batch_clone[done_batch.bool()])
                        term_value.unsqueeze(0)
                        # actually we need it to equal the terminal reward
                        tv_loss = ((term_value-rew_batch[done_batch.bool()])**2).mean()
                    else:
                        tv_loss = 0

                if self.learned_gamma:
                    gamma = self.policy.gamma()
                if self.learned_temp:
                    log_prob_act = dist_batch.log_prob(act_batch).detach()
                    alpha = self.policy.alpha().detach().item()
                    # log_prob_act.requires_grad = False
                    alpha_loss = - self.policy.log_alpha * (log_prob_act + self.target_entropy).mean()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss_list.append(alpha_loss.item())
                else:
                    alpha = self.alpha

                if self.n_imagined_actions > 0:
                    next_value_batch = torch.concat((
                        next_value_batch_real.unsqueeze(0),
                        next_value_batch_imagined,
                    ), dim=0)
                    log_prob_act = torch.concat((
                        dist_batch.log_prob(act_batch).unsqueeze(0),
                        dist_batch.log_prob(acts_imagined),
                    ), dim=0)
                else:
                    next_value_batch = next_value_batch_real
                    log_prob_act = dist_batch.log_prob(act_batch)

                predicted_reward = alpha * log_prob_act + value_batch - gamma * next_value_batch

                if self.separate_icm:
                    target_reward = rew_batch
                else:
                    target_reward = (rew_batch + self.beta * novelty_loss_all.mean(-1).detach())

                loss = torch.nn.functional.mse_loss(predicted_reward, target_reward)

                if self.reward_incentive:
                    loss -= predicted_reward.mean()

                if self.adv_incentive:
                    # TODO: fix for continuous actions!
                    loss -= dist_batch.logits.max(dim=-1)[0].mean()

                # # (next_value_batch * torch.exp(dist_batch.log_prob(act_batch))).mean()
                # # (next_value_batch * (1-torch.exp(dist_batch.log_prob(act_batch)))).mean()
                # #
                # #
                # plt.scatter(
                #     x=predicted_reward.detach().cpu().numpy(),
                #     y=value_batch.detach().cpu().numpy(),
                # )
                #             # x=torch.exp(dist_batch.log_prob(act_batch)).detach().cpu().numpy(),)
                # plt.xlabel("pred rew")
                # plt.ylabel("value")
                # plt.show()
                # #
                # # torch.corrcoef(torch.stack((predicted_reward[done_batch.bool()], value_batch[done_batch.bool()])))
                # #
                # plt.hist(dist_batch.log_prob(act_batch).detach().cpu().numpy())
                # plt.show()

                # -1 should be the non-imaginary component
                corr = torch.corrcoef(torch.stack((predicted_reward[0], rew_batch)))[0, 1]

                mutual_info, entropy, = cross_batch_entropy(dist_batch)

                if self.separate_icm:
                    total_loss = loss
                else:
                    total_loss = loss + self.novelty_loss_coef * novelty_loss + self.zv_loss_coef * tv_loss
                total_loss.backward()
                # Let model handle the large batch-size with small gpu-memory
                if not self.accumulate_all_grads and grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                grad_accumulation_cnt += 1
                entropy_list.append(entropy.item())
                mutual_info_list.append(mutual_info.item())
                total_loss_list.append(total_loss.item())
                corr_list.append(corr.item())
                gamma_list.append(gamma.item() if self.learned_gamma else gamma)
                alpha_list.append(alpha)
                pred_rew_list.append(predicted_reward.mean().item())
                novelty_loss_list.append(novelty_loss.item())
                loss_list.append(loss.item())
                zv_loss_list.append(tv_loss.item())

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
                   'Loss/alpha': np.mean(alpha_loss_list),
                   'alpha': np.mean(alpha_list),
                   'Loss/pred_reward': np.mean(pred_rew_list),
                   'Loss/novelty_loss': np.mean(novelty_loss_list),
                   'Loss/reward_loss': np.mean(loss_list),
                   'Loss/zero_val_loss': np.mean(zv_loss_list),
                   }
        self.last_predicted_reward = predicted_reward.detach().cpu().numpy()
        self.last_reward = rew_batch.detach().cpu().numpy()
        self.last_target_reward = target_reward.cpu().numpy()
        if self.upload_obs:
            self.last_obs = [x.cpu().numpy().squeeze() for x in torch.split(obs_batch.detach(), 1, dim=-1)]
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
            data = [list(x) for x in
                    zip(self.last_reward, self.last_predicted_reward, self.last_target_reward, *self.last_obs)]
            col_names = [
                "Reward",
                "Predicted Reward",
                "Target Reward",
            ]
            if self.upload_obs:
                col_names += self.env.get_ob_names()
            table = wandb.Table(data=data, columns=col_names)
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
            storage.store(obs, act, rew, done, info, value.squeeze())
            obs = next_obs
        _, last_val = self.predict(obs, greedy)
        storage.store_last(obs, last_val.squeeze())
        return obs
