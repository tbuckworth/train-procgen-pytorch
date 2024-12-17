from common.model import SparseAutoencoder, LinearSAEProbe
from .base_agent import BaseAgent
from common.misc_util import adjust_lr, get_n_params
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class SAE(BaseAgent):
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
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 l1_coef=0.,
                 anneal_lr=True,
                 sae_dim=1024,
                 rho=0.05,
                 sparse_coef=1e-3,
                 **kwargs):

        super(SAE, self).__init__(env, policy, logger, storage, device,
                                  n_checkpoints, env_valid, storage_valid)

        self.sparse_coef = sparse_coef
        self.rho = rho
        self.sae_dim = sae_dim
        self.anneal_lr = anneal_lr
        # self.l1_coef = l1_coef
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        # self.gamma = gamma
        # self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.sae = SparseAutoencoder(input_dim=self.policy.model.output_dim,
                                     hidden_dim=self.sae_dim,
                                     rho=self.rho).to(device)
        self.linear_model = LinearSAEProbe(self.sae_dim, self.policy.action_size)
        self.optimizer = optim.Adam(self.sae.parameters(), lr=learning_rate, eps=1e-5)
        self.l_optimizer = optim.Adam(self.linear_model.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        # self.eps_clip = eps_clip
        # self.value_coef = value_coef
        # self.entropy_coef = entropy_coef
        # self.normalize_adv = normalize_adv
        # self.normalize_rew = normalize_rew
        # self.use_gae = use_gae

    def get_hidden_and_acts(self, obs, policy=None):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            hidden = self.policy.embedder.forward_to_pool(obs)
            h = self.policy.embedder.forward_from_pool(hidden)
            p, v = self.policy.hidden_to_output(h)
            act = p.sample().cpu().numpy()
            if policy is not None:
                logits, v_hat = self.linear_model(self.sae(hidden)[1])
                dist = Categorical(logits=logits)
                act = dist.sample().cpu().numpy()
        return act, hidden.cpu().numpy(), p.logits.cpu().numpy(), v.cpu().numpy()


    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1-done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def predict_w_value_saliency(self, obs, hidden_state, done):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
        mask = torch.FloatTensor(1-done).to(device=self.device)
        dist, value, hidden_state = self.policy(obs, hidden_state, mask)
        value.backward(retain_graph=True)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return act.detach().cpu().numpy(), log_prob_act.detach().cpu().numpy(), value.detach().cpu().numpy(), hidden_state.detach().cpu().numpy(), obs.grad.data.detach().cpu().numpy()

    def predict_for_logit_saliency(self, obs, act, all_acts=False):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        dist, value, hidden_state = self.policy(obs, None, None)
        if all_acts:
            dist.logits.mean().backward(retain_graph=True)
        else:
            log_prob_act = dist.log_prob(torch.tensor(act).to(device=self.device))
            log_prob_act.backward(retain_graph=True)

        return obs.grad.data.detach().cpu().numpy()

    def predict_for_rew_saliency(self, obs, done):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        dist, value, hidden_state = self.policy(obs, None, None)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return act.detach().cpu().numpy(), log_prob_act, value, obs

    def optimize_sae(self):
        recon_loss_list, sparse_loss_list, total_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.sae.train()
        for e in range(self.epoch):
            recurrent = False
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_batch, act_batch, logit_batch, value_batch = sample

                reconstructed, encoded = self.sae(hidden_batch)
                recon_loss = ((reconstructed - hidden_batch)**2).mean()
                sparse_loss = self.sae.kl_divergence(encoded)

                loss = recon_loss + self.sparse_coef * sparse_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                recon_loss_list.append(recon_loss.item())
                sparse_loss_list.append(sparse_loss.item())
                total_loss_list.append(loss.item())

        summary = {
            'Loss/total': np.mean(total_loss_list),
            'Loss/recon': np.mean(recon_loss_list),
            'Loss/sparsity': np.mean(sparse_loss_list),
        }
        return summary

    def optimize_linear_model(self):
        value_loss_list, logit_loss_list, total_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        kldiv = torch.nn.KLDivLoss(reduction='batchmean')

        self.sae.train()
        for e in range(self.epoch):
            recurrent = False
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_batch, act_batch, logit_batch, value_batch = sample

                with torch.no_grad():
                    _, encoded = self.sae(hidden_batch)

                logit_hat, value_hat = self.linear_model(encoded)

                value_loss = ((value_hat-value_batch)**2).mean()
                logit_loss = kldiv(logit_hat, logit_batch.softmax(dim=-1))
                loss = logit_loss + value_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.linear_model.parameters(), self.grad_clip_norm)
                    self.l_optimizer.step()
                    self.l_optimizer.zero_grad()
                grad_accumulation_cnt += 1
                value_loss_list.append(value_loss.item())
                logit_loss_list.append(logit_loss.item())
                total_loss_list.append(loss.item())

        summary = {
            'Loss/total_linear': np.mean(total_loss_list),
            'Loss/value': np.mean(value_loss_list),
            'Loss/logit': np.mean(logit_loss_list),
        }
        return summary

    def train(self, num_timesteps):
        self.train_model(self.sae, self.optimizer, self.optimize_sae, num_timesteps, "sae")
        self.train_model(self.linear_model, self.l_optimizer, self.optimize_linear_model, num_timesteps, "linear")
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()

    def train_model(self, model, optimizer, optimize_func, num_timesteps, model_type):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        if self.env_valid is not None:
            obs_v = self.env_valid.reset()

        action_policy = None
        if model_type == 'linear':
            action_policy = True

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            self.collect_rollouts(obs, self.storage, self.env, action_policy)

            #valid
            if self.env_valid is not None:
                self.collect_rollouts(obs_v, self.storage_valid, self.env_valid, action_policy)

            # Optimize policy & value
            summary = optimize_func()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = None
            self.logger.feed(rew_batch, done_batch, rew_batch_v, done_batch_v)
            self.logger.dump(summary)
            if self.anneal_lr:
                self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            # Save the model
            if self.t > ((checkpoint_cnt+1) * save_every):
                print("Saving model.")
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                             self.logger.logdir + f'/{model_type}_{self.t}.pth')
                checkpoint_cnt += 1


    def collect_rollouts(self, obs, storage, env, policy=None):
        for _ in range(self.n_steps):
            act, hidden, logits, value = self.get_hidden_and_acts(obs, policy)
            next_obs, rew, done, info = env.step(act)
            storage.store(obs, hidden, act, rew, done, info, logits, value)
            obs = next_obs
        act, hidden, logits, value = self.get_hidden_and_acts(obs, policy)
        storage.store_last(obs, hidden, logits, value)
