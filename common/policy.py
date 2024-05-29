import time

from .misc_util import orthogonal_init
from .model import GRU, GraphTransitionModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class CategoricalPolicy(nn.Module):
    def __init__(self,
                 embedder,
                 recurrent,
                 action_size,
                 has_vq=False):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder
        self.has_vq = has_vq
        self.action_size = action_size
        # small scale weight-initialization in policy enhances the stability        
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

        self.recurrent = recurrent
        if self.recurrent:
            self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)

    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hx, masks):
        if self.has_vq:
            hidden, commit_loss = self.embedder(x)
        else:
            hidden = self.embedder(x)
        if self.recurrent:
            hidden, hx = self.gru(hidden, hx, masks)
        p, v = self.hidden_to_output(hidden)

        if self.has_vq:
            return p, v, hx, commit_loss
        return p, v, hx

    def hidden_to_output(self, hidden):
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v


class TransitionPolicy(nn.Module):
    def __init__(self,
                 embedder,
                 transition_model,
                 action_size,
                 n_rollouts,
                 temperature,
                 gamma,
                 ):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(TransitionPolicy, self).__init__()
        assert n_rollouts > 0, "n_rollouts must be > 0"
        self.n_rollouts = n_rollouts
        self.temperature = temperature
        self.gamma = gamma
        self.embedder = embedder
        self.has_vq = False
        self.action_size = action_size
        # small scale weight-initialization in policy enhances the stability
        # self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)
        self.fc_reward = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)
        self.fc_continuation = orthogonal_init(nn.Linear(self.embedder.output_dim + 1, 1), gain=1.0)

        self.value = nn.Sequential(self.embedder, self.fc_value)
        self.reward = nn.Sequential(self.embedder, self.fc_reward)

        self.transition_model = transition_model

    def dones(self, s, a):
        h = self.embedder(s)
        d = self.fc_continuation(torch.concat([h, a.unsqueeze(-1)], dim=-1)).squeeze()
        return nn.Sigmoid()(d)

    def value_reward(self, x):
        h = self.embedder(x)
        v = self.fc_value(h).reshape(-1)
        r = self.fc_reward(h).reshape(-1)
        return v, r

    def is_recurrent(self):
        return False

    def actions_like(self, s, i):
        return torch.full((s.shape[:-1]), i).to(device=self.device)

    def all_actions_like(self, s):
        a = torch.FloatTensor([i for i in range(self.action_size)])
        return a.tile((*s.shape[:-1], 1)).to(device=self.device)

    def forward(self, x):
        # Reward is dependent on action, should be bucketed up with continuation flag
        v, reward = self.value_reward(x)
        rews = []
        cont = []
        s = x
        for _ in range(self.n_rollouts):
            cont.append(self.all_dones(s))
            next_states = [self.transition_model(s, self.actions_like(s, i)).unsqueeze(1) for i in
                           range(self.action_size)]
            s = torch.concat(next_states, dim=1)
            rews.append(self.reward(s))

        # adding discounted rewards
        cum = torch.zeros_like(rews[0])
        for r in rews[:-1]:
            # cum = cum.unsqueeze(-2).tile([self.action_size, 1])
            cum = ((cum + r) * self.gamma).unsqueeze(-2).tile([self.action_size, 1])

        vs = self.value(s).squeeze()
        vs *= self.gamma**self.n_rollouts
        vs += cum.squeeze()
        for i in range(self.n_rollouts - 1):
            vs[cont[-(i+1)].round(decimals=0) == 1] = 0
            vs = ((vs / self.temperature).softmax(-1) * vs).sum(-1)
            # vs = vs.max(-1)[0]

        log_probs = F.log_softmax(vs / self.temperature, dim=1)
        p = Categorical(logits=log_probs)
        # if abs(x[0][2]) >= 0.18 or abs(x[0][0]) >= 2.35:
        #     print(x[0])
        #     print(s[0])
        #     print(vs[0])
        #     print(p.probs[0])
        #     print("pause")
        return p, v.squeeze(), reward.squeeze()

    def all_dones(self, s):
        s1 = s.unsqueeze(-2).tile([self.action_size, 1])
        a = self.all_actions_like(s)
        dones = self.dones(s1, a)
        return dones

    def vectorized_attempt(self, x):
        s1 = x
        for _ in range(self.n_rollouts):
            s1 = self.transition_model(self.expand_for_actions(s1), self.all_actions_like(s1))

    def expand_for_actions(self, s1):
        k = len(s1.shape)
        shp = [self.action_size] + [1 for _ in range(k - 1)]
        return s1.unsqueeze(1).tile(shp)
        shp = [1 for _ in range(k + 1)]
        shp[1] = self.action_size
        return s1.unsqueeze(1).repeat(shp)

    # def old_forward(self, obs):
