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
                 ):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(TransitionPolicy, self).__init__()
        assert n_rollouts > 0, "n_rollouts must be > 0"
        self.n_rollouts = n_rollouts
        self.temperature = temperature
        self.embedder = embedder
        self.has_vq = False
        self.action_size = action_size
        # small scale weight-initialization in policy enhances the stability
        # self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

        self.value = nn.Sequential(self.embedder, self.fc_value)
        self.transition_model = transition_model

    def is_recurrent(self):
        return False

    def actions_like(self, s, i):
        return torch.full((s.shape[:-1]), i).to(device=self.device)

    def all_actions_like(self, s):
        a = torch.FloatTensor([i for i in range(self.action_size)])
        return a.tile((*s.shape[:-1], 1)).to(device=self.device)

    def forward(self, x):
        v = self.value(x)

        s = x
        for _ in range(self.n_rollouts):
            next_states = [self.transition_model(s, self.actions_like(s, i)).unsqueeze(1) for i in
                           range(self.action_size)]
            s = torch.concat(next_states, dim=1)

        vs = self.value(s).squeeze()
        for _ in range(self.n_rollouts - 1):
            # vs = vs.max(-1)[0]
            vs = ((vs / self.temperature).softmax(-1) * vs).sum(-1)
        log_probs = F.log_softmax(vs, dim=1)
        p = Categorical(logits=log_probs)
        return p, v.squeeze()

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
