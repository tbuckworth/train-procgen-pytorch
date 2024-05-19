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
                 action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(TransitionPolicy, self).__init__()
        self.n_rollouts = 3
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

    def forward(self, x):
        v = self.value(x)

        s = x
        for _ in range(self.n_rollouts):
            next_states = [self.transition_model(s, self.actions_like(s, i)).unsqueeze(1) for i in range(self.action_size)]
            s = torch.concat(next_states, dim=1)

        vs = self.value(s).squeeze()
        for _ in range(self.n_rollouts-1):
            vs = vs.max(-1)[0]
        # if self.greedy:
        #     actions = vs.argmax(-1)
        # else:
        log_probs = F.log_softmax(vs, dim=1)
        p = Categorical(logits=log_probs)
        return p, v.squeeze()

    # def value(self, x):
    #     hidden = self.embedder(x)
    #     return self.fc_value(hidden)#.reshape(-1)

