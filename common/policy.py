import re
import time

import einops
import gymnasium
import numpy as np
from torch.backends.cudnn import deterministic

from .misc_util import orthogonal_init
from .model import GRU, GraphTransitionModel, MLPModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import sympy as sy


class CategoricalPolicy(nn.Module):
    def __init__(self,
                 embedder,
                 recurrent,
                 action_size,
                 has_vq=False,
                 continuous_actions=False,
                 logsumexp_logits_is_v=False
                 ):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder
        self.has_vq = has_vq
        self.continuous_actions = continuous_actions
        self.action_size = action_size
        self.logsumexp_logits_is_v = logsumexp_logits_is_v
        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)
        # sigmoid(4.6) = 0.99
        self.learned_gamma = nn.Parameter(torch.tensor(4.6, requires_grad=True))
        # exp(-1.6) = 0.2
        self.log_alpha = nn.Parameter(torch.tensor(-1.6, requires_grad=True))
        self.target_entropy = np.log(action_size) if not continuous_actions else -action_size

        self.recurrent = recurrent
        if self.recurrent:
            self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)

    def gamma(self):
        return self.learned_gamma.sigmoid()

    def alpha(self):
        return self.log_alpha.exp()

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
        p = self.distribution(logits)
        if self.logsumexp_logits_is_v:
            v = logits.logsumexp(-1)
        else:
            v = self.fc_value(hidden).reshape(-1)
        return p, v

    def distribution(self, logits):
        if self.continuous_actions:
            return diag_gaussian_dist(logits, act_scale=None, simple=True)
        log_probs = F.log_softmax(logits, dim=1)
        return Categorical(logits=log_probs)


class ICMPolicy(nn.Module):
    def __init__(self,
                 embedder,
                 recurrent,
                 action_size,
                 has_vq=False,
                 continuous_actions=False,
                 logsumexp_logits_is_v=False,
                 # deterministic=True,
                 ):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(ICMPolicy, self).__init__()
        # self.deterministic = deterministic
        self.logsumexp_logits_is_v = logsumexp_logits_is_v
        self.embedder = embedder
        self.has_vq = has_vq
        self.continuous_actions = continuous_actions
        self.action_size = action_size
        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)
        # det_factor = 1 if self.deterministic else 2
        self.transition = MLPModel(in_channels=self.embedder.output_dim + action_size,
                                   depth=4,
                                   mid_weight=64,
                                   latent_size=self.embedder.output_dim * 2)
        # self.transition = orthogonal_init(
        #     nn.Linear(self.embedder.output_dim + action_size, self.embedder.output_dim * 2), gain=1.0)
        # sigmoid(4.6) = 0.99
        self.learned_gamma = nn.Parameter(torch.tensor(4.6, requires_grad=True))
        # exp(-1.6) = 0.2
        self.log_alpha = nn.Parameter(torch.tensor(-1.6, requires_grad=True))
        self.target_entropy = np.log(action_size) if not continuous_actions else -action_size

        self.recurrent = recurrent
        if self.recurrent:
            self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)

    def gamma(self):
        return self.learned_gamma.sigmoid()

    def alpha(self):
        return self.log_alpha.exp()

    def is_recurrent(self):
        return self.recurrent

    def next_state(self, h, a):
        a_ones = torch.nn.functional.one_hot(a.to(torch.int64)).float()
        x = torch.concat((h, a_ones), dim=-1)

        y = self.transition(x)
        y_reshaped = y.view(*y.shape[:-1], self.embedder.output_dim, 2)
        return diag_gaussian_dist(y_reshaped, act_scale=None, simple=True)

    def forward(self, x):
        hidden = self.embedder(x)
        p, v = self.hidden_to_output(hidden)
        return p, v, hidden

    def hidden_to_output(self, hidden):
        logits = self.fc_policy(hidden)
        p = self.distribution(logits)
        if self.logsumexp_logits_is_v:
            v = logits.logsumexp(dim=-1)
        else:
            v = self.fc_value(hidden).squeeze()
        return p, v

    def distribution(self, logits):
        if self.continuous_actions:
            return diag_gaussian_dist(logits, act_scale=None, simple=True)
        log_probs = F.log_softmax(logits, dim=1)
        return Categorical(logits=log_probs)


class GraphPolicy(nn.Module):
    def __init__(self, graph, embedder=None, continuous_actions=False, act_space=None, device=None,
                 simple_scaling=True):
        super(GraphPolicy, self).__init__()
        self.device = device
        self.continuous_actions = continuous_actions
        self.act_scale = None
        self.act_space = act_space
        if isinstance(self.act_space, gymnasium.spaces.Box):
            self.act_scale = torch.FloatTensor((act_space.high - act_space.low) / 2).to(device=self.device)
            self.act_shape = act_space.shape[-1]
        else:
            self.act_shape = act_space.n
        self.embedder = embedder
        self.graph = graph
        self.has_vq = False
        self.recurrent = False
        self.simple_scaling = simple_scaling
        self.no_var = False

    def set_no_var(self, no_var=True):
        self.no_var = no_var
        self.graph.set_no_var(no_var)

    def is_recurrent(self):
        return self.recurrent

    def set_mode(self, mode):
        try:
            self.graph.set_mode(mode)
        except Exception as e:
            print(f"\nTrying to set_mode on non-espl model type:{type(self.graph)}\n")
            raise e

    def set_temp_target_ratio(self, timesteps):
        try:
            self.graph.set_temp_target_ratio(timesteps)
        except Exception as e:
            print(f"\nTrying to set_temp_target_ratio on non-espl model type:{type(self.graph)}\n")
            raise e

    def forward(self, x, hx=None, masks=None):
        if self.embedder is not None:
            x = self.embedder(x)
        logits, value = self.graph(x)
        # print([x for x in self.graph.messenger.model.children()][0].weight)
        p = self.distribution(logits)
        return p, value, hx

    def distribution(self, logits):
        if self.continuous_actions:
            return diag_gaussian_dist(logits, self.act_scale, self.simple_scaling, self.no_var)
        log_probs = F.log_softmax(logits, dim=1)
        return Categorical(logits=log_probs)

    def forward_fine_tune(self, x):
        if self.embedder is not None:
            x = self.embedder(x)
        logits, a_out, m_out = self.graph.forward_fine_tune(x)
        p = self.distribution(logits)
        return p, a_out, m_out


def min_var_gaussian(logits, act_scale, simple=True):
    if simple:
        mean_actions = logits
    else:
        mean_actions = torch.tanh(logits)
        if act_scale is not None:
            # scale actions to correct range:
            mean_actions = mean_actions * act_scale

    min_real = torch.finfo(mean_actions.dtype).tiny
    action_std = torch.full_like(mean_actions, min_real ** 0.5)

    p = Normal(mean_actions, action_std)
    return p


def diag_gaussian_dist(logits, act_scale, simple=True, no_var=False):
    if no_var:
        return min_var_gaussian(logits, act_scale, simple)
    if simple:
        mean_actions = logits[..., 0]
        logvar = logits[..., -1]
        # TODO: clamp logvar?
        # action_std = torch.ones_like(mean_actions) * logvar.exp()
        action_std = torch.sqrt(logvar.exp())
    else:
        mean_actions = torch.tanh(logits[..., 0])
        if act_scale is not None:
            # scale actions to correct range:
            mean_actions = mean_actions * act_scale
        action_std = F.softplus(logits[..., -1])

    min_real = torch.finfo(action_std.dtype).tiny
    action_std = torch.clamp(action_std, min=min_real ** 0.5)

    p = Normal(mean_actions, action_std)
    return p


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
        # self.fc_reward = orthogonal_init(nn.Linear(self.embedder.output_dim + 1, 1), gain=1.0)
        # self.fc_continuation = orthogonal_init(nn.Linear(self.embedder.output_dim + 1, 1), gain=1.0)

        self.cont_rew = MLPModel(self.embedder.input_size + 1,
                                 self.embedder.depth,
                                 self.embedder.mid_weight,
                                 2)

        self.value = nn.Sequential(self.embedder, self.fc_value)
        # self.reward = nn.Sequential(self.embedder, self.fc_reward)

        self.transition_model = transition_model

        self.done_model = None
        self.r_model = None

    def dr(self, sa):
        if self.done_model is None or self.r_model is None:
            dr = self.cont_rew(sa).squeeze()
            d, r = dr.split(1, dim=-1)
            return nn.Sigmoid()(d.squeeze()), r.squeeze()
        d = self.done_model(sa)
        r = self.r_model(sa)
        return nn.Sigmoid()(d), r

    def is_recurrent(self):
        return False

    def actions_like(self, s, i):
        return torch.full((s.shape[:-1]), i).to(device=self.device)

    def all_actions_like(self, s):
        a = torch.FloatTensor([i for i in range(self.action_size)])
        return a.tile((*s.shape[:-1], 1)).to(device=self.device)

    def forward(self, x):
        # Reward is dependent on action, should be bucketed up with continuation flag
        # v, reward = self.value_reward(x)
        v = self.value(x)
        rews = []
        cont = []
        s = x
        for _ in range(self.n_rollouts):
            d, r = self.all_dones_rewards(s)
            cont.append(d)
            rews.append(r)
            next_states = [self.transition_model(s, self.actions_like(s, i)).unsqueeze(1) for i in
                           range(self.action_size)]
            s = torch.concat(next_states, dim=1)

        # adding discounted rewards
        cum = torch.zeros_like(rews[0])
        for r in rews[:-1]:
            cum = ((cum + r) * self.gamma).unsqueeze(-2).tile([self.action_size, 1])

        vs = self.value(s).squeeze()
        vs *= self.gamma ** self.n_rollouts
        vs += cum.squeeze()
        for i in range(self.n_rollouts - 1):
            vs[cont[-(i + 1)].round(decimals=0) == 1] = 0
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
        return p, v.squeeze()  # , reward.squeeze()

    def all_dones_rewards(self, s):
        sa = self.states_with_all_actions(s)
        dones, rew = self.dr(sa)
        return dones, rew

    def dones_rewards(self, s, a):
        sa = torch.concat([s, a.unsqueeze(-1)], dim=-1)
        return self.dr(sa)

    def states_with_all_actions(self, s):
        s1 = s.unsqueeze(-2).tile([self.action_size, 1])
        a = self.all_actions_like(s)
        sa = torch.concat([s1, a.unsqueeze(-1)], dim=-1)
        return sa

    def vectorized_attempt(self, x):
        s1 = x
        for _ in range(self.n_rollouts):
            s1 = self.transition_model(self.expand_for_actions(s1), self.all_actions_like(s1))

    def expand_for_actions(self, s1):
        k = len(s1.shape)
        shp = [self.action_size] + [1 for _ in range(k - 1)]
        return s1.unsqueeze(1).tile(shp)
        # shp = [1 for _ in range(k + 1)]
        # shp[1] = self.action_size
        # return s1.unsqueeze(1).repeat(shp)


class PixelTransPolicy(nn.Module):
    def __init__(self,
                 encoder,
                 sub_policy,
                 ):
        super(PixelTransPolicy, self).__init__()
        self.encoder = orthogonal_init(encoder)
        self.sub_policy = sub_policy

    def forward(self, x):
        h = self.encoder(x)
        f = torch.flatten(h, start_dim=-2).permute(0, 2, 1)
        # TODO: figure this out.
        #  maybe we project down dimensions or maybe we use graph networks for value func. etc.
        return self.sub_policy(f)


class GraphTransitionPolicy(nn.Module):
    def __init__(self,
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
        super(GraphTransitionPolicy, self).__init__()
        assert n_rollouts > 0, "n_rollouts must be > 0"
        self.n_rollouts = n_rollouts
        self.temperature = temperature
        self.gamma = gamma
        self.has_vq = False
        self.action_size = action_size
        self.transition_model = transition_model

    def is_recurrent(self):
        return False

    def actions_like(self, s, i):
        return torch.full((s.shape[:-1]), i).to(device=self.device)

    def all_actions_like(self, s):
        a = torch.FloatTensor([i for i in range(self.action_size)])
        return a.tile((*s.shape[:-1], 1)).to(device=self.device)

    def forward(self, x):
        cont, rews = [], []
        # order: reward, continuation, value
        s = self.add_zeros(x)
        for r in range(self.n_rollouts):
            s[..., -3:] = 0
            next_states = [self.transition_model(s, self.actions_like(s, i)).unsqueeze(1) for i in
                           range(self.action_size)]
            s = torch.concat(next_states, dim=1)
            rews.append(s[..., -3].clone())
            cont.append(nn.Sigmoid()(s[..., -2].clone()))
            if r == 0:
                v = s[..., -1].clone()

        # rews = s[..., -3]
        # cont = s[..., -2]
        vals = s[..., -1]

        # adding discounted rewards
        cum = torch.zeros_like(rews[0])
        for r in rews[:-1]:
            cum = ((cum + r) * self.gamma).unsqueeze(-2).tile([self.action_size, 1])

        vals *= self.gamma ** self.n_rollouts
        vals += cum.squeeze()
        for i in range(self.n_rollouts - 1):
            vals[cont[-(i + 1)].round(decimals=0) == 1] = 0
            vals = ((vals / self.temperature).softmax(-1) * vals).sum(-1)

        log_probs = F.log_softmax(vals / self.temperature, dim=1)
        p = Categorical(logits=log_probs)
        return p, v.squeeze()  # , reward.squeeze()

    def add_zeros(self, x):
        s = torch.concat((x, torch.zeros((*x.shape[:-1], 3)).to(device=self.device)), dim=-1)
        return s

    def transition(self, x, a):
        s = self.add_zeros(x)
        h = self.transition_model(s, a)
        return h[..., :-3], h[..., -3], nn.Sigmoid()(h[..., -2]), h[..., -1]

    def all_dones_rewards(self, s):
        sa = self.states_with_all_actions(s)
        dones, rew = self.dr(sa)
        return dones, rew

    def dones_rewards(self, s, a):
        sa = torch.concat([s, a.unsqueeze(-1)], dim=-1)
        return self.dr(sa)

    def states_with_all_actions(self, s):
        s1 = s.unsqueeze(-2).tile([self.action_size, 1])
        a = self.all_actions_like(s)
        sa = torch.concat([s1, a.unsqueeze(-1)], dim=-1)
        return sa

    def vectorized_attempt(self, x):
        s1 = x
        for _ in range(self.n_rollouts):
            s1 = self.transition_model(self.expand_for_actions(s1), self.all_actions_like(s1))

    def expand_for_actions(self, s1):
        k = len(s1.shape)
        shp = [self.action_size] + [1 for _ in range(k - 1)]
        return s1.unsqueeze(1).tile(shp)


class DoubleTransitionPolicy(nn.Module):
    def __init__(self,
                 value_model,
                 transition_model,
                 action_size,
                 n_rollouts,
                 temperature,
                 gamma,
                 done_func,
                 rew_func,
                 ):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(DoubleTransitionPolicy, self).__init__()
        assert n_rollouts > 0, "n_rollouts must be > 0"
        self.n_rollouts = n_rollouts
        self.temperature = temperature
        self.gamma = gamma
        self.has_vq = False
        self.action_size = action_size

        self.done_func = done_func
        self.rew_func = rew_func
        self.value_model = value_model
        self.transition_model = transition_model

    def dr(self, state):
        s = state.detach().cpu().numpy()
        d = self.done_func(s)
        r = self.rew_func(s).squeeze()
        d = torch.FloatTensor(d).to(device=self.device)
        r = torch.FloatTensor(r).to(device=self.device)
        return d, r

    def is_recurrent(self):
        return False

    def actions_like(self, s, i):
        return torch.full((s.shape[:-1]), i).to(device=self.device)

    def all_actions_like(self, s):
        a = torch.FloatTensor([i for i in range(self.action_size)])
        return a.tile((*s.shape[:-1], 1)).to(device=self.device)

    def forward(self, x):
        v = self.value_model(x)
        rews = []
        cont = []
        s = x
        for _ in range(self.n_rollouts):
            next_states = [self.transition_model(s, self.actions_like(s, i)).unsqueeze(1) for i in
                           range(self.action_size)]
            s = torch.concat(next_states, dim=1)
            d, r = self.dr(s)
            cont.append(d)
            rews.append(r)

        # adding discounted rewards
        cum = torch.zeros((rews[0].shape)).to(device=self.device)
        for r in rews[:-1]:
            cum = ((cum + r) * self.gamma).unsqueeze(-2).tile([self.action_size, 1])
        # cum = cum.to(device=self.device)

        vs = self.value_model(s)
        vs *= self.gamma ** self.n_rollouts
        vs += cum.squeeze()
        for i in range(self.n_rollouts - 1):
            vs[cont[-(i + 1)] == 1] = 0
            vs = ((vs / self.temperature).softmax(-1) * vs).sum(-1)
            # vs = vs.max(-1)[0]

        log_probs = F.log_softmax(vs / self.temperature, dim=1)
        if log_probs.isnan().any():
            print("check")
        p = Categorical(logits=log_probs)
        if p.probs.isnan().any():
            print("check")
        return p, v

    def all_dones_rewards(self, s):
        sa = self.states_with_all_actions(s)
        dones, rew = self.dr(sa)
        return dones, rew

    def states_with_all_actions(self, s):
        s1 = s.unsqueeze(-2).tile([self.action_size, 1])
        a = self.all_actions_like(s)
        sa = torch.concat([s1, a.unsqueeze(-1)], dim=-1)
        return sa

    def vectorized_attempt(self, x):
        s1 = x
        for _ in range(self.n_rollouts):
            s1 = self.transition_model(self.expand_for_actions(s1), self.all_actions_like(s1))

    def expand_for_actions(self, s1):
        k = len(s1.shape)
        shp = [self.action_size] + [1 for _ in range(k - 1)]
        return s1.unsqueeze(1).tile(shp)


class PureTransitionPolicy(nn.Module):
    def __init__(self,
                 transition_model,
                 action_size,
                 n_rollouts,
                 temperature,
                 gamma,
                 done_func,
                 rew_func,
                 ):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(PureTransitionPolicy, self).__init__()
        assert n_rollouts > 0, "n_rollouts must be > 0"
        self.n_rollouts = n_rollouts
        self.temperature = temperature
        self.gamma = gamma
        self.has_vq = False
        self.action_size = action_size

        self.done_func = done_func
        self.rew_func = rew_func
        self.transition_model = transition_model

    def dr(self, state):
        s = state.detach().cpu().numpy()
        d = self.done_func(s)
        r = self.rew_func(s).squeeze()
        d = torch.FloatTensor(d).to(device=self.device)
        r = torch.FloatTensor(r).to(device=self.device)
        return d, r

    def is_recurrent(self):
        return False

    def actions_like(self, s, i):
        return torch.full((s.shape[:-1]), i).to(device=self.device)

    def all_actions_like(self, s):
        a = torch.FloatTensor([i for i in range(self.action_size)])
        return a.tile((*s.shape[:-1], 1)).to(device=self.device)

    def forward(self, x):
        rews = []
        cont = []
        s = x
        for _ in range(self.n_rollouts):
            next_states = [self.transition_model(s, self.actions_like(s, i)).unsqueeze(1) for i in
                           range(self.action_size)]
            s = torch.concat(next_states, dim=1)
            d, r = self.dr(s)
            cont.append(d)
            rews.append(r)

        # adding discounted rewards
        cum = torch.zeros((rews[0].shape)).to(device=self.device)
        for r in rews[:-1]:
            cum = ((cum + r) * self.gamma).unsqueeze(-2).tile([self.action_size, 1])
        # cum = cum.to(device=self.device)

        vs = torch.zeros(s.shape[:-1]).to(device=self.device)
        vs += cum.squeeze()
        for i in range(self.n_rollouts - 1):
            vs[cont[-(i + 1)] == 1] = 0
            vs = ((vs / self.temperature).softmax(-1) * vs).sum(-1)
            # vs = vs.max(-1)[0]

        log_probs = F.log_softmax(vs / self.temperature, dim=-1)
        # if log_probs.isnan().any():
        #     print("check")
        p = Categorical(logits=log_probs)
        if (p.probs != 0.5).any():
            # if p.probs.isnan().any():
            print("check")
        return p

    def all_dones_rewards(self, s):
        sa = self.states_with_all_actions(s)
        dones, rew = self.dr(sa)
        return dones, rew

    def states_with_all_actions(self, s):
        s1 = s.unsqueeze(-2).tile([self.action_size, 1])
        a = self.all_actions_like(s)
        sa = torch.concat([s1, a.unsqueeze(-1)], dim=-1)
        return sa

    def vectorized_attempt(self, x):
        s1 = x
        for _ in range(self.n_rollouts):
            s1 = self.transition_model(self.expand_for_actions(s1), self.all_actions_like(s1))

    def expand_for_actions(self, s1):
        k = len(s1.shape)
        shp = [self.action_size] + [1 for _ in range(k - 1)]
        return s1.unsqueeze(1).tile(shp)


class GoalSeekerPolicy(nn.Module):
    def __init__(self,
                 embedder,
                 action_size,
                 model_constructor=lambda x, y: orthogonal_init(nn.Linear(x, y), gain=0.01),
                 predict_continuous=True,
                 n_action_samples=5,
                 temp=1.,
                 continuous_actions=False,
                 greedy_distance_minimization=False,
                 ):
        super(GoalSeekerPolicy, self).__init__()

        self.temp = temp
        self.greedy_distance_minimization = greedy_distance_minimization
        self.embedder = embedder
        self.action_size = action_size
        self.h_size = self.embedder.output_dim
        self.n_action_samples = n_action_samples
        self.predict_continuous = predict_continuous
        self.continuous_actions = continuous_actions
        scale_out = 1
        if self.predict_continuous:
            scale_out = 2
        action_scale = 1
        if self.continuous_actions:
            action_scale = 2

        self.action_model = model_constructor(self.h_size * 2, action_size * action_scale)
        self.forward_model = model_constructor(self.h_size + action_size, self.h_size * scale_out)
        # self.backward_model = model_constructor(self.h_size + action_size, self.h_size * scale_out)

        # self.reward_model = model_constructor(self.h_size, 1 * scale_out)

        self.goal_model = model_constructor(self.h_size, self.h_size * scale_out)
        self.critic = model_constructor(self.h_size, 1)
        self.actor = model_constructor(self.h_size, action_size * action_scale)
        self.traj_model = model_constructor(self.h_size * 2, 1)
        # self.reverse_actor = model_constructor(self.h_size, action_size * scale_out)

    def distribution(self, logits, categorical=False):
        if not categorical:
            n = logits.shape[-1]//2
            assert n*2 == logits.shape[-1], "last logits dim must be even"
            mean = logits[..., :n]
            logvar = logits[..., n:]
            # TODO: clamp logvar?
            std = torch.sqrt(logvar.exp())
            min_real = torch.finfo(std.dtype).tiny
            std = torch.clamp(std, min=min_real ** 0.5)
            return Normal(mean, std)
        logits = F.log_softmax(logits, dim=-1)
        return Categorical(logits=logits)

    def traj_distance(self, hidden, goal_hidden):
        goal_hidden_expanded = self.expand_for_concat(smaller=goal_hidden, larger=hidden, n_diff_dims=0)
        hgh = torch.concat((hidden, goal_hidden_expanded), dim=-1)
        return self.traj_model(hgh)

    def predict_action_hidden(self, hidden, next_hidden):
        hnh = torch.concat((hidden, next_hidden), dim=-1)
        out = self.action_model(hnh)
        return self.distribution(out)

    def predict_next_hidden(self, hidden, action):
        a_hot = action
        if not self.continuous_actions:
            a_hot = torch.nn.functional.one_hot(action).float()
        h = self.expand_for_concat(hidden, a_hot, 1)
        ha = torch.concat((h, a_hot), dim=-1)
        out = self.forward_model(ha)
        return self.distribution(out)

    def expand_for_concat(self, smaller, larger, n_diff_dims = 1):
        shp_diff = len(larger.shape) - len(smaller.shape)
        if shp_diff > 0:
            a_batches = larger.shape[shp_diff:-n_diff_dims]
            h_batches = smaller.shape[:-n_diff_dims]
            assert a_batches == h_batches, f"batch sizes must be the same: {a_batches}, {h_batches}"
            shp = [1 for _ in larger.shape]
            shp[:shp_diff] = larger.shape[:shp_diff]
            return smaller.unsqueeze(0).tile(shp)
        return smaller

    def predict_prev_hidden(self, next_hidden, action):
        nha = torch.concat((next_hidden, action), dim=-1)
        out = self.backward_model(nha)
        return self.distribution(out)

    def predict_goal_hidden(self, hidden):
        out = self.goal_model(hidden)
        return self.distribution(out)

    def predict_reward(self, hidden):
        out = self.reward_model(hidden)
        return self.distribution(out)

    def predict_action_from_hidden(self, hidden):
        out = self.actor(hidden)
        return self.distribution(out)

    def predict_reverse_action_from_hidden(self, hidden):
        out = self.reverse_actor(hidden)
        return self.distribution(out)

    def predict_action(self, state, next_state):
        return self.predict_action_hidden(self.embedder(state), self.embedder(next_state))

    def predict_next(self, state, action):
        return self.predict_next_hidden(self.embedder(state), action)

    def predict_prev(self, next_state, action):
        return self.predict_prev_hidden(self.embedder(next_state), action)

    def predict_goal(self, state):
        return self.predict_goal_hidden(self.embedder(state))

    def forward(self, state):
        hidden = self.embedder(state)
        p = self.actor_dist(hidden)
        v = self.critic(hidden)
        return p, v

    def reduce_temp(self, target_temp=0.0001, decay_rate=0.001):
        self.temp += (target_temp-self.temp)*decay_rate

    def plan(self, state, goal_override=None):
        hidden = self.embedder(state)
        if goal_override is None:
            goal_dist = self.predict_goal_hidden(hidden)
        else:
            goal_dist = goal_override

        p = self.actor_dist(hidden)
        acts = self.sample_n(p, self.n_action_samples)
        next_hid_dist = self.predict_next_hidden(hidden, acts)
        # TODO: take into account variance?
        distance = self.traj_distance(next_hid_dist.loc, goal_dist.loc)

        if self.greedy_distance_minimization:
            best_acts = distance.argmin(dim=0).squeeze()
        else:
            # or sample proportional to distance - add a temp in?
            distance_distr = self.distribution(distance.squeeze().T * self.temp,categorical=True)
            best_acts = distance_distr.sample()

        idx = tuple(i for i in range(len(best_acts)))
        selected_action = acts[best_acts, idx]

        # train actor to select based on distance
        return selected_action

    def actor_dist(self, hidden):
        a_out = self.actor(hidden)
        return self.distribution(a_out, categorical=not self.continuous_actions)

    def plan_bhatt(self, state):
        # Bhattacharyya bound
        hidden = self.embedder(state)
        goal_hidden = self.goal_model(hidden)

        base = hidden
        goal = goal_hidden.loc
        base_actions = []
        goal_actions = []

        n_samples = 3

        for _ in range(self.rollouts):
            p = self.predict_action_from_hidden(base)
            a = self.sample_n(p, n_samples)
            next_hidden = self.predict_next_hidden(base, a)

            rp = self.predict_reverse_action_from_hidden(goal)
            ra = self.sample_n(rp, n_samples)
            pen_hidden = self.predict_prev_hidden(goal, ra)

            # Bhattacharyya bound
            dist = self.bhattacharyya_bound(next_hidden, pen_hidden)

            # sample instead of argmin?
            idx_base, idx_goal = dist.argmin()
            base = next_hidden[idx_base].loc
            base_actions.append(a[idx_goal])

            goal = pen_hidden[idx_goal].loc
            goal_actions.append(ra[idx_goal])

    def sample_n(self, dist, n_samples):
        return torch.stack([dist.sample() for _ in range(n_samples)])

    def bhattacharyya_bound(self, d1, d2):

        print(d1)


def solve(value_dict, eq_pattern):
    equation = sy.Eq(*sy.S(f"{eq_pattern}, ti")).subs(value_dict)
    solution = sy.solve(equation, manual=True)
    return solution


def auto_concat(inputs, pattern):
    assert isinstance(inputs, (list, tuple)), "inputs must be a list or tuple"
    n = len(inputs)
    in_pat, out_pat = pattern.split("->")
    shapes = in_pat.split(",")
    assert len(shapes) == n, "pattern must have a shape for each input"

    input_shapes = [list(b.shape) for b in inputs]
    input_dims = [[d for d in s.split(' ') if d != ''] for s in shapes]

    vals = {}
    for s, d in zip(input_shapes, input_dims):
        assert len(s) == len(d), "incorrect dimensions in pattern, ensure pattern matches inputs shapes"
        for k, v in zip(d, s):
            assert not (k in vals.keys() and vals[k] != v), f"multiple definitions for {k}: {vals[k]} and {v}"
            vals[k] = v

    out_dims = [a for a in out_pat.split(' ') if a != '']
    out_shape = []
    for d in out_dims:
        if d in vals.keys():
            v = vals[d]
        else:
            v = solve(vals, d)[0]
        out_shape.append(v)

    concat_dim = np.argwhere([bool(re.search("\+",d)) for d in out_dims]).squeeze().item()

    concat_eq = out_dims[concat_dim]

    # ndims = max([len(id) for id in input_dims])
    ndims = len(out_dims)
    filled_dims = [id + ['' for _ in range(ndims - len(id))] for id in input_dims]
    fd = np.array(filled_dims)


    for i, tens in enumerate(inputs):
        shp = []
        for d in input_dims[i]:
            np.array(input_dims).flatten().count()


    shapes

