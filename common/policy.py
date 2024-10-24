import time

import gymnasium

from .misc_util import orthogonal_init
from .model import GRU, GraphTransitionModel, MLPModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class CategoricalPolicy(nn.Module):
    def __init__(self,
                 embedder,
                 recurrent,
                 action_size,
                 has_vq=False,
                 continuous_actions=False,
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
        # small scale weight-initialization in policy enhances the stability        
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)
        # sigmoid(4.6) = 0.99
        self.learned_gamma = nn.Parameter(torch.tensor(4.6,requires_grad=True))

        self.recurrent = recurrent
        if self.recurrent:
            self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)

    def gamma(self):
        return self.learned_gamma.sigmoid()

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
        v = self.fc_value(hidden).reshape(-1)
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
        action_std = torch.ones_like(mean_actions) * logvar.exp()
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
