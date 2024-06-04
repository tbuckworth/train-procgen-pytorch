import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from helper_local import sigmoid, inverse_sigmoid, softmax, sample_numpy_probs, \
    match_to_nearest

def greater(a, b):
    return (a > b).astype(np.int32)

class CustomModel:
    def __init__(self, degrees=12):
        self.theta_threshold_radians = degrees * 2 * np.pi / 360
    def swing(self, obs):
        x0, x1, x2, x3, x4, x5, x6, x7, x8 = obs.T
        return np.sin(greater(np.sin(((np.sign(x3 + (np.sin(x2) / np.sin(x5))) + -0.06463726) + x2) + x2), x0) + 0.04201813)

    def balance(self, obs):
        x0, x1, x2, x3, x4, x5, x6, x7, x8 = obs.T
        return greater(x1 + (x2+2*x3)/x7, -0.165)

    def predict(self, obs):
        x0, x1, x2, x3, x4, x5, x6, x7, x8 = obs.T
        theta_ib = np.bitwise_and(x2 >= -self.theta_threshold_radians,
                                  x2 <= self.theta_threshold_radians)
        predictions = self.swing(obs)
        predictions[theta_ib] = self.balance(obs[theta_ib])
        return predictions


def flatten_batches_to_numpy(arr):
    return arr.reshape(-1, arr.shape[-1]).cpu().numpy()

def invert_list_levels(l):
    return [[sl[i] for sl in l] for i in range(len(l[0]))]

class GraphSymbolicAgent:
    def __init__(self, policy, msg_model=None, up_model=None, v_model=None, r_model=None, done_model=None):
        self.policy = policy
        if msg_model is not None:
            self.policy.transition_model.message_model = msg_model
        if up_model is not None:
            self.policy.transition_model.updater = up_model
        if v_model is not None:
            self.policy.value = v_model
        if r_model is not None:
            self.policy.r_model = r_model
        if done_model is not None:
            self.policy.done_model = done_model

    def forward(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            dist, value, reward = self.policy(obs)
            act = dist.sample()
            return act.cpu().numpy()

    def sample(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            # dist, value, reward = self.policy(obs)
            data_list = [self.msg_in_out(i, obs) for i in range(self.policy.action_size)]
            dl = invert_list_levels(data_list)
            dt = [np.concatenate(l, axis=0) for l in dl]

            sa = self.policy.states_with_all_actions(obs)
            dones, rew = self.policy.dr(sa)
            v = self.policy.value(obs).squeeze()

            return dt[0], dt[1], dt[2], dt[3], sa, dones, rew, v

    def msg_in_out(self, i, obs):
        action = self.policy.actions_like(obs, i)
        n, x = self.policy.transition_model.prep_input(obs)
        msg_in = self.policy.transition_model.vectorize_for_message_pass(action, n, x)
        messages = self.policy.transition_model.messenger(msg_in)
        h, u = self.policy.transition_model.vec_for_update(messages, x)

        m_in = flatten_batches_to_numpy(msg_in)
        m_out = flatten_batches_to_numpy(messages)
        u_in = flatten_batches_to_numpy(h)
        u_out = flatten_batches_to_numpy(u)

        return m_in, m_out, u_in, u_out


class SymbolicAgent:
    def __init__(self, model, policy, stochastic, action_mapping):
        self.model = model
        self.policy = policy
        self.stochastic = stochastic
        self.single_output = self.policy.action_size <= 2
        self.action_mapping = action_mapping
        self.n = len(np.unique(self.action_mapping)) // 2


    def forward(self, observation):
        with (torch.no_grad()):
            h = self.model.predict(observation)
            if self.single_output:
                if self.stochastic:
                    p = sigmoid(h)
                    return np.int32(np.random.random(len(h)) < p)
                return match_to_nearest(h, self.action_mapping)
            return self.pred_to_action(h)

    def sample(self, observation):
        with torch.no_grad():
            x = torch.FloatTensor(observation).to(self.policy.device)
            h = self.policy.embedder(x)
            dist, value = self.policy.hidden_to_output(h)
            y = dist.logits.detach().cpu().numpy()
            if self.single_output:
                # deterministic policy:
                act = y.argmax(axis=1)
                if self.stochastic:
                    # inverse sigmoid enables prediction of single logit:
                    p = dist.probs.detach().cpu().numpy()
                    z = inverse_sigmoid(p)
                    y = z[:, 1]
                    act = dist.sample().cpu().numpy()
                return observation, y, act, value.cpu().numpy()
            act = y.argmax(1)
            if self.stochastic:
                act = dist.sample()
            return observation, y, act, value.cpu().numpy()

    def pred_to_action(self, h):
        if self.stochastic:
            p = softmax(h)
            return sample_numpy_probs(p)

        if self.single_output:
            return h.argmax(axis=1)
        try:
            return match_to_nearest(h, self.action_mapping)
        except Exception as e:
            print(e)
            raise Exception("Deterministic agent requires action_mapping")


class NeuralAgent:
    def __init__(self, policy):
        self.policy = policy

    def forward(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            h = self.policy.embedder(obs)
            dist, value = self.policy.hidden_to_output(h)
            act = dist.sample()
        return act.cpu().numpy()


class DeterministicNeuralAgent:
    def __init__(self, policy):
        self.policy = policy

    def forward(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            h = self.policy.embedder(obs)
            dist, value = self.policy.hidden_to_output(h)
            y = dist.logits.detach().cpu().numpy()
            act = y.argmax(axis=1)
        return act


class RandomAgent:
    def __init__(self, n_actions):
        self.actions = np.arange(n_actions)

    def forward(self, observation):
        return np.random.choice(self.actions, size=len(observation))


class AnalyticModel:
    def forward(self, observation):
        out = np.ones((observation.shape[0],))
        term = 3 * observation[:, 2] + observation[:, 3]
        out[term <= 0] = 0
        return out

    def predict(self, observation):
        return self.forward(observation)


class NeuroSymbolicAgent:
    def __init__(self, model, policy, stochastic, action_mapping):
        self.model = model
        self.policy = policy
        self.stochastic = stochastic
        self.action_mapping = action_mapping
        self.single_output = self.policy.action_size <= 2
        self.n = len(np.unique(self.action_mapping)) // 2

    def forward(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            x = self.policy.embedder.forward_to_pool(obs)
            h = self.model.predict(x)
            return self.pred_to_action(h)

    def pred_to_action(self, h):
        if self.stochastic:
            logits = torch.FloatTensor(h).to(self.policy.device)
            log_probs = F.log_softmax(logits, dim=1)
            p = Categorical(logits=log_probs)
            act = p.sample()
            return act.cpu().numpy()
        return match_to_nearest(h, self.action_mapping)
        # rads = np.round(h / (np.pi / self.n), 0) * (np.pi / self.n)
        # rads[rads < 0] = -1
        # return match(rads, self.action_mapping)

    def sample(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            x = self.policy.embedder.forward_to_pool(obs)
            h = self.policy.embedder.forward_from_pool(x)
            dist, value = self.policy.hidden_to_output(h)
            y = dist.logits.detach().cpu().numpy()
            act = dist.sample().cpu().numpy()
            if not self.stochastic:
                act = y.argmax(axis=1)
        return x.cpu().numpy(), y, act, value.cpu().numpy()

    def sample_latent_output_impala(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            x = self.policy.embedder.forward_to_pool(obs)
            h = self.policy.embedder.forward_from_pool(x)
            dist, value = self.policy.hidden_to_output(h)
            y = dist.logits.detach().cpu().numpy()
            act = dist.sample()
        return x.cpu().numpy(), y, act.cpu().numpy(), value.cpu().numpy()

    def sample_latent_output_fsqmha(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            x = self.policy.embedder.forward_to_pool(obs)
            h = self.policy.embedder.forward_from_pool(x)
            dist, value = self.policy.hidden_to_output(h)
            y = dist.logits.detach().cpu().numpy()
            act = dist.sample()
            # if not stochastic:
            #     act = y.argmax(1)
        return x.cpu().numpy(), y, act.cpu().numpy(), value.cpu().numpy()

    def sample_latent_output_fsqmha_coinrun(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            x = self.policy.embedder.forward_to_pool(obs)
            h = self.policy.embedder.forward_from_pool(x)
            dist, value = self.policy.hidden_to_output(h)
            # y = dist.logits.detach().cpu().numpy()
            p = dist.probs.detach().cpu().numpy()
            z = inverse_sigmoid(p)
            y = z[:, (1, 3)]
            act = dist.sample()
        return x.cpu().numpy(), y, act.cpu().numpy(), value.cpu().numpy()




class MostlyNeuralAgent:
    def __init__(self, model, policy, stochastic, action_mapping):
        self.model = model
        self.policy = policy
        self.stochastic = stochastic
        self.action_mapping = action_mapping
        self.single_output = self.policy.action_size <= 2
        self.n = len(np.unique(self.action_mapping)) // 2

    def forward(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            x = self.policy.embedder(obs)
            h = self.model.predict(x)
            return self.pred_to_action(h)

    def pred_to_action(self, h):
        if self.stochastic:
            logits = torch.FloatTensor(h).to(self.policy.device)
            log_probs = F.log_softmax(logits, dim=1)
            p = Categorical(logits=log_probs)
            act = p.sample()
            return act.cpu().numpy()
        return match_to_nearest(h, self.action_mapping)

    def sample(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            x = self.policy.embedder(obs)
            dist, value = self.policy.hidden_to_output(x)
            y = dist.logits.detach().cpu().numpy()
            act = dist.sample().cpu().numpy()
            if not self.stochastic:
                act = y.argmax(axis=1)
        return x.cpu().numpy(), y, act, value.cpu().numpy()
