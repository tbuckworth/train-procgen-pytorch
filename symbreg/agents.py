import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from helper_local import sigmoid, inverse_sigmoid, match, softmax, sample_numpy_probs


class SymbolicAgent:
    def __init__(self, model, policy, stochastic, action_mapping):
        self.model = model
        self.policy = policy
        self.stochastic = stochastic
        self.single_output = self.policy.action_size <= 2

    def forward(self, observation):
        with torch.no_grad():
            h = self.model.predict(observation)
            if self.single_output:
                if self.stochastic:
                    p = sigmoid(h)
                    return np.int32(np.random.random(len(h)) < p)
                return np.round(h, 0)
            if self.stochastic:
                p = softmax(h)
                return sample_numpy_probs(p)
            return h.argmax(1)
    #TODO: use this:
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
        rads = np.round(h / (np.pi / self.n), 0) * (np.pi / self.n)
        rads[rads < 0] = -1
        return match(rads, self.action_mapping)
