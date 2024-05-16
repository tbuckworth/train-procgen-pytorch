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


class SymbolicAgent:
    def __init__(self, model, policy, stochastic, action_mapping):
        self.model = model
        self.policy = policy
        self.stochastic = stochastic
        self.single_output = self.policy.action_size <= 2
        self.action_mapping = action_mapping
        self.n = len(np.unique(self.action_mapping)) // 2


    def forward(self, observation):
        with torch.no_grad():
            h = self.model.predict(observation)
            if self.single_output:
                if self.stochastic:
                    p = sigmoid(h)
                    return np.int32(np.random.random(len(h)) < p)
                return np.round(h, 0)
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
