import numpy as np
import random
import gym
import torch
import torch.nn as nn


def cross_batch_entropy(p):
    '''
    The idea here is to emulate torch.distributions.Categorical.entropy(), but instead of computing it per batch
    item, we also compute it across the batches. This is to encourage the model to learn a diverse policy and avoid
    it always returning the same logits (effectively ignoring the inputs).
    '''
    min_real = torch.finfo(p.logits.dtype).min
    logits = torch.clamp(p.logits, min=min_real)
    p_log_p = logits * p.probs
    return -p_log_p.sum(-1), -p_log_p.sum(0)


def set_global_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_global_log_levels(level):
    gym.logger.set_level(level)


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_n_params(model):
    return str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3)) + ' M params'
