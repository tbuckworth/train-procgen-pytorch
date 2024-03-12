import numpy as np
import random
import gym
import torch
import torch.nn as nn


def attention_entropy(atn):
    '''
    Expected shape: (batch, heads, height, width)
    each col should sum to 1
    '''
    p_log_p = torch.log(atn)*atn
    # ln(0)*0 = -inf*0 = nan, but can safely be replaced with 0
    # other option is to clamp with min real number, and
    # recalculate probs, but that is computationally more expensive
    p_log_p[torch.isnan(p_log_p)] = 0.
    # Normalizing as max entropy = ln(n) where n is number of variables
    entropy = -p_log_p.sum(3) / np.log(atn.shape[3])
    normalized_mean_entropy = entropy.mean()
    return normalized_mean_entropy

def cross_batch_entropy(p):
    '''
    The idea here is to emulate torch.distributions.Categorical.entropy(), but instead of computing it per batch
    item, we also compute it across the batches. This is to encourage the model to learn a diverse policy and avoid
    it always returning the same logits (effectively ignoring the inputs).
    '''

    ln = torch.log(p.probs)

    p_log_p = p.probs * ln

    entropy = - p_log_p.sum(-1)

    cond_entropy = entropy.mean()

    action_p = p.probs.mean(0)

    ap_log_p = action_p * torch.log(action_p)

    marg_entropy = - ap_log_p.sum()

    #subtract this from loss:
    return marg_entropy - cond_entropy, cond_entropy

    # Smallest representable floating point number:
    min_real = torch.finfo(p.logits.dtype).min
    # let nothing be smaller than this:
    logits = torch.clamp(p.logits, min=min_real)
    # classic entropy calc:
    p_log_p = logits * p.probs

    # Action prob entropy calc (across batch).
    action_probs = torch.nn.functional.softmax(logits, dim=0)
    action_p_log_p = logits * action_probs

    return -p_log_p.sum(-1), -action_p_log_p.sum(0)


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

def adjust_lr_grok(optimizer, init_lr, timesteps, max_timesteps):
    lr = min(init_lr * (1.1**(timesteps / 1e6)), 1.)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_n_params(model):
    return str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3)) + ' M params'
