from math import log, pi

import torch


def gaussian_logp(x, mean, logsd):
    return -0.5 * log(2 * pi) - logsd - 0.5 * (x - mean) ** 2 / torch.exp(2 * logsd)


def gaussian_sample(eps, mean, logsd):
    return mean + torch.exp(logsd) * eps
