from nnTreeVB.typing import TorchDistribution

import torch

__author__ = "Amine Remita"


def build_distribution(
        dist_type:str,
        dist_params:list)->TorchDistribution:
    
    build_dist = {
        "build_normal":build_Normal,
        "build_lognormal":build_LogNormal,
        "build_gamma":build_Gamma,
        "build_dirichlet":build_Dirichlet,
        "build_categorical":build_Categorical,
        "build_exponential":build_Exponential,
        "build_uniform":build_Uniform
            }

    return build_dist["build_"+dist_type.lower()](dist_params)


def build_Normal(
        params: list)->TorchDistribution:

    dist_params = torch.tensor(params)

    assert dist_params.shape[-1] == 2

    mu = dist_params[0]
    sigma = dist_params[1]

    return torch.distributions.normal.Normal(mu, sigma)


def build_LogNormal(
        params: list)->TorchDistribution:

    dist_params = torch.tensor(params)

    assert dist_params.shape[-1] == 2

    mu = dist_params[0]
    sigma = dist_params[1]

    return torch.distributions.log_normal.LogNormal(mu, sigma)


def build_Gamma(
        params: list)->TorchDistribution:

    dist_params = torch.tensor(params)

    assert dist_params.shape[-1] == 2

    alpha = dist_params[0]
    beta = dist_params[1]

    return torch.distributions.gamma.Gamma(alpha, beta)


def build_Dirichlet(
        params: list)->TorchDistribution:

    alphas = torch.tensor(params)

    return torch.distributions.dirichlet.Dirichlet(alphas)


def build_Categorical(
        params: list)->TorchDistribution:

    probs = torch.tensor(params)

    return torch.distributions.categorical.Categorical(
            probs=probs)


def build_Exponential(
        params: list)->TorchDistribution:

    rate = torch.tensor(params)

    return torch.distributions.exponential.Exponential(rate)


def build_Uniform(
        params: list)->TorchDistribution:

    dist_params = torch.tensor(params)

    assert dist_params.shape[-1] == 2

    low = dist_params[0]
    high = dist_params[1]

    return torch.distributions.uniform.Uniform(low, high)
