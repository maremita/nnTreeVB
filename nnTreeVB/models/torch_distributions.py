from nnTreeVB.typing import TorchDistribution

import torch

__author__ = "Amine Remita"

torch_dist_names = [
        "normal",
        "lognormal",
        "gamma",
        "dirichlet",
        "categorical",
        "exponential",
        "uniform"
        ]

def build_torch_distribution(
        dist_type:str,
        dist_params:list,
        dtype:torch.dtype = torch.float32)->TorchDistribution:
 
    build_dist = {
        "normal":build_Normal,
        "lognormal":build_LogNormal,
        "gamma":build_Gamma,
        "dirichlet":build_Dirichlet,
        "categorical":build_Categorical,
        "exponential":build_Exponential,
        "uniform":build_Uniform
            }

    return build_dist[dist_type.lower()](dist_params, dtype)


def build_Normal(
        params: list,
        dtype:torch.dtype = torch.float32)->TorchDistribution:

    dist_params = torch.tensor(params, dtype=dtype)

    assert dist_params.shape[-1] == 2

    mu = dist_params[0]
    sigma = dist_params[1]

    return torch.distributions.normal.Normal(mu, sigma)


def build_LogNormal(
        params: list,
        dtype:torch.dtype = torch.float32)->TorchDistribution:

    dist_params = torch.tensor(params, dtype=dtype)

    assert dist_params.shape[-1] == 2

    mu = dist_params[0]
    sigma = dist_params[1]

    return torch.distributions.log_normal.LogNormal(mu, sigma)


def build_Gamma(
        params: list,
        dtype:torch.dtype = torch.float32)->TorchDistribution:

    dist_params = torch.tensor(params, dtype=dtype)

    assert dist_params.shape[-1] == 2

    alpha = dist_params[0]
    beta = dist_params[1]

    return torch.distributions.gamma.Gamma(alpha, beta)


def build_Dirichlet(
        params: list,
        dtype:torch.dtype = torch.float32)->TorchDistribution:

    dist_params = torch.tensor(params, dtype=dtype)

    return torch.distributions.dirichlet.Dirichlet(dist_params)


def build_Categorical(
        params: list,
        dtype:torch.dtype = torch.float32)->TorchDistribution:

    dist_params = torch.tensor(params, dtype=dtype)

    return torch.distributions.categorical.Categorical(
            probs=dist_params)


def build_Exponential(
        params: list,
        dtype:torch.dtype = torch.float32)->TorchDistribution:

    dist_params = torch.tensor(params, dtype=dtype)

    return torch.distributions.exponential.Exponential(
            *dist_params)


def build_Uniform(
        params: list,
        dtype:torch.dtype = torch.float32)->TorchDistribution:

    dist_params = torch.tensor(params, dtype=dtype)

    assert dist_params.shape[-1] == 2

    low = dist_params[0]
    high = dist_params[1]

    return torch.distributions.uniform.Uniform(low, high)
