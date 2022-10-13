#################################
##                             ##
##         nnTreeVB            ##
##  2022 (C) Amine Remita      ##
##                             ##
#################################

from .dirichlet import *
from .exponential import *
from .fixed import *
from .gamma import *
from .log_normal import *
from .normal import *

from nnTreeVB.typing import *

import torch

__author__ = "amine remita"

__all__ = [
        "get_distribution",
        "build_distribution",
        ]

def get_distribution(dist_type="gamma"):
    dist_type = dist_type.lower()

    if dist_type == "gamma":
        distribution = VB_Gamma

    elif dist_type == "gamma_nn":
        distribution = VB_Gamma_NN

    elif dist_type == "lognormal":
        distribution = VB_LogNormal

    elif dist_type == "lognormal_nn":
        distribution = VB_LogNormal_NN

    elif dist_type == "dirichlet":
        distribution = VB_Dirichlet

    elif dist_type == "dirichlet_nn":
        distribution = VB_Dirichlet_NN

    elif dist_type == "dirichlet_nnx":
        distribution = VB_Dirichlet_NNX

    elif dist_type == "normal":
        distribution = VB_Normal

    elif dist_type == "normal_nn":
        distribution = VB_Normal_NN

    elif dist_type == "fixed":
        distribution = VB_Fixed

    else:
        print("warning distribution type")
        distribution = VB_Fixed

    return distribution


def build_distribution(
        in_shape: list,
        out_shape: list,
        dist_type: str = "gamma", # gamma
        init_params: list = [0.1, 0.1], 
        # if not deep: list of 2 floats
        # if deep: list of 2 floats, uniform, normal
        # or False for nn distributions
        learn_params: bool = True,
        transform_dist: TorchTransform = None,
        # Following parameters are needed if nn
        h_dim: int = 16,
        nb_layers: int = 3,
        bias_layers: bool = True,    # True or False
        activ_layers: str = "relu",  # relu, tanh, or False
        dropout_layers:float = 0.,
        device: torch.device = torch.device("cpu")):

    distribution = get_dist(dist_type)

    dist_args = dict(
            device=device)

    if init_params is not False:
        dist_args.update(
            init_params=init_params)

    if transform_dist is not None:
        dist_args.update(
            transform_dist=transform_dist)

    if "nn" in dist_type:
        dist_args.update(
                h_dim=h_dim,
                nb_layers=nb_layers,
                bias_layers=bias_layers,
                activ_layers=activ_layers,
                dropout_layers=dropout_layers)
    else:
        dist_args.update(learn_params=learn_params)

    return distribution(in_shape, out_shape, **dist_args)
