#################################
##                             ##
##         nnTreeVB            ##
##  2022 (C) Amine Remita      ##
##                             ##
#################################


from .vb_dirichlet import *
from .vb_categorical import *
from .vb_gamma import *
from .vb_logNormal import *
from .vb_normal import *
from .vb_fixed import *
from .build_distributions import build_distribution 

from nnTreeVB.typing import *

import torch

__author__ = "amine remita"


__all__ = [
        "get_vb_encoder",
        "build_vb_encoder"
        "build_distribution"
        ]

def get_vb_encoder(encoder_type="gamma"):
    encoder_type = encoder_type.lower()

    if encoder_type == "gamma_ind":
        encoder = VB_Gamma_IndEncoder

    elif encoder_type == "gamma_nn_ind":
        encoder = VB_Gamma_NNIndEncoder

    elif encoder_type == "lognormal_ind":
        encoder = VB_LogNormal_IndEncoder

    elif encoder_type == "lognormal_nn_ind":
        encoder = VB_LogNormal_NNIndEncoder

    elif encoder_type == "dirichlet_ind":
        encoder = VB_Dirichlet_IndEncoder

    elif encoder_type == "dirichlet_nn_ind":
        encoder = VB_Dirichlet_NNIndEncoder

    elif encoder_type == "dirichlet_nn":
        encoder = VB_Dirichlet_NNEncoder

    elif encoder_type == "categorical_nn":
        encoder = VB_Categorical_NNEncoder

    elif encoder_type == "normal_ind":
        encoder = VB_Normal_IndEncoder

    elif encoder_type == "normal_nn_ind":
        encoder = VB_Normal_NNIndEncoder

    elif encoder_type == "fixed":
        encoder = VB_FixedEncoder

    else:
        print("warning encoder type")
        encoder = VB_FixedEncoder

    return encoder


def build_vb_encoder(
        in_shape: list,
        out_shape: list,
        encoder_type: str = "gamma", # gamma_ind
        init_distr: list = [0.1, 0.1], 
        # if not deep: list of 2 floats
        # if deep: list of 2 floats, uniform, normal for nnInd
        # or False for nn encoders
        # or tensor for fixed encoder
        prior_dist: TorchDistribution = None,
        transform: TorchTransform = None,
        # Following parameters are needed if nn
        h_dim: int = 16,
        nb_layers: int = 3,
        bias_layers: bool = True,     # True or False
        activ_layers: str = "relu",  # relu, tanh, or False
        dropout_layers:float = 0.,
        device: torch.device = torch.device("cpu")):

    encoder = get_vb_encoder(encoder_type)

    encoder_args = dict(
            device=device)

    if encoder_type != "fixed":
        encoder_args.update(
            prior_dist=prior_dist)
    else:
        assert isinstance(init_distr, torch.Tensor)

    if init_distr is not False:
        encoder_args.update(
            init_distr=init_distr)

    if transform is not None:
        encoder_args.update(
            transform=transform)

    if "nn" in encoder_type:
        encoder_args.update(
                h_dim=h_dim,
                nb_layers=nb_layers,
                bias_layers=bias_layers,
                activ_layers=activ_layers,
                dropout_layers=dropout_layers)

    return encoder(in_shape, out_shape, **encoder_args)
