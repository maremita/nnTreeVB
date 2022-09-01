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

import torch

__author__ = "amine remita"


__all__ = [
        "get_vb_encoder_type",
        "build_vb_encoder"
        ]


def get_vb_encoder_type(encoder_type="gamma"):
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

    else:
        print("warning encoder type")
        encoder = VB_Gamma_IndEncoder
    
    return encoder


def build_vb_encoder(
        in_shape,
        out_shape,
        encoder_type="gamma",  # gamma_ind|lognormal_nnind
        init_distr=[0.1, 0.1], 
        # if not deep: list of 2 floats
        # if deep: list of 2 floats, uniform, normal for nnInd
        # or False for nn encoders
        prior_hp=[0.2, 0.2],
        # Following parameters are needed id deep_encoder is True
        h_dim=16, 
        nb_layers=3,
        bias_layers=True,     # True or False
        activ_layers="relu",  # relu, tanh, or False
        device=torch.device("cpu")):

    encoder = get_vb_encoder_type(encoder_type)

    encoder_args = dict(
            prior_hp=prior_hp,
            device=device)

    if init_distr:
        encoder_args.update(
            init_distr=init_distr)

    if "nn" in encoder_type:
        encoder_args.update(
                h_dim=h_dim,
                nb_layers=nb_layers,
                bias_layers=bias_layers,
                activ_layers=activ_layers)

    return encoder(in_shape, out_shape, **encoder_args)