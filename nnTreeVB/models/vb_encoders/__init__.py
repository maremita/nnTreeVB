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
from .vb_fixed import *

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

    elif encoder_type == "fixed":
        encoder = VB_FixedEncoder

    else:
        print("warning encoder type")
        encoder = VB_FixedEncoder

    return encoder


def build_vb_encoder(
        in_shape,
        out_shape,
        encoder_type="gamma",  # gamma_ind|lognormal_nnind
        init_distr=[0.1, 0.1], 
        # if not deep: list of 2 floats
        # if deep: list of 2 floats, uniform, normal for nnInd
        # or False for nn encoders
        # or tensor for fixed encoder
        prior_hp=[0.2, 0.2],
        # Following parameters are needed if nn
        h_dim=16,
        nb_layers=3,
        bias_layers=True,     # True or False
        activ_layers="relu",  # relu, tanh, or False
        dropout_layers=0.,
        device=torch.device("cpu")):

    encoder = get_vb_encoder_type(encoder_type)

    encoder_args = dict(
            device=device)

    if encoder_type != "fixed":
        encoder_args.update(
            prior_hp=prior_hp)
    else:
        assert isinstance(init_distr, torch.Tensor)

    if init_distr is not False:
        encoder_args.update(
            init_distr=init_distr)

    if "nn" in encoder_type:
        encoder_args.update(
                h_dim=h_dim,
                nb_layers=nb_layers,
                bias_layers=bias_layers,
                activ_layers=activ_layers,
                dropout_layers=dropout_layers)

    return encoder(in_shape, out_shape, **encoder_args)
