from nnTreeVB.utils import init_parameters
from nnTreeVB.utils import build_neuralnet
from nnTreeVB.utils import freeze_model_params
from nnTreeVB.typing import *

import torch
import torch.nn as nn
from torch.distributions.gamma import Gamma
from torch.distributions import transform_to
from torch.distributions import TransformedDistribution

__author__ = "Amine Remita"


class VB_Gamma(nn.Module):
    def __init__(self,
            in_shape: list,         # [..., b_dim]
            out_shape: list,        # [..., b_dim]
            # list of 2 floats, "uniform", "normal" or False
            init_params: Union[list, str, bool] = [0.1, 0.1],
            learn_params: bool = True,
            # Sample biject transformation
            transform_dist: TorchTransform = None,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 

        self.transform_dist = transform_dist

        # in_shape and out_shape should be updated if
        # the sample transform is to a simplex
        if isinstance(self.transform_dist,
                torch.distributions.StickBreakingTransform):
            self.in_shape[-1] -= 1
            self.out_shape[-1] -= 1

        # shape (concentration, alpha) and rate (beta)
        self.nb_params = 2
        self.init_params = init_params
        self.learn_params = learn_params
        self.device_ = device

        # init parameters initialization
        self.input = init_parameters(self.init_params,
                self.nb_params)

        # Dist parameters transforms
        self.tr_to_alpha_constr = transform_to(
                Gamma.arg_constraints['concentration'])

        self.tr_to_beta_constr = transform_to(
                Gamma.arg_constraints['rate'])
        
        # Pay attention here, we use inverse transforms to 
        # transform the initial values from constrained to
        # unconstrained space
        init_alpha_unconstr = self.tr_to_alpha_constr.inv(
                self.input[0].repeat([*self.in_shape]))
        init_beta_unconstr = self.tr_to_beta_constr.inv(
                self.input[1].repeat([*self.in_shape]))

        # Initialize the parameters of the distribution
        if self.learn_params:
            self.alpha_unconstr = nn.Parameter(
                    init_alpha_unconstr,
                    requires_grad=True).to(self.device_)
            self.beta_unconstr = nn.Parameter(
                    init_beta_unconstr,
                    requires_grad=True).to(self.device_)
        else:
            self.alpha_unconstr = init_alpha_unconstr.detach(
                    ).clone().to(self.device_)
            self.beta_unconstr = init_beta_unconstr.detach(
                    ).clone().to(self.device_)

    def forward(self):

        # Transform params from unconstrained to
        # constrained space
        self.alpha = self.tr_to_alpha_constr(
                self.alpha_unconstr)
        self.beta = self.tr_to_beta_constr(
                self.beta_unconstr)

        # Initialize the distribution
        base_dist = Gamma(self.alpha, self.beta)

        if self.transform_dist is not None:
            self.dist = TransformedDistribution(base_dist,
                    self.transform_dist)
        else:
            self.dist = base_dist

        return self.dist


class VB_Gamma_NN(nn.Module):
    def __init__(self,
            in_shape,             # [..., b_dim]
            out_shape,            # [..., b_dim]
            # list of 2 floats, uniform, normal or False
            init_params: Union[list, str, bool] = [0.1, 0.1],
            learn_params: bool = True,
            # Sample biject transformation
            transform_dist: TorchTransform = None,
            h_dim: int = 16, 
            nb_layers: int =3,
            bias_layers: bool = True,
            activ_layers: str = "relu",# relu, tanh, or False
            dropout_layers: float = 0.,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 

        self.transform_dist = transform_dist

        # in_shape and out_shape should be updated if
        # the sample transform is to a simplex
        if isinstance(self.transform_dist,
                torch.distributions.StickBreakingTransform):
            self.in_shape[-1] -= 1
            self.out_shape[-1] -= 1

        self.in_dim = self.in_shape[-1]
        self.out_dim = self.out_shape[-1]

        # shape (concentration, alpha) and rate (beta)
        self.nb_params = 2
        self.init_params = init_params
        self.learn_params = learn_params

        self.h_dim = h_dim          # hidden layer size
        self.nb_layers = nb_layers
        self.bias_layers = bias_layers
        self.activ_layers = activ_layers
        self.dropout = dropout_layers 
        self.device_ = device

        # Input of the neural network
        self.input = init_parameters(self.init_params,
                self.nb_params)
 
        self.input = self.input.repeat(
                [*self.in_shape, 1]).to(self.device_)

        # Construct the neural networks
        self.net_alpha = build_neuralnet(
            self.in_dim,
            self.out_dim,
            self.h_dim,
            self.nb_layers,
            self.bias_layers,
            self.activ_layers,
            self.dropout,
            nn.Softplus(),
            self.device_)

        self.net_beta = build_neuralnet(
            self.in_dim,
            self.out_dim,
            self.h_dim,
            self.nb_layers,
            self.bias_layers,
            self.activ_layers,
            self.dropout,
            nn.Softplus(),
            self.device_)

        if not self.learn_params:
            freeze_model_params(self.net_alpha)
            freeze_model_params(self.net_beta)

    def forward(self): 

        eps = torch.finfo().eps

        if not self.learn_params:
            freeze_model_params(self.net_alpha)
            freeze_model_params(self.net_beta)

        self.alpha = self.net_alpha(
                self.input[...,0]).clamp(min=0.+eps)
        self.beta = self.net_beta(
                self.input[...,1]).clamp(min=0.+eps)

        # Initialize the distribution
        base_dist = Gamma(self.alpha, self.beta)

        if self.transform_dist is not None:
            self.dist = TransformedDistribution(base_dist,
                    self.transform_dist)
        else:
            self.dist = base_dist

        return self.dist
