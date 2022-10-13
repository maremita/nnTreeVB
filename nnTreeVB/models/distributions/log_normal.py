from nnTreeVB.utils import init_parameters
from nnTreeVB.utils import build_neuralnet
from nnTreeVB.typing import *

import torch
import torch.nn as nn
from torch.distributions.log_normal import LogNormal
from torch.distributions import transform_to
from torch.distributions import TransformedDistribution

__author__ = "Amine Remita"


class VB_LogNormal(nn.Module):
    def __init__(self,
            in_shape: list,         # [..., b_dim]
            out_shape: list,        # [..., b_dim]
            # list of 2 floats, uniform, normal or False
            init_params: Union[list, str, bool] = [0.1, 0.1],
            learn_params: bool = True,
            # Sample biject transformation
            transform_dist: TorchTransform = None,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 
 
        # in_shape and out_shape should be updated if
        # the sample transform is to a simplex
        if isinstance(self.transform_dist,
                torch.distributions.StickBreakingTransform):
            self.in_shape[-1] -= 1
            self.out_shape[-1] -= 1
        
        # loc (mu) and scale (sigma) 
        self.nb_params = 2
        self.init_params = init_params

        self.transform_dist = transform_dist
        self.device_ = device

        # init parameters initialization
        self.input = init_parameters(self.init_params,
                self.nb_params)

        # Distr parameters transforms
        self.tr_to_sigma_constr = transform_to(
                LogNormal.arg_constraints['scale'])

        init_mu = self.input[0].repeat(
                [*self.in_shape])
        # Pay attention here, we use inverse transforms to 
        # transform the initial values from constrained to
        # unconstrained space
        init_sigma_unconstr = self.tr_to_sigma_constr.inv(
                self.input[1].repeat([*self.in_shape]))

        # Initialize the parameters of the distribution
        if learn_params:
            self.mu = nn.Parameter(init_mu,
                    requires_grad=True).to(self.device_)
            self.sigma_unconstr = nn.Parameter(
                    init_sigma_unconstr,
                    requires_grad=True).to(self.device_)
        else:
            self.mu = init_mu.detach(
                    ).clone().to(self.device_)
            self.sigma_unconstr = init_sigma_unconstr.detach(
                    ).clone().to(self.device_)

    def forward(self):

        # Transform sigma from unconstrained to
        # constrained space
        self.sigma = self.tr_to_sigma_constr(
                self.sigma_unconstr)

        # Approximate distribution
        base_dist = LogNormal(self.mu, self.sigma)

        if self.transform_dist is not None:
            self.dist = TransformedDistribution(base_dist,
                    self.transform_dist)
        else:
            self.dist = base_dist

        return self.dist


class VB_LogNormal_NN(nn.Module):
    def __init__(self,
            in_shape,             # [..., b_dim]
            out_shape,            # [..., b_dim]
            # list of 2 floats, uniform, normal or False
            init_params: Union[list, str, bool] = [0.1, 0.1],
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

        # in_shape and out_shape should be updated if
        # the sample transform is to a simplex
        if isinstance(self.transform_dist,
                torch.distributions.StickBreakingTransform):
            self.in_shape[-1] -= 1
            self.out_shape[-1] -= 1

        self.in_dim = self.in_shape[-1]
        self.out_dim = self.out_shape[-1]

        # loc (mu) and scale (sigma) 
        self.nb_params = 2
        self.init_params = init_params

        self.transform_dist = transform_dist

        self.h_dim = h_dim          # hidden layer size
        self.nb_layers = nb_layers
        self.bias_layers = bias_layers
        self.activ_layers = activ_layers
        self.dropout = dropout_layers
        self.device_ = device

        # Input of the neural network
        self.input = init_parameters(self.init_params,
                self.nb_params)

        self.input = self.input.repeat([*self.in_shape, 1])

        # Construct the neural networks
        self.net_mu = build_neuralnet(
            self.in_dim,
            self.out_dim,
            self.h_dim,
            self.nb_layers,
            self.bias_layers,
            self.activ_layers,
            self.dropout,
            None,
            self.device_)

        self.net_sigma = build_neuralnet(
            self.in_dim,
            self.out_dim,
            self.h_dim,
            self.nb_layers,
            self.bias_layers,
            self.activ_layers,
            self.dropout
            nn.Softplus(),
            self.device_)

    def forward(self): 

        eps = torch.finfo().eps

        self.mu = self.net_mu(self.input[...,0])
        self.sigma = self.net_sigma(
                self.input[...,1]).clamp(min=0.+eps)

        # Approximate distribution
        base_dist = LogNormal(self.mu, self.sigma)

        if self.transform_dist is not None:
            self.dist = TransformedDistribution(base_dist,
                    self.transform_dist)
        else:
            self.dist = base_dist

        return self.dist
