from nnTreeVB.utils import init_parameters
from nnTreeVB.utils import build_neuralnet
from nnTreeVB.utils import freeze_model_params
from nnTreeVB.typing import *

import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet
from torch.distributions import transform_to

__author__ = "Amine Remita"


class VB_Dirichlet(nn.Module):
    def __init__(self,
            in_shape: list,            # [..., r_dim]
            out_shape: list,           # [..., r_dim]
            # list of floats, "uniform", "normal" or False
            init_params: Union[list, str, bool] = False,
            learn_params: bool = True,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 
 
        # Concentrations
        self.nb_params = self.in_shape[-1]
        self.init_params = init_params
        self.learn_params = learn_params
        self.device_ = device

        # init parameters initialization
        self.input = init_parameters(self.init_params,
                self.nb_params)

        # Dist parameters transforms
        self.tr_to_alphas_constr = transform_to(
                Dirichlet.arg_constraints['concentration'])

        # Pay attention here, we use inverse transforms to 
        # transform the initial values from constrained to
        # unconstrained space
        init_alphas_unconstr = self.tr_to_alphas_constr.inv(
                self.input.repeat([*self.in_shape[:-1],1]))

        # Initialize the parameters of the distribution
        if self.learn_params:
            self.alphas_unconstr = nn.Parameter(
                    init_alphas_unconstr,
                    requires_grad=True).to(self.device_)
        else:
            self.alphas_unconstr = init_alphas_unconstr.detach(
                    ).clone().to(self.device_)

    def forward(self): 

        # Transform params from unconstrained to
        # constrained space
        self.alphas = self.tr_to_alphas_constr(
                self.alphas_unconstr)

        # Initialize the distribution
        self.dist = Dirichlet(self.alphas)

        return self.dist


class VB_Dirichlet_NN(nn.Module):
    def __init__(self,
            in_shape: list,            # [..., r_dim]
            out_shape: list,           # [..., r_dim]
            init_params: Union[list, str, bool] = [0.1, 0.1],
            # list of floats, "uniform", "normal" or False
            learn_params: bool = True,
            h_dim: int = 16, 
            nb_layers: int = 3,
            bias_layers: bool = True,
            activ_layers: str = "relu",# relu, tanh, or False
            dropout_layers: float = 0.,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 
        self.in_dim = self.in_shape[-1]
        self.out_dim = self.out_shape[-1]

        # Concentrations
        self.nb_params = self.in_shape[-1]
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
                [*self.in_shape[:-1],1]).to(self.device_)

        self.net = build_neuralnet(
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
            freeze_model_params(self.net)

    def forward(self): 

        if not self.learn_params:
            freeze_model_params(self.net)

        self.alphas = self.net(self.input)

        # Initialize the distribution
        self.dist = Dirichlet(self.alphas)

        return self.dist


class VB_Dirichlet_NNX(nn.Module):
    def __init__(self,
            in_shape: list,   # [n_dim, x_dim , m_dim]
            out_shape: list,  # [n_dim, a_dim, x_dim]
            learn_params: bool = True,
            h_dim: int = 16, 
            nb_layers: int =3,
            bias_layers: bool = True,
            activ_layers: str = "relu",# relu, tanh, or False
            dropout_layers: float = 0.,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 

        #in_dim,    # x_dim * m_dim
        self.in_dim = self.in_shape[-1] * self.in_shape[-2] 
        #out_dim,   # x_dim * a_dim
        self.out_dim = self.out_shape[-1] * self.out_shape[-2]

        self.learn_params = learn_params
        self.h_dim = h_dim  # hidden layer size
        self.n_layers = n_layers
        self.bias_layers = bias_layers
        self.activ_layers = activ_layers
        self.dropout = dropout_layers
        self.device_ = device

        self.net = build_neuralnet(
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
            freeze_model_params(self.net)

    def forward(self, X):

        if not self.learn_params:
            freeze_model_params(self.net)

        # Flatten the data X
        #data = X.squeeze(0).flatten(0)
        data = X.flatten(-2)
        #print("data_flatten.shape")
        #print(data.shape)
        # [n_dim, m_dim * x_dim]

        self.alphas = self.net(data).view(*self.out_shape)
        #print("alphas")
        #print(alphas.shape)
        #print(alphas)

        # Initialize the distribution
        self.dist = Dirichlet(self.alphas)

        return self.dist
