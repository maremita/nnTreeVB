from nnTreeVB.utils import min_max_clamp
from nnTreeVB.typing import *

import torch
import torch.nn as nn
from torch.distributions.gamma import Gamma
from torch.distributions.kl import kl_divergence
from torch.distributions import transform_to

__author__ = "Amine Remita"


class VB_Gamma_IndEncoder(nn.Module):
    def __init__(self,
            in_shape: list,         # [..., b_dim]
            out_shape: list,        # [..., b_dim]
            # list of 2 floats, uniform, normal or False
            init_distr: list = [0.1, 0.1],
            # initialized prior distribution
            prior_dist: TorchDistribution = None,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 

        # shape (concentration, alpha) and rate (beta)
        self.nb_params = 2
        self.init_distr = init_distr

        # Prior distribution
        self.dist_p = prior_dist
        
        self.device_ = device

        # init parameters initialization
        if isinstance(self.init_distr, (list)):
            assert len(self.init_distr) == self.nb_params
            self.input = torch.tensor(self.init_distr)
        else:
            self.input = torch.ones(self.nb_params)

        if self.init_distr == "uniform":
            self.input = self.input.uniform_()
        elif self.init_distr == "normal":
            self.input = self.input.normal_()

        # Distr parameters transforms
        self.tr_to_alpha_const = transform_to(
                Gamma.arg_constraints['concentration'])

        self.tr_to_beta_const = transform_to(
                Gamma.arg_constraints['rate'])
        
        # Pay attention here, we use inverse transforms to 
        # transform the initial values from constrained to
        # unconstrained space
        init_alpha_unconst = self.tr_to_alpha_const.inv(
                self.input[0].repeat([*self.in_shape]))
        init_beta_unconst = self.tr_to_beta_const.inv(
                self.input[1].repeat([*self.in_shape]))

        # Initialize the parameters of the variational
        # distribution q
        self.alpha_unconst = nn.Parameter(init_alpha_unconst,
                requires_grad=True)
        self.beta_unconst = nn.Parameter(init_beta_unconst,
                requires_grad=True)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=0.0000001,
            max_clamp=False):
 
        # Transform params from unconstrained to
        # constrained space
        self.alpha = self.tr_to_alpha_const(
                self.alpha_unconst)
        self.beta = self.tr_to_beta_const(
                self.beta_unconst)

        # Approximate distribution 
        self.dist_q = Gamma(self.alpha, self.beta)

        # Sample from approximate distribution q
        samples = self.dist_q.rsample(
                torch.Size([sample_size]))
        #print("samples gamma shape {}".format(samples.shape))
        #[sample_size, b_dim]

        samples = min_max_clamp(
                samples,
                min_clamp,
                max_clamp)

        with torch.set_grad_enabled(KL_gradient):
            kl = kl_divergence(self.dist_q, self.dist_p)
            #print("kl.shape {}".format(kl.shape))
            #[b_dim]

        with torch.set_grad_enabled(not KL_gradient):
            # Compute log prior of samples p(d)
            logprior = self.dist_p.log_prob(samples)
            #print("logprior.shape {}".format(logprior.shape))
            #[sample_size, b_dim]

            # Compute the log of approximate posteriors q(d)
            logq = self.dist_q.log_prob(samples)
            #print("logq.shape {}".format(logq.shape))
            #[sample_size, b_dim] 

        return logprior, logq, kl, samples


class VB_Gamma_NNIndEncoder(nn.Module):
    def __init__(self,
            in_shape,             # [..., b_dim]
            out_shape,            # [..., b_dim]
            init_distr="uniform", # list of 2 floats, uniform,
                                  # normal or False
            prior_hp=[0.2, 0.2],
            h_dim=16, 
            nb_layers=3,
            bias_layers=True,     # True or False
            activ_layers="relu", # relu, tanh, or False
            dropout_layers=0.,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 

        # shape (concentration, alpha) and rate (beta)
        self.nb_params = 2
        self.init_distr = init_distr

        self.prior_hp = torch.tensor(prior_hp)
        self.device_ = device

        assert self.prior_hp.shape[-1] == self.nb_params

        self.prior_alpha = self.prior_hp[0]
        self.prior_beta = self.prior_hp[1]

        self.h_dim = h_dim          # hidden layer size
        self.nb_layers = nb_layers
        self.bias_layers = bias_layers
        self.activ_layers = activ_layers
        self.dropout = dropout_layers

        if self.activ_layers == "relu":
            activation = nn.ReLU
        elif self.activ_layers == "tanh":
            activation = nn.Tanh

        if self.nb_layers < 2:
            self.nb_layers = 2
            print("The number of layers in {} should"\
                    " be >= 2. It's set set to 2".format(self))

        assert 0. <= self.dropout <= 1.

        # Input of the variational neural network
        if isinstance(self.init_distr, (list)):
            assert len(self.init_distr) == self.nb_params
            self.input = torch.tensor(self.init_distr)
        else:
            self.input = torch.ones(self.nb_params)
        #print("self.input.shape {}".format(self.input.shape))

        if self.init_distr == "uniform":
            self.input = self.input.uniform_()
        elif self.init_distr == "normal":
            self.input = self.input.normal_()
 
        self.input = self.input.repeat([*self.in_shape, 1])
        #print("self.input.shape {}".format(self.input.shape))

        # Construct the neural network
        self.net_in_alpha = nn.Linear(self.in_shape[-1],
                h_dim, bias=self.bias_layers)
        self.net_in_beta = nn.Linear(self.in_shape[-1],
                h_dim, bias=self.bias_layers)

        layers = [nn.Linear(h_dim * 2,
            self.h_dim,
            bias=self.bias_layers)]
        if self.activ_layers: layers.append(activation())
        if self.dropout: layers.append(
                nn.Dropout(p=self.dropout))

        for i in range(1, self.nb_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=self.bias_layers)])
            if self.activ_layers: layers.append(activation())
            if self.dropout: layers.append(
                    nn.Dropout(p=self.dropout))

        self.net_h = nn.Sequential(*layers)

        self.net_out_alpha = nn.Sequential(
            nn.Linear(self.h_dim, self.out_shape[-1]),
            nn.Softplus())

        self.net_out_beta = nn.Sequential(
            nn.Linear(self.h_dim, self.out_shape[-1]),
            nn.Softplus())

        # Prior distribution
        self.dist_p = Gamma(self.prior_alpha, self.prior_beta)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=0.0000001,
            max_clamp=False):

        eps = torch.finfo().eps

        out_a = self.net_in_alpha(self.input[...,0])
        out_b = self.net_in_beta(self.input[...,1])
        h_ab = self.net_h(torch.cat([out_a, out_b]))
        
        self.alpha = self.net_out_alpha(h_ab).clamp(min=0.+eps)
        self.beta = self.net_out_beta(h_ab).clamp(min=0.+eps)

        # Approximate distribution
        self.dist_q = Gamma(self.alpha, self.beta)

        # Sample
        samples = self.dist_q.rsample(
                torch.Size([sample_size]))
        #print("samples gamma shape {}".format(samples.shape))

        samples = min_max_clamp(samples, min_clamp, max_clamp)
 
        with torch.set_grad_enabled(KL_gradient):
            kl = kl_divergence(self.dist_q, self.dist_p)
            #print("kl.shape {}".format(kl.shape))

        with torch.set_grad_enabled(not KL_gradient):
            # Compute log prior of samples p(d)
            logprior = self.dist_p.log_prob(samples)
            #print("logprior.shape {}".format(logprior.shape))

            # Compute the log of approximate posteriors q(d)
            logq = self.dist_q.log_prob(samples)
            #print("logq.shape {}".format(logq.shape))

        return logprior, logq, kl, samples
