from nnTreeVB.utils import min_max_clamp
from nnTreeVB.typing import *

import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
from torch.distributions import transform_to

__author__ = "Amine Remita"


class VB_Dirichlet_IndEncoder(nn.Module):
    def __init__(self,
            in_shape: list,            # [..., 6]
            out_shape: list,           # [..., 6]
            init_distr: list = [1., 1.], 
            # list of floats, "uniform", "normal" or False
            # initialized prior distribution
            prior_dist: TorchDistribution = None,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 
        
        # Concentrations
        self.nb_params = self.in_shape[-1]

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
        self.tr_to_alphas_const = transform_to(
                Dirichlet.arg_constraints['concentration'])

        # Pay attention here, we use inverse transforms to 
        # transform the initial values from constrained to
        # unconstrained space
        init_alphas_unconst = self.tr_to_alphas_const.inv(
                self.input.repeat([*self.in_shape[:-1],1]))

        # Initialize the parameters of the variational
        # distribution q
        self.alphas_unconst = nn.Parameter(init_alphas_unconst,
                requires_grad=True)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=0.0000001,    # should be <= to 10^-7
            max_clamp=False):

        # Transform params from unconstrained to
        # constrained space
        self.alphas = self.tr_to_alphas_const(
                self.alphas_unconst)

        # Approximate distribution 
        self.dist_q = Dirichlet(self.alphas)

        # Sample from approximate distribution q
        samples = self.dist_q.rsample(
                torch.Size([sample_size]))
        #print("samples dirichlet shape {}".format(
        #    samples.shape)) # [sample_size, 6]

        samples = min_max_clamp(
                samples,
                min_clamp,
                max_clamp)

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


class VB_Dirichlet_NNIndEncoder(nn.Module):
    def __init__(self,
            in_shape,             # [..., 6]
            out_shape,            # [..., 6]
            init_distr=[1., 1.],  # list of floats, "uniform",
                                  # "normal" or False
            prior_hp=[1., 1.],
            h_dim=16, 
            nb_layers=3,
            bias_layers=True,     # True or False
            activ_layers="relu",  # relu, tanh, or False
            dropout_layers=0.,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 
        self.in_dim = self.in_shape[-1]

        # Concentration
        self.nb_params = self.in_shape[-1]
        self.init_distr = init_distr
 
        self.prior_hp = torch.tensor(prior_hp)
        self.device_ = device

        assert self.prior_hp.shape[-1] == self.nb_params

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

        # Input of the variational neural network
        if isinstance(self.init_distr, (list)):
            assert len(self.init_distr) == self.nb_params
            self.input = torch.tensor(self.init_distr)
        else:
            self.input = torch.ones(self.nb_params)

        if self.init_distr == "uniform":
            self.input = self.input.uniform_()
        elif self.init_distr == "normal":
            self.input = self.input.normal_()

        self.input = self.input.repeat([*self.in_shape[:-1],1])

        # Construct the neural network
        layers = [nn.Linear(self.in_shape[-1], self.h_dim,
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

        layers.extend([nn.Linear(self.h_dim,self.out_shape[-1],
            bias=self.bias_layers), nn.Softplus()])

        self.net = nn.Sequential(*layers)

        # Prior distribution
        self.dist_p = Dirichlet(self.prior_hp)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=0.0000001,    # should be <= to 10^-7
            max_clamp=False):

        self.alphas = self.net(self.input)

        # Approximate distribution
        self.dist_q = Dirichlet(self.alphas)

        # Sample
        samples = self.dist_q.rsample(
                torch.Size([sample_size]))
        #print("samples dirichlet deep shape {}".format(
        #    samples.shape)) # [sample_size, 6]

        samples = min_max_clamp(samples, min_clamp, max_clamp)
 
        with torch.set_grad_enabled(KL_gradient):
            kl = kl_divergence(self.dist_q, self.dist_p)

        with torch.set_grad_enabled(not KL_gradient):
            # Compute log prior of samples p(d)
            logprior = self.dist_p.log_prob(samples)
            #print("logprior.shape {}".format(logprior.shape))

            # Compute the log of approximate posteriors q(d)
            logq = self.dist_q.log_prob(samples)
            #print("logq.shape {}".format(logq.shape))

        return logprior, logq, kl, samples


class VB_Dirichlet_NNEncoder(nn.Module):
    def __init__(self,
            in_shape,   # [n_dim, x_dim , m_dim]
            #in_dim,    # x_dim * m_dim
            #out_dim,   # x_dim * a_dim
            out_shape,  # [n_dim, a_dim, x_dim]
            prior_hp=[1., 1.],
            h_dim=16,
            nb_layers=3,
            bias_layers=True,     # True or False
            activ_layers="relu", # relu, tanh, or False
            dropout_layers=0.,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 

        self.in_dim = self.in_shape[-1] * self.in_shape[-2] 
        self.out_dim = self.out_shape[-1] * self.out_shape[-2]

        self.prior_hp = torch.tensor(prior_hp)

        self.h_dim = h_dim  # hidden layer size
        self.n_layers = n_layers
        self.bias_layers = bias_layers
        self.activ_layers = activ_layers
        self.dropout = dropout_layers
        self.device_ = device

        if self.activ_layers == "relu":
            activation = nn.ReLU
        elif self.activ_layers == "tanh":
            activation = nn.Tanh

        if self.nb_layers < 2:
            self.nb_layers = 2
            print("The number of layers in {} should"\
                    " be >= 2. It's set set to 2".format(self))

        # Construct the neural network
        layers = [nn.Linear(self.in_dim, self.h_dim,
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

        layers.extend([nn.Linear(self.h_dim, self.out_dim,
            bias=self.bias_layers), nn.Softplus()])

        self.net = nn.Sequential(*layers)

        # Prior distribution
        self.dist_p = Dirichlet(probs=self.prior_hp)

    def forward(
            self,
            data,
            sample_size=1,
            KL_gradient=False,
            min_clamp=0.0000001,    # should be <= to 10^-7
            max_clamp=False):

        # Flatten the data
        #data = data.squeeze(0).flatten(0)
        data = data.flatten(-2)
        #print("data_flatten.shape")
        #print(data.shape)
        # [n_dim, m_dim * x_dim]

        self.alphas = self.net(data).view(*self.out_shape)
        #print("alphas")
        #print(alphas.shape)
        #print(alphas)

        # Approximate distribution
        self.dist_q = Dirichlet(self.alphas)

        # Sample
        samples = self.dist_q.rsample(
                torch.Size([sample_size]))
        #print("samples dirichlet deep shape {}".format(
        #    samples.shape)) # [sample_size, 6]
 
        samples = min_max_clamp(samples, min_clamp, max_clamp)
 
        with torch.set_grad_enabled(KL_gradient):
            kl = kl_divergence(self.dist_q, self.dist_p)

        with torch.set_grad_enabled(not KL_gradient):
            # Compute log prior of samples p(d)
            logprior = self.dist_p.log_prob(samples)
            #print("logprior.shape {}".format(logprior.shape))

            # Compute the log of approximate posteriors q(d)
            logq = self.dist_q.log_prob(samples)
            #print("logq.shape {}".format(logq.shape))

        return logprior, logq, kl, samples
