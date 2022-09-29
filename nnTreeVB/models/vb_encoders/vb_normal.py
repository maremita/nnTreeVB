from nnTreeVB.utils import min_max_clamp
from torch.distributions import transform_to

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

__author__ = "Amine Remita"


class VB_Normal_IndEncoder(nn.Module):
    def __init__(self,
            in_shape,              # [..., b_dim]
            out_shape,             # [..., b_dim]
            init_distr=[0.1, 0.1], # list of 2 floats, uniform,
                                   # normal or False
            prior_hp=[0.2, 0.2],   # prior hyper-parameters
            transform=None,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 
 
        # loc (mu) and scale (sigma) 
        self.nb_params = 2
        self.init_distr = init_distr

        self.prior_hp = torch.tensor(prior_hp)
        self.transform = transform
        self.device_ = device

        # in_shape will be updated if the sample transform 
        # is to a simplex
        if isinstance(self.transform,
                torch.distributions.StickBreakingTransform):
            self.in_shape[-1] -= 1

        assert self.prior_hp.shape[-1] == self.nb_params

        self.prior_mu = self.prior_hp[0].detach()
        self.prior_sigma = self.prior_hp[1].detach()

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
        self.tr_to_sigma_const = transform_to(
                Normal.arg_constraints['scale'])

        init_mu = self.input[0].repeat(
                [*self.in_shape])
        # Pay attention here, we use inverse transforms to 
        # transform the initial values from constrained to
        # unconstrained space
        init_sigma_unconst = self.tr_to_sigma_const.inv(
                self.input[1].repeat([*self.in_shape]))

        # Initialize the parameters of the variational
        # distribution q
        self.mu = nn.Parameter(init_mu,
                requires_grad=True)
        self.sigma_unconst = nn.Parameter(init_sigma_unconst,
                requires_grad=True)

        # Prior distribution
        self.dist_p = Normal(self.prior_mu,
                self.prior_sigma)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=False,
            max_clamp=False):

        # Transform sigma from unconstrained to
        # constrained space
        self.sigma = self.tr_to_sigma_const(
                self.sigma_unconst)

        # Approximate distribution
        self.dist_q = Normal(self.mu, self.sigma)

        # Sample from approximate distribution q
        # in the unconstrained space
        samples_unconst = self.dist_q.rsample(
                torch.Size([sample_size]))
        #print("samples N shape {}".format(samples.shape))

        samples_unconst = min_max_clamp(
                samples_unconst, 
                min_clamp, 
                max_clamp)

        # Transform samples_unconst to constrained space
        if self.transform is not None:
            samples = self.transform(samples_unconst)
        else:
            samples = samples_unconst

        with torch.set_grad_enabled(KL_gradient):
            kl = kl_divergence(self.dist_q, self.dist_p)
            #print("kl.shape {}".format(kl.shape))

        with torch.set_grad_enabled(not KL_gradient):
            # Compute log prior of samples p(d)
            logprior = self.dist_p.log_prob(samples)
            #print("logprior.shape {}".format(logprior.shape))

            # Compute the log of approximate posteriors q(d)
            logq = self.dist_q.log_prob(samples_unconst)
            #print("logq.shape {}".format(logq.shape))

        return logprior, logq, kl, samples


class VB_Normal_NNIndEncoder(nn.Module):
    def __init__(self,
            in_shape,              # [..., b_dim]
            out_shape,             # [..., b_dim]
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

        self.nb_params = 2
        self.init_distr = init_distr

        self.prior_hp = torch.tensor(prior_hp)
        self.device_ = device

        assert self.prior_hp.shape[-1] == self.nb_params

        self.prior_mu = self.prior_hp[0].detach()
        self.prior_sigma = self.prior_hp[1].detach()
 
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

        self.input = self.input.repeat([*self.in_shape, 1])

        # Construct the neural network
        self.net_in_mu = nn.Linear(self.in_shape[-1],
                h_dim, bias=self.bias_layers)
        self.net_in_sigma = nn.Linear(self.in_shape[-1],
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

        self.net_out_mu = nn.Sequential(
            nn.Linear(self.h_dim, self.out_shape[-1]))

        self.net_out_sigma = nn.Sequential(
            nn.Linear(self.h_dim, self.out_shape[-1]),
            nn.Softplus()) 

        # Prior distribution
        self.dist_p = Normal(self.prior_mu,
                self.prior_sigma)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=False,
            max_clamp=False):

        eps = torch.finfo().eps

        out_m = self.net_in_mu(self.input[...,0])
        out_s = self.net_in_sigma(self.input[...,1])
        h_ms = self.net_h(torch.cat([out_m, out_s]))

        self.mu = self.net_out_mu(h_ms)
        self.sigma = self.net_out_sigma(h_ms).clamp(min=0.+eps)

        # Approximate distribution
        self.dist_q = LogNormal(self.mu, self.sigma)

        # Sample
        samples = self.dist_q.rsample(
                torch.Size([sample_size]))
        # print("samples shape {}".format(samples.shape)) #

        samples = min_max_clamp(samples, min_clamp, max_clamp)

        with torch.set_grad_enabled(KL_gradient):
            kl = kl_divergence(self.dist_q, self.dist_p)
            #print("kl.shape {}".format(kl.shape))

        with torch.set_grad_enabled(not KL_gradient):
            # Compute log prior of samples p(d)
            logprior = self.dist_p.log_prob(samples)
            # print("logprior.shape {}".format(logprior.shape))

            # Compute the log of approximate posteriors q(d)
            logq = self.dist_q.log_prob(samples)
            # print("logq.shape {}".format(logq.shape))

        return logprior, logq, kl, samples
