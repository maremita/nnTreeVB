import torch
import torch.nn as nn
from torch.distributions.log_normal import LogNormal
from torch.distributions.kl import kl_divergence

__author__ = "Amine Remita"


class VB_LogNormal_IndEncoder(nn.Module):
    def __init__(self,
            in_shape,              # [..., 2],
            out_shape,             # [..., 1]
            init_distr=[0.1, 0.1], # list of 2 floats, "uniform",
                                   # "normal" or False
            prior_hp=[0.2, 0.2],  # prior hyper-parameters
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 

        self.in_dim = self.in_shape[-1]   # 2
        self.out_dim = self.out_shape[-1] # 1

        self.init_distr = init_distr

        self.prior_hp = torch.tensor(prior_hp)
        self.device_ = device

        assert self.in_shape[-1] == 2       # last dim must be 2
        assert self.out_shape[-1] == 1       # last dim must be 2
        assert self.prior_hp.shape[-1] == 2  # for mu and sigma

        self.prior_mu = self.prior_hp[0]
        self.prior_sigma = self.prior_hp[1]

        # init parameters initialization
        if isinstance(self.init_distr, (list)):
            assert len(self.init_distr) == self.in_dim
            self.input = torch.tensor(self.init_distr)
        else:
            self.input = torch.ones(self.in_dim)

        if self.init_distr == "uniform":
            self.input = self.input.uniform_()
        elif self.init_distr == "normal":
            self.input = self.input.normal_()

        self.init_mu = self.input[0].repeat(
                [*self.in_shape[:-1], 1])
        self.init_log_sigma = torch.log(
                self.input[1].repeat([*self.in_shape[:-1], 1]))

        # Initialize the parameters of the variational distribution q
        self.mu = nn.Parameter(self.init_mu,
                requires_grad=True)
        self.log_sigma = nn.Parameter(self.init_log_sigma,
                requires_grad=True)

        # Prior distribution
        self.dist_p = LogNormal(self.prior_mu, self.prior_sigma)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=0.000001,
            max_clamp=False):

        # Approximate distribution
        self.dist_q = LogNormal(
                self.mu, 
                torch.exp(self.log_sigma))

        # Sample from approximate distribution q
        samples = self.dist_q.rsample(
                torch.Size([sample_size]))
        # print("samples shape {}".format(samples.shape)) # [sample_size]

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)

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


class VB_LogNormal_NNIndEncoder(nn.Module):
    def __init__(self,
            in_shape,              # [..., 2],
            out_shape,             # [..., 1]
            init_distr="uniform", # list of 2 floats, uniform,
                                  # normal or False
            prior_hp=[0.2, 0.2],
            h_dim=16, 
            nb_layers=3,
            bias_layers=True,     # True or False
            activ_layers="relu", # relu, tanh, or False
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 

        self.in_dim = self.in_shape[-1]   # 2
        self.out_dim = self.out_shape[-1] # 1

        #self.in_dim = 2
        #self.out_dim = 1            # one for mu and one for sigma
 
        self.init_distr = init_distr

        self.prior_hp = torch.tensor(prior_hp)
        self.device_ = device

        assert self.in_shape[-1] == 2       # last dim must be 2
        assert self.out_shape[-1] == 1       # last dim must be 2
        assert self.prior_hp.shape[-1] == 2  # for alpha and rate

        self.prior_mu = self.prior_hp[0]
        self.prior_sigma = self.prior_hp[1]
 
        self.h_dim = h_dim          # hidden layer size
        self.nb_layers = nb_layers
        self.bias_layers = bias_layers
        self.activ_layers = activ_layers

        if self.activ_layers == "relu":
            activation = nn.ReLU
        elif self.activ_layers == "tanh":
            activation = nn.Tanh

        if self.nb_layers < 2:
            self.nb_layers = 2
            print("The number of layers in {} should be >= 2."+\
                    " It's set set to 2".format(self))

        # Input of the variational neural network
        if isinstance(self.init_distr, (list)):
            assert len(self.init_distr) == self.in_dim
            self.net_input = torch.tensor(self.init_distr)
        else:
            self.net_input = torch.ones(self.in_dim)

        if self.init_distr == "uniform":
            self.net_input = self.net_input.uniform_()
        elif self.init_distr == "normal":
            self.net_input = self.net_input.normal_()

        self.input = self.input.repeat([*self.in_shape[:-1], 1])

        # Construct the neural network
        layers = [nn.Linear(self.in_dim, self.h_dim,
            bias=self.bias_layers)]
        if self.activ_layers: layers.append(activation())

        for i in range(1, self.nb_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=self.bias_layers)])
            if self.activ_layers: layers.append(activation())

        self.net = nn.Sequential(*layers)

        self.net_mu = nn.Sequential(
            nn.Linear(self.h_dim, self.out_dim))

        self.net_sigma = nn.Sequential(
            nn.Linear(self.h_dim, self.out_dim),
            nn.Softplus()) 

        # Prior distribution
        self.dist_p = LogNormal(self.prior_mu, self.prior_sigma)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=False,
            max_clamp=False):

        enc = self.net(self.net_input)
        mu = self.net_mu(enc) #.clamp(max=10.)
        sigma = self.net_sigma(enc) #.clamp(max=100.)

        # Approximate distribution
        self.dist_q = LogNormal(mu, sigma)

        # Sample
        #samples = sample_logNormal(mu, sigma, sample_size)
        samples = self.dist_q.rsample(
                torch.Size([sample_size]))
        # print("samples shape {}".format(samples.shape)) #

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)
 
        with torch.set_grad_enabled(KL_gradient):
            kl = kl_divergence(self.dist_q, self.dist_p)

        with torch.set_grad_enabled(not KL_gradient):
            # Compute log prior of samples p(d)
            logprior = self.dist_p.log_prob(samples)
            # print("logprior.shape {}".format(logprior.shape))

            # Compute the log of approximate posteriors q(d)
            logq = self.dist_q.log_prob(samples)
            # print("logq.shape {}".format(logq.shape))

        return logprior, logq, kl, samples