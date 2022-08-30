import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

__author__ = "Amine Remita"


class VB_Dirichlet_IndEncoder(nn.Module):
    def __init__(self,
            in_dim,              # in_shape[-1]
            in_shape,            # [2],
            init_distr=[1., 1.], # list of floats, "uniform",
                                 # "normal" or False
            prior_hp=[1., 1.]):

        super().__init__()

        self.in_dim = in_dim
        self.in_shape = in_shape
        self.init_distr = init_distr
        self.prior_hp = torch.tensor(prior_hp)

        assert self.prior_hp.shape[-1] == self.in_dim

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

        self.input = self.input.repeat([*self.in_shape[:-1], 1])

        self.init_log_alphas = torch.log(self.input)

        # Initialize the parameters of the variational distribution q
        self.log_alphas = nn.Parameter(self.init_log_alphas,
                requires_grad=True)

        # Prior distribution
        self.dist_p = Dirichlet(self.prior_hp)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=False,    # should be <= to 10^-7
            max_clamp=False):

        # Approximate distribution
        self.dist_q = Dirichlet(torch.exp(self.log_alphas))

        # Sample from approximate distribution q
        samples = self.dist_q.rsample(torch.Size([sample_size]))
        #print("samples dirichlet shape {}".format(samples.shape)) # [sample_size, 6]

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


class VB_Dirichlet_NNIndEncoder(nn.Module):
    def __init__(self,
            in_dim,               # in_shape[-1]
            in_shape,             # [2],
            init_distr=[1., 1.],  # list of floats, "uniform",
                                  # "normal" or False
            prior_hp=[1., 1.],
            h_dim=16, 
            nb_layers=3,
            bias_layers=True,     # True or False
            activ_layers="relu"): # relu, tanh, or False

        super().__init__()

        self.in_dim = in_dim 
        self.out_dim = in_dim
        self.in_shape = in_shape
        self.init_distr = init_distr
 
        self.prior_hp = torch.tensor(prior_hp)

        assert len(self.prior_hp) == self.in_dim

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
            self.input = torch.tensor(self.init_distr)
        else:
            self.input = torch.ones(self.in_dim)

        if self.init_distr == "uniform":
            self.input = self.input.uniform_()
        elif self.init_distr == "normal":
            self.input = self.input.normal_()

        self.input = self.input.repeat([*self.in_shape[:-1], 1])

        # Construct the neural network
        layers = [nn.Linear(self.in_dim, self.h_dim,
            bias=self.bias_layers)]
        if self.activ_layers: layers.append(activation())

        for i in range(1, self.nb_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=self.bias_layers)])
            if self.activ_layers: layers.append(activation())

        layers.extend([nn.Linear(self.h_dim, self.out_dim,
            bias=self.bias_layers), nn.Softplus()])

        self.net = nn.Sequential(*layers)

        # Prior distribution
        self.dist_p = Dirichlet(self.prior_hp)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False,
            min_clamp=False,    # should be <= to 10^-7
            max_clamp=False):

        alphas = self.net(self.input)

        # Approximate distribution
        # no need to reparameterize alpha
        self.dist_q = Dirichlet(alphas)

        # Sample
        samples = self.dist_q.rsample(torch.Size([sample_size]))
        # print("samples dirichlet deep shape {}".format(samples.shape)) # [sample_size, 6]
 
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


class VB_Dirichlet_NNEncoder(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim, 
            out_shape,
            prior_hp=[1., 1.],
            h_dim=16,
            nb_layers=3,
            bias_layers=True,     # True or False
            activ_layers="relu"): # relu, tanh, or False

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_shape = out_shape

        self.prior_hp = torch.tensor(prior_hp)

        self.h_dim = h_dim  # hidden layer size
        self.n_layers = n_layers
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

        # Construct the neural network
        layers = [nn.Linear(self.in_dim, self.h_dim,
            bias=self.bias_layers)]
        if self.activ_layers: layers.append(activation())

        for i in range(1, self.nb_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=self.bias_layers)])
            if self.activ_layers: layers.append(activation())

        layers.extend([nn.Linear(self.h_dim, self.out_dim,
            bias=self.bias_layers), nn.Softplus()])

        self.net = nn.Sequential(*layers)

        # Prior distribution
        self.dist_p = Dirichlet(probs=self.prior_hp)

    def forward(
            self,
            data,
            sample_size=1,
            sample_temp=1
            KL_gradient=False,
            min_clamp=False,    # should be <= to 10^-7
            max_clamp=False):

        # Flatten the data when passing it to this function
        #data = data.squeeze(0).flatten(0)
        #print("data_flatten.shape")
        #print(data.shape)  # [m_dim * x_dim]

        alphas = self.net(data).view(*self.out_shape)
        #print("alphas")
        #print(alphas.shape)
        #print(alphas)

        # Approximate distribution
        # no need to reparameterize alpha
        self.dist_q = Dirichlet(alphas) 

        # Sample
        samples = self.dist_q.rsample(torch.Size([sample_size]))
        # print("samples dirichlet deep shape {}".format(samples.shape)) # [sample_size, 6]
 
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
