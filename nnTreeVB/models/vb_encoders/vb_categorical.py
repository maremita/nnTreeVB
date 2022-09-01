import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

__author__ = "amine remita"


class VB_Categorical_NNEncoder(nn.Module):
    def __init__(self, 
            in_shape,  # [n_dim, x_dim , m_dim]
            #in_dim,   # x_dim * m_dim
            #out_dim,  # x_dim * a_dim
            out_shape, # [n_dim, x_dim, a_dim]
            prior_hp=[1., 1.],
            h_dim=16,
            nb_layers=3,
            bias_layers=True,     # True or False
            activ_layers="relu", # relu, tanh, or False
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape 
        self.in_dim = self.in_shape[-1] * self.in_shape[-2] 
        self.out_dim = self.out_shape[-1] * self.out_shape[-2]

        self.prior_hp = torch.tensor(prior_hp)
        self.device_ = device

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
            bias=self.bias_layers), nn.LogSoftmax(-1)])

        self.net = nn.Sequential(*layers)
        
        # Prior distribution
        self.dist_p = Categorical(probs=self.prior_hp)

    def forward(
            self,
            data,
            sample_size=1,
            sample_temp=0.1,
            KL_gradient=False,
            min_clamp=False,    # should be <= to 10^-7
            max_clamp=False):
 
        # Flatten the data
        data = data.flatten(-2)
        #print("data shape {}".format(data.shape))
        # [n_dim, m_dim * x_dim]

        logit = self.net(data).view(*self.out_shape)

        # Approximate distribution
        self.dist_q = Categorical(logits=logit)

        # Sample
        samples = self.rsample(logit.expand(
            [sample_size, *self.out_shape]), temperature=sample_temp)
        #print("samples shape {}".format(samples.shape))
        # [sample_size, n_dim, a_dim, x_dim]

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)
 
        with torch.set_grad_enabled(KL_gradient):
            kl = kl_divergence(self.dist_q, self.dist_p)

        with torch.set_grad_enabled(not KL_gradient):
            # Compute log prior of samples p()
            logprior = self.dist_p.log_prob(samples)
            #print("logprior.shape {}".format(logprior.shape))

            # Compute the log of approximate posteriors q()
            logq = self.dist_q.log_prob(samples)
            #print("logq.shape {}".format(logq.shape))

        return logprior, logq, kl, samples

    #def sample_gs(self, logits, temperature=1, hard=False):
    #    # reparameterized sampling of categorical distribution
    #    return gumbel_softmax_sample(logits, temperature, hard)

    def rsample(self, logits, temperature=1):
        # Reparameterized sampling of discrete distribution
        U = torch.log(torch.rand(logits.shape) + 1e-20)
        #print("U shape {}".format(U.shape))
        #print(U)
        y = logits + U
        y = F.softmax(y/temperature, dim=-1)

        return y
