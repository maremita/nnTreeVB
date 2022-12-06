from nnTreeVB.utils import min_max_clamp
from nnTreeVB.checks import check_sample_size

import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence

__author__ = "Amine Remita"


class VB_Encoder(nn.Module):
    def __init__(self,
            dist_p,
            dist_q):

        super().__init__()

        self.dist_p = dist_p
        self.dist_q = dist_q

    def forward(
            self, 
            sample_size=torch.Size([1]),
            KL_gradient=False,
            min_clamp=False,
            max_clamp=False):

        sample_size = check_sample_size(sample_size)

        # Initialize distributions with updated params
        # if learn_params is True (forward pass)
        self.dist_p()
        self.dist_q()

        # Sample from approximate distribution q
        # in the constrained space
        samples = self.dist_q.dist.rsample(sample_size)
        #print("samples shape {}".format(samples.shape))

        samples = min_max_clamp(
                samples,
                min_clamp,
                max_clamp)

        with torch.set_grad_enabled(KL_gradient):
            nb_samples = list(sample_size)[0] 
            try:
                kl = kl_divergence(
                        self.dist_q.dist,
                        self.dist_p.dist)
                #print("kl.shape {}".format(kl.shape))
            except Exception as e:
                # Compute Monte Carlo based KL divergence

                # log of approximate posteriors
                _logq = self.dist_q.dist.log_prob(samples)

                # log prior of samples
                _logp = self.dist_p.dist.log_prob(samples)

                # Monte Carlo based KL divergence
                kl = (_logq - _logp).sum()/nb_samples

        with torch.set_grad_enabled(not KL_gradient):
            # Compute log prior of samples
            logp = self.dist_p.dist.log_prob(samples)
            #print("logp.shape {}".format(logp.shape))

            # Compute the log of approximate posteriors
            # If transformed distribution, the log_abs_det_jac
            # will be added in log_prob
            logq = self.dist_q.dist.log_prob(samples)
            #print("logq.shape {}".format(logq.shape))

        return logp, logq, kl, samples
