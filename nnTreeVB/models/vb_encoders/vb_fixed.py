import torch
import torch.nn as nn

__author__ = "Amine Remita"

class VB_FixedEncoder(nn.Module):
    def __init__(self,
            in_shape,              # [..., 6]
            out_shape,             # [..., 6]
            init_distr=torch.ones(6),
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.data = init_distr
        self.device_ = device
 
        if self.data.device != self.device_:
            self.data = self.data.to(self.device_)

    def forward(
            self, 
            sample_size=1,
            KL_gradient=False):

        samples = self.data.expand(
                [sample_size, *self.out_shape])
        #print("samples fixed shape {}".format(samples.shape))

        logprior = torch.zeros(1).to(self.device_)
        logq = torch.zeros(1).to(self.device_)
        kl = torch.zeros(1).to(self.device_)

        return logprior, logq, kl, samples
