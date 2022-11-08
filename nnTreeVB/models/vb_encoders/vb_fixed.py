import torch
import torch.nn as nn
from torch.distributions import Distribution
from torch.distributions.kl import register_kl


__author__ = "Amine Remita"


class Fixed(Distribution):
    def __init__(self, batch_shape, data):
        super().__init__(batch_shape=batch_shape)
 
        self.data = data
        self.device_ = data.device

    def rsample(
            self, 
            sample_size=torch.Size([1])):

        samples = self.data.expand(
                [*list(sample_size), *self.batch_shape])
        #print("samples fixed shape {}".format(samples.shape))

        return samples

    def sample(
            self, 
            sample_size=torch.Size([1])):

        samples = self.data.expand(
                [*list(sample_size), *self.batch_shape])

        return samples
 
    def log_prob(self, samples):
        return torch.zeros(1).to(self.device_)


@register_kl(Fixed, Fixed)
def kl_fixed_fixed(p, q):
    return torch.zeros(1).to(p.device)


class VB_Fixed(nn.Module):
    def __init__(self,
            in_shape,              # [..., 6]
            out_shape,             # [..., 6]
            init_params=torch.ones(6),
            learn_params: bool = False,
            device=torch.device("cpu")):

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.data = init_params
        self.learn_params = False
        self.device_ = device

        if isinstance(self.data, list):
            self.data = torch.tensor(self.data)
        elif not isinstance(self.data, torch.Tensor):
            raise ValueError("{} data should be list or"\
                    " tensor in Fixed dist".format(self.data))

        if self.data.device != self.device_:
            self.data = self.data.to(self.device_)

    def forward(self): 
        self.dist = Fixed(torch.Size(self.out_shape), 
                self.data)

        return self.dist
