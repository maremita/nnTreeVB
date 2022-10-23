from typing import List, Union

import torch

__author__ = "Amine Remita"


TorchDistribution = torch.distributions.Distribution
TorchTransform = torch.distributions.transforms.Transform

dist_types = [
        "gamma",
        "gamma_nn",
        "lognormal",
        "lognormal_nn",
        "dirichlet",
        "dirichlet_nn",
        "dirichlet_nnx",
        "normal",
        "normal_nn",
        "exponential",
        "exponential_nn",
        "fixed"
        ]
