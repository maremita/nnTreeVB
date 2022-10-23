from nnTreeVB.typing import dist_types
from nnTreeVB.utils import str2floats
from nnTreeVB.utils import str2values
from nnTreeVB.models.torch_distributions import\
        build_torch_distribution, torch_dist_names

import re

import numpy as np
import torch
import torch.nn as nn

__author__ = "amine remita"

""" 
Check funcitons
"""

def check_sim_blengths(sim_blengths):
    """
    Possible values:

    gamma(x,y)
    gamma(x1,y1);gamma(x2,y2)
    uniform(x,y)
    uniform(x1,y1);uniform(x2,y2)
    exponential(x)
    exponential(x1);exponential(x2)
    
    Other distributions:
    normal, lognormal, dirichlet, categorical
    """

    blen_dists = []
    str_dists = sim_blengths.lower().split(";")

    for str_dist in str_dists:
        dist_split = str_dist.split("(")
        dist_name = dist_split[0]
        #param_list = str2floats(dist_split[1].strip(")"))

        param_list = str2floats(
                re.split(dist_name+"\(|\)", str_dist)[1])

        if dist_name in torch_dist_names:
            blen_dists.append(build_torch_distribution(
                dist_name, param_list))
        else:
            raise ValueError("{} distribution name "\
                    "is not valid".format(dist_name))

    return blen_dists

def check_sim_float(sim_float):
    """
    Possible values:
    
    a float or a distribution

    gamma(x,y)
    uniform(x,y)
    exponential(x)
 
    Other distributions:
    normal, lognormal, dirichlet, categorical

    """

    try:
        return float(sim_float)

    except ValueError as e:
        float_str = sim_float.lower()

        dist_split = float_str.split("(")
        dist_name = dist_split[0]
        param_list = str2floats(
                re.split(dist_name+"\(|\)", float_str)[1])

        if dist_name in torch_dist_names:
            torch_dist = build_torch_distribution(
                dist_name, param_list)

            with torch.no_grad():
                return torch_dist.sample().item()

        else:
            raise ValueError("{} distribution name "\
                    "is not valid".format(dist_name))

def check_sim_simplex(sim_simplex, nb_params):
    simplex_str = sim_simplex.lower()

    # for now, the program accepts only dirichlet 
    dist_name = "dirichlet"

    if dist_name in simplex_str:
        param_list = str2floats(
                re.split(dist_name+"\(|\)", simplex_str)[1])
        
        # dirichlet(1.)
        if len(param_list) == 1:
            param_list = [param_list[0]] * nb_params
        elif len(param_list) != nb_params:
            raise ValueError("{} lacks parameters".format(
                sim_simplex))

        rates_dist = build_torch_distribution(
                dist_name, param_list)
        
        with torch.no_grad():
            values = rates_dist.sample().numpy()

    else:
        values = np.array(
                str2values(simplex_str, nb_params, cast=float))
        
        if len(values) != nb_params:
            raise ValueError("[{}] lacks values".format(
                sim_simplex))

    values = values/values.sum()

    assert np.isclose(values.sum(), 1.)

    return values.tolist()

def check_dist_type(dist_type):
    dist = dist_type.lower()

    if not dist in dist_types:
        raise ValueError("{} distribution is not"\
                " supported".format(dist_type))
    return dist

def check_dist_params(dist_params):
    params = dist_params.lower()

    if params == "uniform":
        return params
    elif params == "normal":
        return params
    elif params == "false":
        return False
    elif params == "true":
        return True
    elif params == "none":
        return None
    else:
        return str2floats(params, sep=",")

def check_dist_transform(dist_transform):
    transform = dist_transform.lower()
    
    # TODO implement other transofmations
    if transform == "none":
        return None
    elif transform == "lower_0":
        return torch.distributions.ExpTransform()
    elif transform == "simplex":
        return torch.distributions.StickBreakingTransform()
    else:
        raise ValueError("{} transform is not valide".format(
            dist_transform))

def check_verbose(verbose):
    v = verbose.lower()

    if v == "false":
        return 0
    elif v == "none":
        return 0
    elif v == "true":
        return 1
    else:
        try:
            v = int(verbose)
            if v < 0:
                print("\nInvalid value for verbose"\
                        " {}".format(verbose))
                print("Valid values are: True, False, None"\
                        " and positive integers")
                print("Verbose is set to 0")
                return 0
        except ValueError as e:
            print("\nInvalid value for verbose {}".format(
                verbose))
            print("Valid values are: True, False, None and"\
                    " positive integers")
            print("Verbose is set to 1")
            return 1

def check_sample_size(sample_size):
    
    if isinstance(sample_size, torch.Size):
        return sample_size

    if isinstance(sample_size, int):
        return torch.Size([sample_size])

    elif isinstance(sample_size, list):
        return torch.Size(sample_size)

    else:
        raise ValueError("Sample size type is not valid")

def check_finite_grads(model, epoch, verbose=False):

    finite = True
    for name, param in model.named_parameters():
        if param.grad is None or\
                not torch.isfinite(param.grad).all():
            finite = False

            if verbose:
                print("{} Nonfinit grad {} : {}".format(epoch,
                    name, param.grad))
            else:
                return finite

    if not finite and verbose: print()
    return finite
