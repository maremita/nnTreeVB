from nnTreeVB.typing import dist_types
from nnTreeVB.utils import str2floats
from nnTreeVB.utils import str2values
from nnTreeVB.utils import getboolean
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

def check_subs_model(model_str):
    model = model_str.lower()

    if model not in ["jc69", "k80", "hky", "gtr"]:
        model = "jc69"
        print("\nSubstitution model should be"\
                " jc69|k80|hky|gtr,"\
                " not {}".format(model_str))
        print("Substitution model set to {}".format(model))

    return model

def check_sim_blengths(sim_blengths, nb_taxa, nb_rep=1):
    """
    Possible values:

    gamma(x,y)
    gamma(x,y)|False #False: same values for all replicates
    gamma(x,y)|True  #True: different values for each rep.
    gamma(x1,y1);gamma(x2,y2)
    uniform(x,y)
    uniform(x1,y1);uniform(x2,y2)
    exponential(x)
    exponential(x1);exponential(x2)
    
    Other distributions:
    normal, lognormal, dirichlet, categorical
    """

    nb_interns = nb_taxa - 3 # unrooted tree (2*nb_taxa -3)
    if nb_interns <0: nb_interns=0

    sim_blengths_str = sim_blengths.lower()

    # Use (False) or not (True) the same set of branch lens
    # for each replicate
    sim_reps = False

    re_vals = re.split("\|", sim_blengths_str.strip())

    if len(re_vals) >= 2:
        sim_blengths_str = re_vals[0]
        sim_reps = getboolean(re_vals[1])

    blen_dists = []
    dists_str = sim_blengths_str.split(";")

    for dist_str in dists_str:
        dist_split = dist_str.split("(")
        dist_name = dist_split[0]

        param_list = str2floats(
                re.split(dist_name+"\(|\)", dist_str)[1])

        if dist_name in torch_dist_names:
            blen_dists.append(build_torch_distribution(
                dist_name, param_list, dtype=torch.float64))
        else:
            raise ValueError("{} distribution name "\
                    "is not valid".format(dist_name))

    if len(blen_dists) == 1:
        # use the same distribution to sample internal edges
        blen_dists.append(blen_dists[0])

    with torch.no_grad():
        if sim_reps:
            # A set of nb_rep will be simulated
            values = torch.cat((blen_dists[0].sample([nb_rep,
                nb_taxa]), blen_dists[1].sample([nb_rep,
                    nb_interns])), dim=1).numpy()
        else:
            # Only one set of branches will be simulated
            values=torch.cat((blen_dists[0].sample([nb_taxa]),
                blen_dists[1].sample([nb_interns]))).numpy()

            values = np.resize(values, (nb_rep,
                nb_taxa+nb_interns))
    
    return values.tolist()

def check_sim_float(sim_float, nb_rep=1):
    """
    Possible values:
    
    a float or a distribution

    gamma(x,y)
    uniform(x,y)|False
    exponential(x)|True
 
    Other distributions:
    normal, lognormal, dirichlet, categorical

    """
    float_str = sim_float.lower()

    # Use (False) or not (True) the same set of values
    # for each replicate
    sim_reps = False

    re_vals = re.split("\|", float_str.strip())

    if len(re_vals) >= 2:
        float_str = re_vals[0]
        sim_reps = getboolean(re_vals[1])

    try:
        number = np.array([float(float_str)])
        values = np.resize(number, (nb_rep, 1))

    except ValueError as e:

        dist_split = float_str.split("(")
        dist_name = dist_split[0]
        param_list = str2floats(
                re.split(dist_name+"\(|\)", float_str)[1])

        if dist_name in torch_dist_names:
            torch_dist = build_torch_distribution(
                dist_name, param_list, dtype=torch.float64)

            with torch.no_grad():
                if sim_reps:
                    # A set of nb_rep will be simulated
                    values=torch_dist.sample([nb_rep]).numpy()
                else:
                    # Only one set of branches will be sim
                    values = torch_dist.sample().numpy()
                    values = np.resize(values,
                            (nb_rep, 1)).flatten()

        else:
            raise ValueError("{} distribution name "\
                    "is not valid".format(dist_name))
 
    return values.tolist()

def check_sim_simplex(sim_simplex, nb_params, nb_rep=1):
    simplex_str = sim_simplex.lower()

    # Use or not the same set of values for each replicate
    sim_reps = False

    re_vals = re.split("\|", simplex_str.strip())

    if len(re_vals) >= 2:
        simplex_str = re_vals[0]
        sim_reps = getboolean(re_vals[1])

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

        torch_dist = build_torch_distribution(
                dist_name, param_list, dtype=torch.float64)

        with torch.no_grad():
            if sim_reps:
                # A set of nb_rep will be simulated
                values = torch_dist.sample([nb_rep]).numpy()
            else:
                # Only one set of branches will be simulated
                values = torch_dist.sample().numpy()
                values = np.resize(values,(nb_rep, nb_params))

    else:
        values = np.array(
                str2values(simplex_str, nb_params, cast=float))
 
        if len(values) != nb_params:
            raise ValueError("[{}] lacks values".format(
                sim_simplex))

        values = np.resize(values, (nb_rep, nb_params))

    row_sums = values.sum(axis=1)
    values = values/row_sums[:, np.newaxis]

    assert np.isclose(values.sum(1), 1.).all()

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
    if transform in ["none", "false"]:
        return None
    elif transform == "lower_0":
        return torch.distributions.ExpTransform()
    elif transform == "simplex":
        return torch.distributions.StickBreakingTransform()
    else:
        raise ValueError("{} transform is not valide".format(
            dist_transform))

def check_prior_option(option_str):
    """
    option_str = "exponential|10.|False"
    """
    values = re.split("\|", option_str.strip())

    assert len(values) > 2

    dist = check_dist_type(values[0])
    params = check_dist_params(values[1])
    learn = getboolean(values[2])

    lr = False
    if len(values) > 3 and values[3].strip() != "":
        lr = float(values[3])

    return dist, params, learn, lr

def check_var_option(option_str):
    """
    option_str = "normal|0.1,0.1|lower_0"
    """
    values = re.split("\|", option_str.strip())

    assert len(values) > 2

    dist = check_dist_type(values[0])
    params = check_dist_params(values[1])
    transform = check_dist_transform(values[2])

    lr = False
    if len(values) > 3 and values[3].strip() != "":
        lr = float(values[3])

    return dist, params, transform, lr

def check_seed(seed):
    s = seed.lower()

    if s == "false":
        return None
    elif s == "none":
        return None
    else:
        try:
            s = int(seed)
            if s < 0:
                print("\nInvalid value for seef"\
                        " {}".format(seed))
                print("Valid values are: False, None"\
                        " and positive integers")
                print("Seed is set to None")
                return None
            # TODO Check the max value for seed
            else:
                return s
        except ValueError as e:
            print("\nInvalid value for seed {}".format(
                seed))
            print("Valid values are: False, None and"\
                    " positive integers")
            print("Seed is set to None")
            return None

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
            else:
                return v
        except ValueError as e:
            print("\nInvalid value for verbose {}".format(
                verbose))
            print("Valid values are: True, False, None and"\
                    " positive integers")
            print("Verbose is set to 0")
            return 0

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
