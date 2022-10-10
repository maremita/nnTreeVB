from nnTreeVB.data import SeqCollection 

import time
import math
import platform
import importlib
import configparser
from pprint import pformat

import numpy as np
import torch
from joblib import Parallel, delayed
import scipy.stats
from scipy.stats.stats import pearsonr#, spearmanr

__author__ = "amine remita"


def check_sample_size(sample_size):
    
    if isinstance(sample_size, torch.Size):
        return sample_size

    if isinstance(sample_size, int):
        return torch.Size([sample_size])

    elif isinstance(sample_size, list):
        return torch.Size(sample_size)

    else:
        raise ValueError("Sample size type is not valid")


def sum_kls(kls):
    sumkls = torch.zeros(1)

    for kl in kls:
        sumkls += kl.sum()

    return sumkls


def sum_log_probs(log_probs, sample_size, sum_by_samples=True):

    nb_s_dim = len(sample_size)

    if sum_by_samples:
        joint = torch.zeros(sample_size)
    else:
        joint = torch.zeros(1)

    for log_p in log_probs:
        # Sum log probs by samples
        if sum_by_samples:
            if len(log_p.shape) <= nb_s_dim:
                # No batch
                joint += log_p
            else:
                # Sum first by batch
                joint += log_p.sum(-1)

        # Sum log probs independentely from samples 
        else:
            joint += log_p.mean(0).sum(0)

    return joint


def min_max_clamp(x, min_clamp=False, max_clamp=False):
    if not isinstance(min_clamp, bool):
        if isinstance(min_clamp, (float, int)):
            x = x.clamp(min=min_clamp)

    if not isinstance(max_clamp, bool):
        if isinstance(max_clamp, (float, int)):
            x = x.clamp(max=max_clamp)

    return x


def getboolean(value):
    return configparser.RawConfigParser()._convert_to_boolean(
            value)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def get_categorical_prior(conf, prior_type, verbose=False):
    priors = []

    if prior_type in ["ancestor", "freqs"]:
        nb_categories = 4
    elif prior_type == "rates":
        nb_categories = 6
    else:
        raise ValueError(
                "prior type value should be ancestor, freqs or rates")

    if conf == "uniform":
        priors = torch.ones(nb_categories)/nb_categories
    elif "," in conf:
        priors = str2float_tensor(conf, ',', nb_categoriesi,
                prior_type)
    #elif conf == "empirical": # to be implemented
    #    pass
    else:
        raise ValueError(
                "Check {} prior config values".format(prior_type))

    if verbose:
        print("{} prior hyper-parameters: {}".format(
            prior_type, priors))

    return priors

def get_branch_prior(conf, verbose=False):
    priors = str2float_tensor(conf, ",", 2, "branch")

    if verbose:
        print("Branch prior hyper-parameters: {}".format(priors))

    return priors 

def get_kappa_prior(conf, verbose=False):
    priors = str2float_tensor(conf, ",", 2, "kappa")

    if verbose:
        print("Kappa prior hyper-parameters: {}".format(priors))

    return priors 

def str2float_tensor(chaine, sep, nb_values, prior_type):
    values = [float(v) for v in chaine.strip().split(sep)]
    if len(values) != nb_values:
        raise ValueError(
                "the Number of prior values for {} "\
                        "is not correct".format(prior_type))
    return torch.FloatTensor(values)

def str2ints(chaine, sep=","):
    return [int(s) for s in chaine.strip().split(sep)]

def str2floats(chaine, sep=","):
    return [float(s) for s in chaine.strip().split(sep)]

def fasta_to_list(fasta_file, verbose=False):
    # fetch sequences from fasta
    if verbose: print("Fetching sequences from {}".format(fasta_file))
    seqRec_list = SeqCollection.read_bio_file(fasta_file)
    return [str(seqRec.seq._data) for seqRec in seqRec_list] 

def str_to_list(chaine, sep=",", cast=None):
    c = lambda x: x
    if cast: c = cast

    return [c(i.strip()) for i in chaine.strip().split(sep)]

def str_to_values(chaine, nb_repeat=1, sep=",", cast=None):
    chaine = chaine.rstrip(sep)
    values = str_to_list(chaine, sep=sep, cast=cast)
    if len(values)==1 : values = values * nb_repeat

    return values


def get_lognorm_params(m, s):
    # to produce a distribution with desired mean m
    # and standard deviation s
    mu = np.log(m**2) - np.log(m) - ((np.log(s**2 + m**2)\
            + np.log(m**2))/2)
    std = np.sqrt(np.log(s**2 + m**2) - np.log(m**2))
    return [mu, std]

def get_grad_stats(module):
    grads = []
    for param in module.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads).detach().cpu().numpy()
    return {"mean":np.nanmean(grads), 
            "var": np.nanvar(grads),
            "min": np.nanmin(grads),
            "max": np.nanmax(grads)}

def get_grad_list(module):
    grads = []
    for param in module.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))

    if len(grads) > 0:
        grads = torch.cat(grads)

    return grads

def get_weight_stats(module):
    poids = []
    for param in module.parameters():
        poids.append(param.data.view(-1))
    poids = torch.cat(poids).detach().cpu().numpy()
    return {"mean": np.nanmean(poids),
            "var": np.nanvar(poids),
            "min": np.nanmin(poids),
            "max": np.nanmax(poids)}

def get_weight_list(module):
    poids = []
    for param in module.parameters():
        poids.append(param.data.view(-1))

    if len(poids) > 0:
        poids = torch.cat(poids)
    return poids

def apply_on_submodules(func, nn_module):
    ret = dict()
    for name, sub_module in nn_module.named_children():
        if len(get_weight_list(sub_module))>0:
            ret[name]=func(sub_module)

    return ret

def compute_corr(main, batch, verbose=False):

    def pearson(v1, v2):
        return pearsonr(v1, v2)[0]
        #return spearmanr(v1, v2)[0]

    nb_reps, nb_epochs, shape = batch.shape

    parallel = Parallel(prefer="processes", verbose=verbose)

    corrs = np.zeros((nb_reps, nb_epochs))

    for i in range(nb_reps):
        pears = parallel(delayed(pearson)(main , batch[i, j]) 
                for j in range(nb_epochs))
        corrs[i] = np.array(pears)

    return corrs


def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = 1.0 * np.array(data)
    n = np.size(a, axis=axis)

    m, se = np.mean(a, axis=axis), scipy.stats.sem(a, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return m, m-h, m+h


def check_finite_grads(model, epoch, verbose=False):

    finit = True
    for name, param in model.named_parameters():
        if not torch.isfinite(param.grad).all():
            finit = False

            if verbose:
                print("{} Nonfinit grad {} : {}".format(epoch,
                    name, param.grad))
            else:
                return finit

    if not finit and verbose: print()
    return finit

def dict_to_cpu(some_dict):
    new_dict = dict()

    for key in some_dict:
        new_dict[key] = some_dict[key].cpu()

    return new_dict

def dict_to_numpy(some_dict):
    new_dict = dict()

    for key in some_dict:
        if isinstance(some_dict[key], torch.Tensor):
            new_dict[key] = some_dict[key].cpu().detach().numpy()
        elif isinstance(some_dict[key], list):
            new_dict[key] = np.array(some_dict[key])
        elif isinstance(some_dict[key], (int, float)):
            new_dict[key] = np.array([some_dict[key]])
        elif isinstance(some_dict[key], np.ndarray):
            new_dict[key] = some_dict[key]
        else:
            raise ValueError("{} in dict_to_numpy() should be"\
                    " tensor, array, list, int or float")

    return new_dict

def write_conf_packages(args, out_file):

    with open(out_file, "wt") as f:
        f.write("\n# Program arguments\n# #################\n\n")
        args.write(f)

        f.write("\n# Package versions\n# ################\n\n")
        modules = pformat(get_modules_versions())
        f.write( "#" + modules.replace("\n", "\n#"))

def get_modules_versions():
    versions = dict()

    versions["python"] = platform.python_version()

    module_names = ["nnTreeVB", "numpy", "scipy", "pandas",
            "torch", "Bio", "joblib", "matplotlib", "pyvolve",
            "seaborn", "ete3"]

    for module_name in module_names:
        found = importlib.util.find_spec(module_name)
        if found:
            module = importlib.import_module(module_name)
            versions[module_name] = module.__version__
        else:
            versions[module_name] = "Not found"

    return versions

