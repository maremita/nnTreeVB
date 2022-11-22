from nnTreeVB.data import SeqCollection 

import copy
import time
import math
import platform
import importlib
import configparser
from pprint import pformat
import itertools
import pickle
import gzip

import numpy as np
from joblib import Parallel, delayed
import scipy.stats
from scipy.stats.stats import pearsonr#, spearmanr

import torch
import torch.nn as nn

def _GZcompressed(file_name):
    # https://stackoverflow.com/a/13044946
    # https://www.garykessler.net/library/file_sigs.html

    gz_magic = b"\x1f\x8b\x08"
    with open(file_name, "rb") as fh:
        file_start = fh.read(len(gz_magic))

    if file_start.startswith(gz_magic):
        return True
    else:
        return False

def dump(data, file_name, compress=False):
 
    if compress:
        open_fun = gzip.open
    else:
        open_fun = open

    with open_fun(file_name, "wb") as fh:
        pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)

def load(file_name):

    if _GZcompressed(file_name):
        open_fun = gzip.open
    else:
        open_fun = open

    with open_fun(file_name, "rb") as fh:
        data = pickle.load(fh)

    return data

def update_sim_parameters(obj):
    """
    obj is an object with attributes:
        nb_rep_data
        subs_model
        sim_blengths (not used here)
        sim_rates
        sim_freqs
        sim_kappa
    """

    nb_data = obj.nb_rep_data # nb of replicates 

    assert len(obj.sim_rates) == nb_data
    assert len(obj.sim_freqs) == nb_data
    assert len(obj.sim_kappa) == nb_data

    # Update frequencies
    if obj.subs_model in ["jc69", "k80"]:
        obj.sim_freqs = (np.ones((nb_data, 4))/4).tolist()

    # Update rates
    if obj.subs_model == "jc69":
        obj.sim_rates = (np.ones((nb_data,6))/6).tolist()

    elif obj.subs_model in ["k80", "hky"]:
        obj.sim_rates = [compute_rates_from_kappa(
                k).tolist() for k in obj.sim_kappa]

    # Update kappa if model is jc69 or gtr
    #(for information purpose, won't be used)
    if obj.subs_model in ["jc69", "gtr"]:
        obj.sim_kappa = [compute_kappa_from_rates(r) for r in\
                obj.sim_rates]

def compute_rates_from_kappa(kappa):
        """
        "AG", "AC", "AT", "GC", "GT", "CT"
        Multiply AG and CT transition rates by kappa
        See  The Phylogenetic_Handbook page 131
        (Lemey, Salemi, Vandamme 2009)
        """

        if not isinstance(kappa, np.ndarray):
            kappa = np.array(kappa)

        rates = np.hstack((kappa*1., (np.ones(4)), kappa*1.))

        return rates/rates.sum()

def compute_kappa_from_rates(r):
    return (r[0]+r[-1])/2/((sum(r[1:-1])/4))

def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False

def build_neuralnet(
        in_dim,
        out_dim,
        h_dim,
        nb_layers,
        bias_layers,
        activ_layers,
        dropout,
        last_layers, #nn.Softplus() for example
        device):

    if activ_layers == "relu":
        activation = nn.ReLU
    elif activ_layers == "tanh":
        activation = nn.Tanh
    else:
        activ_layers = False

    if nb_layers < 2:
        nb_layers = 2
        print("The number of layers in {} should"\
                " be >= 2. It's set set to 2".format(self))

    assert 0. <= dropout <= 1.

    # Construct the neural network
    layers = [nn.Linear(in_dim, h_dim,
        bias=bias_layers).to(device)]
    if activ_layers: layers.append(activation())
    if dropout: layers.append(
            nn.Dropout(p=dropout))

    for i in range(1, nb_layers-1):
        layers.extend([nn.Linear(h_dim, h_dim,
            bias=bias_layers).to(device)])
        if activ_layers: layers.append(activation())
        if dropout: layers.append(
                nn.Dropout(p=dropout))

    layers.extend([nn.Linear(h_dim, out_dim,
        bias=bias_layers).to(device)])

    if last_layers is not None:
        layers.extend([last_layers])

    return nn.Sequential(*layers)

def init_parameters(init_params, nb_params):
    if isinstance(init_params, (list)):
        # For example if freqs are represented by dirichlet(1)
        # we replicate init_params into 4 params
        # so we'll have dirichlet(1, 1, 1, 1)
        if len(init_params) == 1 and nb_params > 1:
            init_params = [init_params[0]] * nb_params

        assert len(init_params) == nb_params
        init_input = torch.tensor(init_params)
    else:
        init_input = torch.ones(nb_params)

    if init_params == "uniform":
        init_input = init_input.uniform_()
    elif init_params == "normal":
        init_input = init_input.normal_()

    return init_input

def sum_kls(kls, device=torch.device("cpu")):
    sumkls = torch.zeros(1).to(device)

    for kl in kls:
        sumkls += kl.sum()

    return sumkls

def sum_log_probs(log_probs, sample_size, sum_by_samples=True,
        device=torch.device("cpu")):

    nb_s_dim = len(sample_size)

    if sum_by_samples:
        joint = torch.zeros(sample_size).to(device)
    else:
        joint = torch.zeros(1).to(device)

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

def str2tensor(chaine, sep=","):
    values = [float(v) for v in chaine.strip().split(sep)]
    return torch.FloatTensor(values)

def str2ints(chaine, sep=","):
    return [int(s) for s in chaine.strip().split(sep)]

def str2floats(chaine, sep=","):
    return [float(s) for s in chaine.strip().split(sep)]

def str2list(chaine, sep=",", cast=None):
    c = lambda x: x
    if cast: c = cast

    return [c(i.strip()) for i in chaine.strip().split(sep)]

def str2values(chaine, nb_repeat=1, sep=",", cast=None):
    chaine = chaine.rstrip(sep)
    values = str2list(chaine, sep=sep, cast=cast)
    if len(values)==1 : values = values * nb_repeat

    return values

def fasta2list(fasta_file, verbose=False):
    # fetch sequences from fasta
    if verbose: print("Fetching sequences from {}".format(
        fasta_file))
    seqRec_list = SeqCollection.read_bio_file(fasta_file)
    return [str(seqRec.seq._data) for seqRec in seqRec_list] 

def dictLists2combinations(data):
    keys, values = zip(*data.items())
    return [tuple(zip(keys, v)) for v in\
            itertools.product(*values)]

def get_lognorm_params(m, s):
    # to produce a distribution with desired mean m
    # and standard deviation s
    mu = np.log(m**2) - np.log(m) - ((np.log(s**2 + m**2)\
            + np.log(m**2))/2)
    std = np.sqrt(np.log(s**2 + m**2) - np.log(m**2))
    return [mu, std]

def get_all_grad_stats(module):
    grads = []
    for param in module.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads).detach().cpu().numpy()

    return {"mean":np.nanmean(grads), 
            "var": np.nanvar(grads),
            "min": np.nanmin(grads),
            "max": np.nanmax(grads)}

def get_all_weight_stats(module):
    poids = []
    for param in module.parameters():
        poids.append(param.data.view(-1))
    poids = torch.cat(poids).detach().cpu().numpy()

    return {"mean": np.nanmean(poids),
            "var": np.nanvar(poids),
            "min": np.nanmin(poids),
            "max": np.nanmax(poids)}

def get_named_grad_stats(module):
    grads = dict()
    stats = dict()
    for name, param in module.named_parameters():
        # If module has a network, concatenate the gradients
        # of the whole network
        name = name.split(".")[0]

        if param.grad is not None:
            if not name in grads: grads[name] = []
            grads[name].append(param.grad.view(-1))

    for name in grads:
        grads[name] = torch.cat(
                grads[name]).detach().cpu().numpy()
        stats[name] = compute_estim_stats(grads[name],
                stats=["mean", "var"])

    return stats

def get_named_weight_stats(module):
    poids = dict()
    stats = dict()
    for name, param in module.named_parameters():
        # If module has a network, concatenate the gradients
        # of the whole network
        name = name.split(".")[0]

        if param.data is not None:
            if not name in poids: poids[name] = []
            poids[name].append(param.data.view(-1))

    for name in poids:
        poids[name] = torch.cat(
                poids[name]).detach().cpu().numpy()
        stats[name] = compute_estim_stats(poids[name],
                stats=["mean", "var"])

    return stats

def get_grad_list(module):
    grads = []
    for param in module.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))

    if len(grads) > 0:
        grads = torch.cat(grads)

    return grads

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
        #if len(get_weight_list(sub_module))>0:
        #TODO: Try another way to check the type 
        # values to be collected (grads or weigths)
        if len(get_grad_list(sub_module))>0\
                and "_dist_" in name:
            ret[name]=func(sub_module)

    return ret

def compute_corr(main, batch, verbose=False):

    def pearson(v1, v2):
        r = np.nan 
        if np.isfinite(v1).all() and np.isfinite(v2).all():
            r = pearsonr(v1, v2)[0]
            #r = spearmanr(v1, v2)[0]
        return r 

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

    m, se = np.mean(a, axis=axis), scipy.stats.sem(a,
            axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return m, m-h, m+h

def compute_estim_stats(
        sample,
        confidence=0.95,
        axis=0,
        stats=["mean","ci_min","ci_max","var","min","max"]):

    res_stats = dict()

    mean, ci_min, ci_max = mean_confidence_interval(sample,
            confidence, axis)
    
    if "mean" in stats: res_stats["mean"] = mean

    if "ci_min" in stats: res_stats["ci_min"] = ci_min

    if "ci_max" in stats: res_stats["ci_max"] = ci_max

    if "var" in stats:
        res_stats["var"] = np.var(sample, axis=axis)
    
    if "min" in stats:
        res_stats["min"] = np.min(sample, axis=axis)
    
    if "max" in stats:
        res_stats["max"] = np.max(sample, axis=axis)

    return res_stats

def dict_to_cpu(some_dict):
    new_dict = dict()

    for key in some_dict:
        new_dict[key] = some_dict[key].cpu()

    return new_dict

def dict_to_numpy(some_dict):
    new_dict = dict()

    for key in some_dict:
        if isinstance(some_dict[key], torch.Tensor):
            new_dict[key] = some_dict[
                    key].cpu().detach().numpy()
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

def dict_to_tensor(
        some_dict,
        device=torch.device("cpu"),
        dtype=torch.float32):
    new_dict = dict()

    for key in some_dict:
        if isinstance(some_dict[key], (list, np.ndarray)):
            new_dict[key] = torch.tensor(some_dict[key],
                    device=device, dtype=dtype)
        elif isinstance(some_dict[key], (int, float)):
            new_dict[key] = torch.tensor([some_dict[key]],
                    device=device, dtype=dtype)
        elif isinstance(some_dict[key], torch.Tensor):
            new_dict[key] = (some_dict[key]).to(device=device,
                    dtype=dtype)
        else:
            raise ValueError("{} in dict_to_tensor() should"\
                    " be tensor, array, list, int or float")

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
            "ete3"]

    for module_name in module_names:
        found = importlib.util.find_spec(module_name)
        if found:
            module = importlib.import_module(module_name)
            versions[module_name] = module.__version__
        else:
            versions[module_name] = "Not found"

    return versions
