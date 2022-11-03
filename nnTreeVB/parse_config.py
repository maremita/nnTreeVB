from dataclasses import dataclass
import configparser

from nnTreeVB.checks import check_sim_blengths
from nnTreeVB.checks import check_sim_simplex
from nnTreeVB.checks import check_sim_float
#from nnTreeVB.checks import check_seed
from nnTreeVB.checks import check_verbose
from nnTreeVB.checks import check_dist_type
from nnTreeVB.checks import check_dist_params
from nnTreeVB.checks import check_dist_transform
from nnTreeVB.utils import getboolean

import re

@dataclass
class ArgObject:
    """
    A data class object to hold config attributes
    """
    def to_dict(self):
        return self.__dict__


def parse_config(config_file):
    arg = ArgObject()
    config = configparser.ConfigParser(
            interpolation=\
                    configparser.ExtendedInterpolation())

    with open(config_file, "r") as cf:
        config.read_file(cf)


    # IO files
    arg.io = ArgObject()
    arg.io.output_path = config.get("io", "output_path")
 
    arg.io.seq_file = config.get("io", "seq_file",
            fallback="")
    arg.io.nwk_file = config.get("io", "nwk_file",
            fallback="")

    arg.io.scores_from_file = config.getboolean("io",
            "scores_from_file", fallback=False)

    # Simulation data
    arg.sim = ArgObject()
    arg.sim.sim_data = config.getboolean(
            "sim_data", "sim_data", fallback=True)
    arg.sim.seq_from_file = config.getboolean(
            "sim_data", "seq_from_file", fallback=True)
    arg.sim.nwk_from_file = config.getboolean(
            "sim_data", "nwk_from_file", fallback=True)
    arg.sim.nb_sites = config.getint(
            "sim_data", "nb_sites", fallback=100)
    arg.sim.nb_taxa = config.getint(
            "sim_data", "nb_taxa", fallback=100)
    arg.sim.subs_model = config.get(
            "sim_data", "subs_model", fallback="jc69")
    arg.sim.sim_blengths = check_sim_blengths(config.get(
        "sim_data", "sim_blengths", fallback="0.1,1."))
    arg.sim.sim_rates = check_sim_simplex(config.get(
        "sim_data", "sim_rates", fallback="0.16"), 6)
    arg.sim.sim_freqs = check_sim_simplex(config.get(
        "sim_data", "sim_freqs", fallback="0.25"), 4)
    arg.sim.sim_kappa = check_sim_float(config.get(
        "sim_data", "sim_kappa", fallback="1."))

    # Hyper parameters
    arg.mdl = ArgObject()
    # Evo variational model type
    arg.mdl.subs_model = config.get(
            "hyperparams", "subs_model", fallback="jc69")

    # Hyper-parameters of prior distributions
    # Branches
    b_prior = re.split("\|", config.get("hyperparams",
        "b_prior", fallback="exponential(10.)False").strip())

    arg.mdl.b_prior_dist = check_dist_type(b_prior[0])
    arg.mdl.b_prior_params = check_dist_params(b_prior[1])
    arg.mdl.b_learn_prior = getboolean(b_prior[2])

    b_var = re.split("\|", config.get("hyperparams",
        "b_var", fallback="normal(0.1,0.1)lower_0").strip())

    arg.mdl.b_var_dist = check_dist_type(b_var[0])
    arg.mdl.b_var_params = check_dist_params(b_var[1])
    arg.mdl.b_var_transform = check_dist_transform(b_var[2])

    ## Tree length
    t_prior = re.split("\|", config.get("hyperparams",
        "t_prior", fallback="gamma(1.,1.)False").strip())

    arg.mdl.t_prior_dist = check_dist_type(t_prior[0])
    arg.mdl.t_prior_params = check_dist_params(t_prior[1])
    arg.mdl.t_learn_prior = getboolean(t_prior[2])

    t_var = re.split("\|", config.get("hyperparams",
        "t_var", fallback="normal(0.1,0.1)lower_0").strip())

    arg.mdl.t_var_dist = check_dist_type(t_var[0])
    arg.mdl.t_var_params = check_dist_params(t_var[1])
    arg.mdl.t_var_transform = check_dist_transform(t_var[2])

    ## Rates
    r_prior = re.split("\|", config.get("hyperparams",
        "r_prior", fallback="dirichlet(uniform)False").strip())

    arg.mdl.r_prior_dist = check_dist_type(r_prior[0])
    arg.mdl.r_prior_params = check_dist_params(r_prior[1])
    arg.mdl.r_learn_prior = getboolean(r_prior[2])

    r_var = re.split("\|", config.get("hyperparams",
        "r_var", fallback="normal(0.1,0.1)simplex").strip())

    arg.mdl.r_var_dist = check_dist_type(r_var[0])
    arg.mdl.r_var_params = check_dist_params(r_var[1])
    arg.mdl.r_var_transform = check_dist_transform(r_var[2])

    ## Frequencies
    f_prior = re.split("\|", config.get("hyperparams",
        "f_prior", fallback="dirichlet(uniform)False").strip())

    arg.mdl.f_prior_dist = check_dist_type(f_prior[0])
    arg.mdl.f_prior_params = check_dist_params(f_prior[1])
    arg.mdl.f_learn_prior = getboolean(f_prior[2])

    f_var = re.split("\|", config.get("hyperparams",
        "f_var", fallback="normal(0.1,0.1)simplex").strip())

    arg.mdl.f_var_dist = check_dist_type(f_var[0])
    arg.mdl.f_var_params = check_dist_params(f_var[1])
    arg.mdl.f_var_transform = check_dist_transform(f_var[2])

    ## Kappa
    k_prior = re.split("\|", config.get("hyperparams",
        "k_prior", fallback="gamma(1.,1.)False").strip())

    arg.mdl.k_prior_dist = check_dist_type(k_prior[0])
    arg.mdl.k_prior_params = check_dist_params(k_prior[1])
    arg.mdl.k_learn_prior = getboolean(k_prior[2])

    k_var = re.split("\|", config.get("hyperparams",
        "k_var", fallback="normal(0.1,0.1)lower_0").strip())

    arg.mdl.k_var_dist = check_dist_type(k_var[0])
    arg.mdl.k_var_params = check_dist_params(k_var[1])
    arg.mdl.k_var_transform = check_dist_transform(k_var[2])

    # Neural net hyperparameters
    arg.mdl.h_dim = config.getint("hyperparams",
            "h_dim", fallback=16)
    arg.mdl.nb_layers = config.getint("hyperparams",
            "nb_layers", fallback=3)
    arg.mdl.bias_layers = config.getboolean("hyperparams",
            "bias_layers", fallback=True)
    arg.mdl.activ_layers = config.get("hyperparams",
            "activ_layers", fallback="relu")
    arg.mdl.dropout_layers = config.getfloat("hyperparams",
            "dropout_layers", fallback=0.)
 
    # Fitting hyperparameters
    arg.fit = ArgObject()
    arg.fit.nb_replicates = config.getint(
            "hyperparams", "nb_replicates", fallback=2)
    arg.fit.elbo_type = config.get(
            "hyperparams", "elbo_type", fallback="elbo")
    arg.fit.grad_samples = config.getint(
            "hyperparams", "grad_samples", fallback=1)
    arg.fit.K_grad_samples = config.getint(
            "hyperparams", "K_grad_samples", fallback=0)
    arg.fit.nb_samples = config.getint(
            "hyperparams", "nb_samples", fallback=10)
    arg.fit.alpha_kl = config.getfloat(
            "hyperparams", "alpha_kl", fallback=0.0001)
    arg.fit.max_iter = config.getint(
            "hyperparams", "max_iter", fallback=100)
    arg.fit.optimizer = config.get(
            "hyperparams", "optimizer", fallback="adam")
    arg.fit.learning_rate = config.getfloat(
            "hyperparams", "learning_rate", fallback=0.005)
    arg.fit.weight_decay = config.getfloat(
            "hyperparams", "weight_decay", fallback=0.00001)
    arg.fit.save_fit_history = config.getboolean(
            "hyperparams", "save_fit_history", fallback=False)
    arg.fit.save_val_history = config.getboolean(
            "hyperparams", "save_val_history", fallback=False)
    arg.fit.save_grad_stats = config.getboolean(
            "hyperparams", "save_grad_stats", fallback=False)
    arg.fit.save_weight_stats = config.getboolean(
            "hyperparams", "save_weight_stats", fallback=False)

    # setting parameters
    arg.stg = ArgObject()
    arg.stg.job_name = config.get("settings", "job_name",
            fallback=None)
    #arg.stg.seed = check_seed(config.get("settings", "seed",
    #        fallback=None))
    arg.stg.device = config.get("settings", "device",
            fallback="cpu")
    arg.stg.verbose = check_verbose(config.get("settings", 
        "verbose", fallback=1))
    arg.stg.compress_files = config.getboolean("settings", 
            "compress_files", fallback=False)

    # plotting settings
    arg.plt = ArgObject()
    arg.plt.size_font = config.getint("plotting", "size_font",
            fallback=16)
    arg.plt.plt_usetex = config.getboolean("plotting",
            "plt_usetex", fallback=False)
    arg.plt.print_xtick_every = config.getint("plotting",
            "print_xtick_every", fallback=10)

    return arg, config
