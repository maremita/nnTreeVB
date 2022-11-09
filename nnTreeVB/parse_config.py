from dataclasses import dataclass
import configparser

from nnTreeVB.checks import check_sim_blengths
from nnTreeVB.checks import check_sim_simplex
from nnTreeVB.checks import check_sim_float
#from nnTreeVB.checks import check_seed
from nnTreeVB.checks import check_verbose
from nnTreeVB.checks import check_prior_option
from nnTreeVB.checks import check_var_option

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
            "sim_data", "sim_data", fallback=False)
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

    lr = dict() # For learning rates for each distr type

    # Evo variational model type
    arg.mdl.subs_model = config.get(
            "hyperparams", "subs_model", fallback="jc69")

    # Hyper-parameters of prior distributions
    # Branches
    b_prior = check_prior_option(config.get("hyperparams",
        "b_prior", fallback="exponential|10.|False"))

    arg.mdl.b_prior_dist = b_prior[0]
    arg.mdl.b_prior_params = b_prior[1]
    arg.mdl.b_learn_prior = b_prior[2]
    if b_prior[3]: lr.update({"b_dist_p": b_prior[3]})

    b_var = check_var_option(config.get("hyperparams",
        "b_var", fallback="normal|0.1,0.1|lower_0"))

    arg.mdl.b_var_dist = b_var[0]
    arg.mdl.b_var_params = b_var[1]
    arg.mdl.b_var_transform = b_var[2]
    if b_var[3]: lr.update({"b_dist_q": b_var[3]})

    ## Tree length
    t_prior = check_prior_option(config.get("hyperparams",
        "t_prior", fallback="gamma|1.,1.|False"))

    arg.mdl.t_prior_dist = t_prior[0]
    arg.mdl.t_prior_params = t_prior[1]
    arg.mdl.t_learn_prior = t_prior[2]
    if t_prior[3]: lr.update({"t_dist_p": t_prior[3]})

    t_var = check_var_option(config.get("hyperparams",
        "t_var", fallback="normal|0.1,0.1|lower_0"))

    arg.mdl.t_var_dist = t_var[0]
    arg.mdl.t_var_params = t_var[1]
    arg.mdl.t_var_transform = t_var[2]
    if t_var[3]: lr.update({"t_dist_q": t_var[3]})

    ## Rates
    r_prior = check_prior_option(config.get("hyperparams",
        "r_prior", fallback="dirichlet|uniform|False"))

    arg.mdl.r_prior_dist = r_prior[0]
    arg.mdl.r_prior_params = r_prior[1]
    arg.mdl.r_learn_prior = r_prior[2]
    if r_prior[3]: lr.update({"r_dist_p": r_prior[3]})

    r_var = check_var_option(config.get("hyperparams",
        "r_var", fallback="normal|0.1,0.1|simplex"))

    arg.mdl.r_var_dist = r_var[0]
    arg.mdl.r_var_params = r_var[1]
    arg.mdl.r_var_transform = r_var[2]
    if r_var[3]: lr.update({"r_dist_q": r_var[3]})

    ## Frequencies
    f_prior = check_prior_option(config.get("hyperparams",
        "f_prior", fallback="dirichlet|uniform|False"))

    arg.mdl.f_prior_dist = f_prior[0]
    arg.mdl.f_prior_params = f_prior[1]
    arg.mdl.f_learn_prior = f_prior[2]
    if f_prior[3]: lr.update({"f_dist_p": f_prior[3]})

    f_var = check_var_option(config.get("hyperparams",
        "f_var", fallback="normal|0.1,0.1|simplex"))

    arg.mdl.f_var_dist = f_var[0]
    arg.mdl.f_var_params = f_var[1]
    arg.mdl.f_var_transform = f_var[2]
    if f_var[3]: lr.update({"f_dist_q": f_var[3]})

    ## Kappa
    k_prior = check_prior_option(config.get("hyperparams",
        "k_prior", fallback="gamma|1.,1.|False"))

    arg.mdl.k_prior_dist = k_prior[0]
    arg.mdl.k_prior_params = k_prior[1]
    arg.mdl.k_learn_prior = k_prior[2]
    if k_prior[3]: lr.update({"k_dist_p": k_prior[3]})

    k_var = check_var_option(config.get("hyperparams",
        "k_var", fallback="normal|0.1,0.1|lower_0"))

    arg.mdl.k_var_dist = k_var[0]
    arg.mdl.k_var_params = k_var[1]
    arg.mdl.k_var_transform = k_var[2]
    if k_var[3]: lr.update({"k_dist_q": k_var[3]})

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
            "hyperparams", "nb_samples", fallback=100)
    arg.fit.alpha_kl = config.getfloat(
            "hyperparams", "alpha_kl", fallback=1.)
    arg.fit.max_iter = config.getint(
            "hyperparams", "max_iter", fallback=100)
    arg.fit.optimizer = config.get(
            "hyperparams", "optimizer", fallback="adam")
    lr.update({"default": config.getfloat(
            "hyperparams", "learning_rate", fallback=0.1)})
    arg.fit.learning_rate = lr
    arg.fit.weight_decay = config.getfloat(
            "hyperparams", "weight_decay", fallback=0.)
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
    arg.stg.n_parallel = config.getint("settings", 
            "n_parallel", fallback=1)
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
