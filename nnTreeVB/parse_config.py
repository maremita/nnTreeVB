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
    arg.mdl.b_prior_dist = check_dist_type(config.get(
        "hyperparams", "b_prior_dist", fallback="exponential"))
    arg.mdl.b_prior_params = check_dist_params(config.get(
        "hyperparams", "b_prior_params", fallback="10."))
    arg.mdl.b_learn_prior = config.getboolean("hyperparams",
            "b_learn_prior", fallback=False)

    arg.mdl.b_var_dist = check_dist_type(config.get(
        "hyperparams", "b_var_dist", fallback="normal"))
    arg.mdl.b_var_params = check_dist_params(config.get(
        "hyperparams", "b_var_params", fallback="0.1,0.1"))
    arg.mdl.b_var_transform = check_dist_transform(config.get(
        "hyperparams", "b_var_transform", fallback="lower_0"))

    # Tree lengths
    arg.mdl.t_prior_dist = check_dist_type(config.get(
        "hyperparams", "t_prior_dist", fallback="gamma"))
    arg.mdl.t_prior_params = check_dist_params(config.get(
        "hyperparams", "t_prior_params", fallback="1.,1."))
    arg.mdl.t_learn_prior = config.getboolean("hyperparams",
            "t_learn_prior", fallback=False)

    arg.mdl.t_var_dist = check_dist_type(config.get(
        "hyperparams", "t_var_dist", fallback="normal"))
    arg.mdl.t_var_params = check_dist_params(config.get(
        "hyperparams", "t_var_params", fallback="0.1,0.1"))
    arg.mdl.t_var_transform = check_dist_transform(config.get(
        "hyperparams", "t_var_transform", fallback="lower_0"))

    # Rates
    arg.mdl.r_prior_dist = check_dist_type(config.get(
        "hyperparams", "r_prior_dist", fallback="dirichlet"))
    arg.mdl.r_prior_params = check_dist_params(config.get(
        "hyperparams", "r_prior_params", fallback="uniform"))
    arg.mdl.r_learn_prior = config.getboolean("hyperparams",
            "r_learn_prior", fallback=False)

    arg.mdl.r_var_dist = check_dist_type(config.get(
        "hyperparams", "r_var_dist", fallback="normal"))
    arg.mdl.r_var_params = check_dist_params(config.get(
        "hyperparams", "r_var_params", fallback="0.1,0.1"))
    arg.mdl.r_var_transform = check_dist_transform(config.get(
        "hyperparams", "r_var_transform", fallback="simplex"))

    # Frequencies
    arg.mdl.f_prior_dist = check_dist_type(config.get(
        "hyperparams", "f_prior_dist", fallback="dirichlet"))
    arg.mdl.f_prior_params = check_dist_params(config.get(
        "hyperparams", "f_prior_params", fallback="uniform"))
    arg.mdl.f_learn_prior = config.getboolean("hyperparams",
            "f_learn_prior", fallback=False)

    arg.mdl.f_var_dist = check_dist_type(config.get(
        "hyperparams", "f_var_dist", fallback="normal"))
    arg.mdl.f_var_params = check_dist_params(config.get(
        "hyperparams", "f_var_params", fallback="0.1,0.1"))
    arg.mdl.f_var_transform = check_dist_transform(config.get(
        "hyperparams", "f_var_transform", fallback="simplex"))

    # Kappa
    arg.mdl.k_prior_dist = check_dist_type(config.get(
        "hyperparams", "k_prior_dist", fallback="gamma"))
    arg.mdl.k_prior_params = check_dist_params(config.get(
        "hyperparams", "k_prior_params", fallback="1.,1."))
    arg.mdl.k_learn_prior = config.getboolean("hyperparams",
            "k_learn_prior", fallback=False)

    arg.mdl.k_var_dist = check_dist_type(config.get(
        "hyperparams", "k_var_dist", fallback="normal"))
    arg.mdl.k_var_params = check_dist_params(config.get(
        "hyperparams", "k_var_params", fallback="0.1,0.1"))
    arg.mdl.k_var_transform = check_dist_transform(config.get(
        "hyperparams", "k_var_transform", fallback="lower_0"))

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
