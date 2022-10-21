#!/usr/bin/env python

from nnTreeVB.data import evolve_seqs_full_homogeneity as evolve_sequences
from nnTreeVB.data import build_tree_from_nwk
from nnTreeVB.data import TreeSeqCollection
from nnTreeVB.data import build_msa_categorical

from nnTreeVB.utils import timeSince
from nnTreeVB.utils import get_categorical_prior 
from nnTreeVB.utils import get_branch_prior 
from nnTreeVB.utils import get_kappa_prior 
from nnTreeVB.utils import str2floats, fasta_to_list
from nnTreeVB.utils import str_to_values
from nnTreeVB.utils import write_conf_packages

from nnTreeVB.reports import plt_elbo_ll_kl_rep_figure
from nnTreeVB.reports import aggregate_estimate_values
from nnTreeVB.reports import plot_fit_estim_dist
from nnTreeVB.reports import plot_fit_estim_corr
from nnTreeVB.reports import plot_fit_seq_dist
from nnTreeVB.reports import aggregate_sampled_estimates
from nnTreeVB.reports import report_sampled_estimates

#from nnTreeVB.models import compute_log_likelihood_data 
from nnTreeVB.models import VB_nnTree

import sys
import os
from os import makedirs
import configparser
import time
from datetime import datetime

import numpy as np
import torch

from joblib import Parallel, delayed
from joblib import dump, load


__author__ = "amine remita"


"""
nntreevb.py is the main program that uses the VB_nnTree 
model with different substitution models (JC69, K80, HKY 
and GTR). The experiment of fitting the model can be done for
<nb_replicates> times.
The program can use a sequence alignment from a Fasta file or
simulate a new sequence alignment using evolutionary parameters
defined in the config file.

Once the nnTreeVB package is installed, you can run nntreevb.py 
using this command line:

# nntreevb.py evovgm_conf_template.ini
where <nntreevb_conf_template.ini> is the config file.

The program is adapted from evovgm.py (MIT license)
"""

## Evaluation function
## ###################
def eval_evomodel(EvoModel, m_args, fit_args):
    overall = dict()

    # Instanciate the model
    e = EvoModel(**m_args)

    ## Fitting and param3ter estimation
    ## ################################
    ret = e.fit(
            fit_args["X"],
            fit_args["X_counts"],
            elbo_type=fit_args["elbo_type"],
            sample_size=fit_args["nb_samples"],
            alpha_kl=fit_args["alpha_kl"], 
            max_iter=fit_args["max_iter"],
            optim=fit_args["optim"], 
            optim_learning_rate=fit_args["learning_rate"],
            optim_weight_decay=fit_args["weight_decay"],
            scheduler_lambda=fit_args["scheduler_lambda"]
            save_fit_history=fit_args["save_fit_history"],
            verbose=fit_args["verbose"]
            )

    fit_hist = [ret["elbos_list"], ret["lls_list"], ret["kls_list"]]

    overall["fit_hist_estim"] = ret["fit_estimates"]

    overall["fit_hist"] = np.array(fit_hist)

    ## Sampling after fitting
    ## ########################
    overall["samples"] = e.sample(
            fit_args["X"],
            fit_args["X_counts"],
            elbo_type=fit_args["elbo_type"],
            sample_size=fit_args["nb_samples"],
            alpha_kl=fit_args["alpha_kl"]
            )

    return overall


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Config file is missing!!")
        sys.exit()

    print("\nRunning {}".format(sys.argv[0]), flush=True)

    ## Fetch argument values from ini file
    ## ###################################
    config_file = sys.argv[1]
    config = configparser.ConfigParser(
            interpolation=\
                    configparser.ExtendedInterpolation())

    with open(config_file, "r") as cf:
        config.read_file(cf)

    # IO files
    output_path = config.get("io", "output_path")
    
    seq_file = config.get("io", "seq_file",
            fallback="")
    nwk_file = config.get("io", "nwk_file",
            fallback="")

    scores_from_file = config.getboolean("io",
            "scores_from_file", fallback=False)

    # Simulation data
    sim_data = config.getboolean("sim_data", "sim_data",
            fallback=True)
    sim_from_fasta = config.getboolean("sim_data",
            "sim_from_fasta", fallback=True)
    nb_sites = config.getint("sim_data",
            "nb_sites", fallback=100)
    nb_taxa = config.getint("sim_data",
            "nb_taxa", fallback=100)
    sim_blengths = config.get("sim_data",
            "branch_lengths", fallback="0.1,0.1")
    sim_rates = str_to_values(config.get("sim_data", "rates",
        fallback="0.16"), 6, cast=float)
    sim_freqs = str_to_values(config.get("sim_data",
        "frequencies", fallback="0.25"), 4, cast=float)
    sim_kappa = config.getfloat("sim_data",
        "kappa", fallback=1.)

    # The order of freqs is different for pyvolve
    # A CGT
    sim_freqs_pyv = [sim_freqs[0], sim_freqs[2],
            sim_freqs[1], sim_freqs[3]]

    # setting parameters
    job_name = config.get("settings", "job_name",
            fallback=None)
    seed = config.getint("settings", "seed",
            fallback=42)
    device_str = config.get("settings", "device",
            fallback="cpu")
    verbose = config.get("settings", "verbose",
            fallback=1)
    compress_files = config.getboolean("settings", 
            "compress_files", fallback=False)

    # Evo variational model type
    subs_model = config.get("hyperparams",
            "subs_model", fallback="gtr")

    # Hyper parameters
    nb_replicates = config.getint("hyperparams",
            "nb_replicates", fallback=2)
    nb_samples = config.getint("hyperparams", "nb_samples",
            fallback=10)
    h_dim = config.getint("hyperparams", "hidden_size",
            fallback=32)
    nb_layers = config.getint("hyperparams", "nb_layers",
            fallback=3)
    alpha_kl = config.getfloat("hyperparams", "alpha_kl",
            fallback=0.0001)
    max_iter = config.getint("hyperparams", "max_iter",
            fallback=100)
    optim = config.get("hyperparams", "optim",
            fallback="adam")
    learning_rate = config.getfloat("hyperparams",
            "learning_rate", fallback=0.005)
    weight_decay = config.getfloat("hyperparams",
            "weight_decay", fallback=0.00001)

    # Hyper-parameters of prior distributions
    ancestor_hp_conf = config.get("priors",
            "ancestor_prior_hp", fallback="uniform")
    branch_hp_conf = config.get("priors", "branch_prior_hp",
            fallback="0.1,0.1")
    kappa_hp_conf = config.get("priors", "kappa_prior_hp",
            fallback="1.,1.")
    rates_hp_conf = config.get("priors", "rates_prior_hp",
            fallback="uniform")
    freqs_hp_conf = config.get("priors", "freqs_prior_hp",
            fallback="uniform")

    # plotting settings
    size_font = config.getint("plotting", "size_font",
            fallback=16)
    plt_usetex = config.getboolean("plotting", "plt_usetex",
            fallback=False)
    print_xtick_every = config.getint("plotting",
            "print_xtick_every", fallback=10)
    make_logos = config.getboolean("plotting",
            "make_logos", fallback=True)

    # Process verbose
    if verbose.lower() == "false":
        verbose = 0
    elif verbose.lower() == "none":
        verbose = 0
    elif verbose.lower() == "true":
        verbose = 1
    else:
        try:
            verbose = int(verbose)
            if verbose < 0:
                print("\nInvalid value for verbose"\
                        " {}".format(verbose))
                print("Valid values are: True, False, None"\
                        " and positive integers")
                print("Verbose is set to 0")
                verbose = 0
        except ValueError as e:
            print("\nInvalid value for verbose {}".format(
                verbose))
            print("Valid values are: True, False, None and"\
                    " positive integers")
            print("Verbose is set to 1")
            verbose = 1

    if subs_model not in ["jc69", "k80", "hky", "gtr"]:
        print("\nsubs_model should be jc69|k80|hky|gtr,"\
                " not {}".format(subs_model),
                file=sys.stderr)
        sys.exit()

    # Computing device setting
    if device_str != "cpu" and not torch.cuda.is_available():
        if verbose: 
            print("\nCuda is not available."\
                    " Changing device to 'cpu'")
        device_str = 'cpu'

    device = torch.device(device_str)

    if str(job_name).lower() in ["auto", "none"]:
        job_name = None

    if not job_name:
        now = datetime.now()
        job_name = now.strftime("%y%m%d%H%M")

    sim_params = dict(
            b=np.array(str2floats(sim_blengths)),
            r=np.array(sim_rates),
            f=np.array(sim_freqs),
            k=np.array([[sim_kappa]])
            )

    ## output name file
    ## ################
    output_path = os.path.join(output_path,
            subs_model, job_name)
    makedirs(output_path, mode=0o700, exist_ok=True)

    if verbose:
        print("\nExperiment output: {}".format(
            output_path))

    ## Get Fasta file names
    ## #########################
    if sim_data:
        # Files paths of simulated data
        # training sequences
        x_fasta_file = output_path+"/{}_train.fasta".format(
                job_name)
    else:
        # Files paths of given FASTA files
        # training sequences
        x_fasta_file = seq_file

    ## Loading results from file
    ## #########################
    results_file = output_path+"/{}_results.pkl".format(
            job_name)

    if os.path.isfile(results_file) and scores_from_file:
        if verbose: print("\nLoading scores from file...")

        # Get the results
        result_data = load(results_file)
        rep_results=result_data["rep_results"]
        
        # Get X_gen ndarray 
        # used in the generation step as input 
        x_gen_fasta = x_fasta_file

        if sim_data:
            # Simulated Fasta contains ancestral sequence, to discard
            x_gen_sequences = fasta_to_list(
                    x_gen_fasta, verbose)[1:]
        else:
            x_gen_sequences = fasta_to_list(
                    x_gen_fasta, verbose)

        X_gen = build_msa_categorical(x_gen_sequences).data

    ## Execute the evaluation and save results
    ## #######################################
    else:
        if verbose: print("\nRunning the evaluation...")

        ## Data preparation
        ## ################
        if sim_data:
            ## Files paths of simulated data
            ## training sequences
            #x_fasta_file = output_path+"/{}_train.fasta".format(
            #        job_name)

            # Extract data from simulated files if they exist
            if os.path.isfile(x_fasta_file) and sim_from_fasta:
                if verbose: 
                    print("\nLoading simulated sequences"\
                            " from files...")
                # Load from files
                # Here, simulated FASTA file contain the
                # root sequence
                ax_sequences = fasta_to_list(x_fasta_file,
                        verbose)
                x_root = ax_sequences[0]
                x_sequences = ax_sequences[1:]

            # Simulate new data
            else:
                if verbose: print("\nSimulating sequences...")
                # Evolve sequences
                tree=build_star_tree(sim_blengths)

                x_root, x_sequences = evolve_sequences(
                        tree,
                        fasta_file=x_fasta_file,
                        nb_sites=nb_sites,
                        subst_rates=sim_rates,
                        state_freqs=sim_freqs_pyv,
                        return_anc=True,
                        seed=seed,
                        verbose=verbose)
            #FIXME
            x_logl_data = compute_log_likelihood_data(
                    AX_unique,
                    AX_counts, sim_blengths, sim_rates, 
                    sim_freqs)

            if verbose:
                print("\nLog likelihood of the fitting data"\
                        " {}".format(x_logl_data))

        # Extract data from given FASTA files
        else:
            ## Files paths of given FASTA files
            ## training sequences
            #x_fasta_file = seq_file

            # Given FASTA files do not contain root sequences
            if verbose: print("\nLoading sequences from"\
                    " files...")
            x_sequences = fasta_to_list(x_fasta_file, verbose)

        # End of fetching/simulating the data

        # Update file paths in config file
        config.set("io", "seq_file", x_fasta_file)

        # Transform fitting sequences
        x_motifs_cats = build_msa_categorical(x_sequences)
        X = torch.from_numpy(x_motifs_cats.data).to(device)
        X_gen = X.clone().detach() # will be used in generation
        X, X_counts = X.unique(dim=0, return_counts=True)

        # Set dimensions
        x_dim = 4
        a_dim = 4
        m_dim = len(x_sequences) # Number of sequences

        ## Get prior hyper-parameter values
        ## ################################
        if verbose:
            print("\nGet the prior hyper-parameters...")

        ancestor_prior_hp = get_categorical_prior(
                ancestor_hp_conf,
                "ancestor", verbose=verbose)
        branch_prior_hp = get_branch_prior(branch_hp_conf,
                verbose=verbose)

        if subs_model == "gtr":
            # Get rate and freq priors if the model is_GTR
            rates_prior_hp = get_categorical_prior(
                    rates_hp_conf, 
                    "rates", verbose=verbose)
            freqs_prior_hp = get_categorical_prior(
                    freqs_hp_conf,
                    "freqs", verbose=verbose)

        elif subs_model == "k80":
            kappa_prior_hp = get_kappa_prior(kappa_hp_conf, 
                    verbose=verbose)

        if verbose: print()
        ## Evo model type
        ## ##############
        if subs_model == "gtr":
            EvoModelClass = EvoVGM_GTR
        elif subs_model == "k80":
            EvoModelClass = EvoVGM_K80
        else:
            EvoModelClass = EvoVGM_JC69

        model_args = {
                "h_dim":h_dim,
                "ancestor_prior_hp":ancestor_prior_hp,
                "branch_prior_hp":branch_prior_hp,
                "device":device
                }

        if subs_model == "gtr":
            # Add rate and freq priors if the model is GTR
            model_args["rates_prior_hp"] = rates_prior_hp
            model_args["freqs_prior_hp"] = freqs_prior_hp

        elif subs_model == "k80":
            model_args["kappa_prior_hp"] = kappa_prior_hp

        save_fit_history=True

        # Fitting the parameters
        fit_args = {
                "X":X,
                "X_counts":X_counts,
                "X_gen":X_gen,
                "nb_samples":nb_samples,
                "alpha_kl":alpha_kl,
                "max_iter":max_iter,
                "optim":"adam",
                "learning_rate":learning_rate,
                "weight_decay":weight_decay,
                "save_fit_history":save_fit_history,
                "":
                "verbose":not sim_data
                }

        parallel = Parallel(n_jobs=nb_replicates, 
                prefer="processes", verbose=verbose)

        rep_results = parallel(delayed(eval_evomodel)(EvoModelClass,
            model_args, fit_args) for s in range(nb_replicates))

        #
        result_data = dict(
                rep_results=rep_results, # rep for replicates
                )

        if sim_data:
            result_data["logl_data"] = x_logl_data

        dump(result_data, results_file,
                compress=compress_files)

        # Writing a config file and package versions
        conf_file = output_path+"/{}_conf.ini".format(
                job_name)
        if not os.path.isfile(conf_file):
            write_conf_packages(config, conf_file)

    ## Report and plot results
    ## #######################
    scores = [result["fit_hist"] for result in rep_results]

    # get min number of epoch of all reps 
    # (maybe some reps stopped before max_iter)
    # to slice the list of epochs with the same length 
    # and be able to cast the list in ndarray        
    min_iter = scores[0].shape[1]
    for score in scores:
        if min_iter >= score.shape[1]:
            min_iter = score.shape[1]
    the_scores = []
    for score in scores:
        the_scores.append(score[:,:min_iter])

    the_scores = np.array(the_scores)
    #print("The scores {}".format(the_scores.shape))

    ## Generate report file from sampling step
    ## #######################################
    if verbose: print("\nGenerate reports...")

    estim_gens = aggregate_sampled_estimates(
            rep_results, "gen_results")

    report_sampled_estimates(
            estim_gens,
            output_path+"/{}_estim_report".format(job_name),
            )

    ## Ploting results
    ## ###############
    if verbose: print("\nPlotting...")
    plt_elbo_ll_kl_rep_figure(
            the_scores,
            output_path+"/{}_rep_fig".format(job_name),
            sizefont=size_font,
            usetex=plt_usetex,
            print_xtick_every=print_xtick_every,
            title=None,
            plot_validation=False)

    hist = "fit"

    estimates = aggregate_estimate_values(rep_results,
            "{}_hist_estim".format(hist))
    #return a dictionary of arrays

    # Distance between estimated paramerters 
    # and values given in the config file
    plot_fit_estim_dist(
            estimates, 
            sim_params,
            output_path+"/{}_{}_estim_dist".format(job_name,
                hist),
            sizefont=size_font,
            usetex=plt_usetex,
            print_xtick_every=print_xtick_every,
            y_limits=[-0.1, 1.1],
            legend='upper right')

    # Correlation between estimated paramerters 
    # and values given in the config file
    plot_fit_estim_corr(
            estimates, 
            sim_params,
            output_path+"/{}_{}_estim_corr".format(job_name,
                hist),
            sizefont=size_font,
            usetex=plt_usetex,
            print_xtick_every=print_xtick_every,
            y_limits=[-1.1, 1.1],
            legend='lower right')

    print("\nFin normale du programme\n")
