#!/usr/bin/env python

from nnTreeVB.data import evolve_seqs_full_homogeneity as\
        evolve_sequences
from nnTreeVB.data import simulate_tree

from nnTreeVB.data import build_tree_from_nwk
from nnTreeVB.data import TreeSeqCollection
from nnTreeVB.data import build_msa_categorical

from nnTreeVB.parse_config import parse_config

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
from nnTreeVB.models.vb_models import VB_nnTree

import sys
import os
from os import makedirs
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
    ret = e.fit(**fit_args)

    fit_hist = [
            ret["elbos_list"],
            ret["lls_list"],
            ret["kls_list"]
            ]

    overall["fit_hist_estim"] = ret["fit_estimates"]

    overall["fit_hist"] = np.array(fit_hist)

    ## Sampling after fitting
    ## ########################
    overall["samples"] = e.sample(
            fit_args["X"],
            fit_args["X_counts"],
            elbo_type=fit_args["elbo_type"],
            nb_samples=fit_args["nb_samples"],
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
    arg, config = parse_config(config_file)

    io = arg.io
    sim = arg.sim
    mdl = arg.mdl
    fit = arg.fit
    stg = arg.stg
    plt = arg.plt

    verbose = verbose

    # The order of freqs is different for pyvolve
    # A C G T
    sim_freqs_pyv = [sim.sim_freqs[0], sim.sim_freqs[2],
            sim.sim_freqs[1], sim.sim_freqs[3]]

    if mdl.subs_model not in ["jc69", "k80", "hky", "gtr"]:
        print("\nsubs_model should be jc69|k80|hky|gtr,"\
                " not {}".format(mdl.subs_model),
                file=sys.stderr)
        sys.exit()

    # Computing device setting
    device = stg.device
    # TODO check other gpu devices
    if device != "cpu" and\
            not torch.cuda.is_available():
        if verbose: 
            print("\nCuda is not available."\
                    " Changing device to 'cpu'")
        device = 'cpu'

    device = torch.device(device)

    # Set the job name
    if str(stg.job_name).lower() in ["auto", "none"]:
        stg.job_name = None

    if not stg.job_name:
        now = datetime.now()
        stg.job_name = now.strftime("%y%m%d%H%M")


    ## output path 
    ## ###########
    output_path = os.path.join(io.output_path,
            mdl.subs_model, stg.job_name)
    makedirs(output_path, mode=0o700, exist_ok=True)

    if verbose:
        print("\nExperiment output: {}".format(
            output_path))

    ## Get Fasta file names
    ## #########################
    if arg.sim_data:
        # Files paths of simulated data
        # training sequences
        fasta_file = output_path+"/{}.fasta".format(
                stg.job_name)
        tree_file = output_path+"/{}.nwk".format(
                stg.job_name)
    else:
        # Files paths of given FASTA files
        # TODO check if files exist or raise Exception
        fasta_file = io.seq_file
        tree_file = io.nwk_file

    # sim_params will be used to compare with estimated 
    # parameters
    sim_params = dict(
            b=np.array(post_branches),
            r=np.array(sim.sim_rates),
            f=np.array(sim.sim_freqs),
            k=np.array([[sim.sim_kappa]])
            )

    ## Loading results from file
    ## #########################
    results_file = output_path+"/{}_results.pkl".format(
            stg.job_name)

    if os.path.isfile(results_file) and scores_from_file:
        if verbose: print("\nLoading scores from file...")

        # Get the results
        result_data = load(results_file)
        rep_results=result_data["rep_results"]

    ## Execute the evaluation and save results
    ## #######################################
    else:
        if verbose: print("\nRunning the evaluation...")

        ## Data preparation
        ## ################
        if sim.sim_data and (not sim.sim_from_files) and \
                not os.path.isfile(fasta_file):
            # Simulate new data
 
            if not os.path.isfile(tree_file):
            # if tree file is not given, simulate a tree
            # using ete3 populate function
            if verbose: print("\nSimulating a new tree...")

                # set seed to numpy here
                tree_obj, post_branches = simulate_tree(
                        nb_taxa, sim.sim_blengths,
                        unroot=True, seed=seed)
                tree_obj.write(format=1, tree_file)

            if verbose:print("\nSimulating new sequences...")
            # Evolve sequences
            all_seqdict = evolve_sequences(
                    tree_obj,
                    fasta_file=fasta_file,
                    nb_sites=sim.nb_sites,
                    subst_rates=sim.sim_rates,
                    state_freqs=sim_freqs_pyv,
                    return_anc=False,
                    seed=stg.seed,
                    verbose=verbose)

        # Update file paths in config file
        config.set("io", "seq_file", fasta_file)
        config.set("io", "nwk_file", tree_file)

        if verbose: print("\nLoading data to"\
                " TreeSeqCollection collection...")
        treeseqs = TreeSeqCollection(fasta_file, tree_file)
        tree_obj = treeseqs.tree

        # Transform fitting sequences
        x_motifs_cats = build_msa_categorical(treeseqs,
                nuc_cat=False)
        X = torch.from_numpy(x_motifs_cats.data).to(device)
        X, X_counts = X.unique(dim=0, return_counts=True)

        #FIXME
        logl_data = compute_log_likelihood_data(
                X_unique,
                X_counts, post_branches, sim.sim_rates, 
                sim.sim_freqs)

        if verbose:
            print("\nLog likelihood of the fitting data"\
                    " {}".format(logl_data))

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
        EvoModelClass = VB_nnTree

        model_args = {
                "tree":tree_obj,
                "subs_model":mdl.subs_model
                #
                "ancestor_prior_hp":ancestor_prior_hp,
                "branch_prior_hp":branch_prior_hp,
                #
                "h_dim":mdl.h_dim,
                "nb_layers":mdl.nb_layers,
                "bias_layers":mdl.bias_layers,
                "activ_layers":mdl.activ_layers,
                "dropout_layers":mdl.dropout_layers,
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
        if fit.K_grad_samples > 1:
            grad_samples = [fit.grad_samples,
                    fit.K_grad_samples
        else:
            grad_samples = fit.grad_samples

        fit_args = {
                "X":X,
                "X_counts":X_counts,
                "elbo_type"=fit.elbo_type,
                "grad_samples"=grad_samples,
                "nb_samples"=fit.nb_samples,
                "alpha_kl"=fit.alpha_kl,
                "max_iter"=fit.max_iter,
                "optim"=fit.optim,
                "optim_learning_rate"=fit.learning_rate,
                "optim_weight_decay"=fit.weight_decay,
                "scheduler_lambda"=fit.scheduler_lambda,
                "save_fit_history"=fit.save_fit_history,
                "verbose":not sim_data
                }

        parallel = Parallel(n_jobs=nb_replicates, 
                prefer="processes", verbose=verbose)

        rep_results = parallel(delayed(eval_evomodel)(
            EvoModelClass,
            model_args, fit_args) for s in\
                    range(nb_replicates))

        #
        result_data = dict(
                rep_results=rep_results, # rep for replicates
                )

        result_data["logl_data"] = logl_data

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

    estim_samples = aggregate_sampled_estimates(
            rep_results, "samples")

    report_sampled_estimates(
            estim_samples,
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
