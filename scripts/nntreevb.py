#!/usr/bin/env python

from nnTreeVB.data import evolve_seqs_full_homogeneity as\
        evolve_sequences
from nnTreeVB.data import simulate_tree
from nnTreeVB.data import build_tree_from_nwk
from nnTreeVB.data import get_postorder_branches
from nnTreeVB.data import TreeSeqCollection
from nnTreeVB.data import build_msa_categorical

from nnTreeVB.parse_config import parse_config

from nnTreeVB.utils import timeSince
from nnTreeVB.utils import write_conf_packages
from nnTreeVB.utils import update_sim_parameters

from nnTreeVB.reports import plt_elbo_ll_kl_rep_figure
from nnTreeVB.reports import aggregate_estimate_values
from nnTreeVB.reports import plot_fit_estim_dist
from nnTreeVB.reports import plot_fit_estim_corr
from nnTreeVB.reports import aggregate_sampled_estimates
from nnTreeVB.reports import report_sampled_estimates

#from nnTreeVB.models import compute_log_likelihood_data 
from nnTreeVB.models.vb_models import VB_nnTree

import sys
import os
from os import makedirs
import time
from datetime import datetime
import random
import argparse

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import numpy as np
import torch

from joblib import Parallel, delayed
from joblib import dump, load

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', 
        "r'All-NaN (slice|axis) encountered'")

__author__ = "amine remita"


"""
nntreevb.py is the main program that uses the VB_nnTree 
model with different substitution models (JC69, K80, HKY 
and GTR). The experiment of fitting the model can be done for
<fit.nb_replicates> times.
The program can use a sequence alignment from a Fasta file or
simulate a new sequence alignment using evolutionary parameters
defined in the config file.

Once the nnTreeVB package is installed, you can run nntreevb.py 
using this command line:

# nntreevb.py evovgm_conf_template.ini
where <nntreevb_conf_template.ini> is the config file.

The program is adapted from evovgm.py ((c) remita 2022,  MIT license)
"""

## Evaluation function
## ###################
def eval_evomodel(EvoModel, m_args, fit_args):
    overall = dict()

    # Instanciate the model
    e = EvoModel(**m_args)

    ## Fitting and param3ter estimation
    ret = e.fit(**fit_args)

    overall["fit_probs"] = np.array([
            ret["elbos_list"],
            ret["lls_list"],
            ret["kls_list"]
            ])

    if fit_args["save_fit_history"]:
        overall["fit_estimates"] = ret["fit_estimates"]

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

    # Parse command line
    ####################
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str,
            required=True)
    parser.add_argument('-s', '--seed', type=int)
    cmd_args = parser.parse_args()

    config_file = cmd_args.config_file
    seed = cmd_args.seed

    print("\nRunning {} with config file: {}".format(
        sys.argv[0], config_file), flush=True)

    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
        print("\tSeed set to {}".format(seed))

    ## Fetch argument values from ini file
    ## ###################################
    cfg_args, config = parse_config(config_file)

    io  = cfg_args.io
    sim = cfg_args.sim
    mdl = cfg_args.mdl
    fit = cfg_args.fit
    stg = cfg_args.stg
    plt = cfg_args.plt

    verbose = stg.verbose

    if verbose:
        print("\tVerbose set to {}".format(verbose))

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

    ## Get Fasta and tree file names
    ## #############################
    if sim.sim_data:
        # Files paths of simulated data
        # training sequences
        fasta_file = output_path+"/{}.fasta".format(
                stg.job_name)
        tree_file = output_path+"/{}.nwk".format(
                stg.job_name)
    else:
        # Files paths of given FASTA files
        fasta_file = io.seq_file
        tree_file = io.nwk_file

        # TODO check if files exist or raise Exception
        if not os.path.isfile(fasta_file):
            raise FileNotFoundError("Fasta file {} does not "\
                    "exist".format(fasta_file))

        if not os.path.isfile(tree_file):
            raise FileNotFoundError("Newick file {} does not "\
                    "exist".format(tree_file))

    # Update file paths in config file
    config.set("io", "seq_file", fasta_file)
    config.set("io", "nwk_file", tree_file)

    ## Loading results from file
    ## #########################
    results_file = output_path+"/{}_results.pkl".format(
            stg.job_name)

    if os.path.isfile(results_file) and io.scores_from_file:
        if verbose: print("\nLoading scores from file...")

        # Get the results
        result_data = load(results_file)
        rep_results=result_data["rep_results"]

    ## Execute the evaluation and save results
    ## #######################################
    else:
        if verbose: print("\nRunning the evaluation...")

        result_data = dict()

        ## Data preparation
        ## ################
        if sim.sim_data:
            
            # Update sim params based on sim.subs_model
            # (useful to update rates using k (k80, hky))
            update_sim_parameters(sim)

            if os.path.isfile(tree_file) and\
                    sim.nwk_from_file:
                if verbose: print("\nExtracting simulated"\
                        " tree from {} ...".format(tree_file))
                tree_obj, taxa, int_nodes =\
                        build_tree_from_nwk(tree_file)

            else:
                # if tree file is not given, or 
                # sim.nwk_from_file is false: simulate a tree
                # using ete3 populate function
                if verbose:print("\nSimulating a new tree...")

                # set seed to numpy here
                tree_obj, taxa, int_nodes = simulate_tree(
                        sim.nb_taxa,
                        sim.sim_blengths,
                        unroot=True)

                tree_obj.write(outfile=tree_file, format=1)

            if not os.path.isfile(fasta_file) or \
                    not sim.seq_from_file:
                if verbose:
                    print("\nSimulating new sequences...")
                tree_nwk = tree_obj.write(format=1)
 
                # The order of freqs is different for pyvolve
                # A C G T
                sim_freqs_pyv = [
                        sim.sim_freqs[0],
                        sim.sim_freqs[2],
                        sim.sim_freqs[1],
                        sim.sim_freqs[3]]

                # Evolve sequences
                all_seqdict = evolve_sequences(
                        tree_nwk,
                        fasta_file=None, # write internal seqs
                        nb_sites=sim.nb_sites,
                        subst_rates=sim.sim_rates,
                        state_freqs=sim_freqs_pyv,
                        return_anc=False,
                        seed=seed,
                        verbose=verbose)
     
                # Write fasta
                seq_taxa = {s:all_seqdict[s] for s in taxa}
                records = [SeqRecord(Seq(seq_taxa[taxon]),
                    taxon, '', '') for taxon in seq_taxa]
                SeqIO.write(records, fasta_file, "fasta")

            post_branches = get_postorder_branches(
                    tree_obj)

            # sim_params will be used to compare with
            # estimated parameters
            result_data["sim_params"] = dict(
                    b=np.array(post_branches),
                    t=np.sum(np.array(post_branches),
                        keepdims=1),
                    r=np.array(sim.sim_rates),
                    f=np.array(sim.sim_freqs),
                    k=np.array([[sim.sim_kappa]])
                    )

        if verbose: print("\nLoading data to"\
                " TreeSeqCollection collection...")
        treeseqs = TreeSeqCollection(fasta_file, tree_file)
        tree_obj = treeseqs.tree # The tree is sorted

        # Transform fitting sequences
        x_motifs_cats = build_msa_categorical(treeseqs,
                nuc_cat=False)
        X = torch.from_numpy(x_motifs_cats.data).to(device)
        X, X_counts = X.unique(dim=0, return_counts=True)

        #FIXME
        #logl_data = compute_log_likelihood_data(
        #        X_unique,
        #        X_counts, post_branches, sim.sim_rates, 
        #        sim.sim_freqs)

        #if verbose:
        #    print("\nLog likelihood of the fitting data"\
        #            " {}".format(logl_data))

        ## Get prior hyper-parameter values
        ## ################################
        #if verbose:
        #    print("\nGet the prior hyper-parameters...")

        if verbose: print()
        ## Evo model type
        ## ##############
        EvoModelClass = VB_nnTree

        model_args = {
                "tree":tree_obj,
                "device":device,
                **mdl.to_dict()
                }

        #print(model_args)

        # Fitting the parameters 
        if fit.K_grad_samples > 1:
            fit.grad_samples = [fit.grad_samples,
                    fit.K_grad_samples]

        fit_args = {
                "X":X,
                "X_counts":X_counts,
                "verbose":not sim.sim_data,
                #"verbose":verbose,
                **fit.to_dict()
                }

        parallel = Parallel(n_jobs=fit.nb_replicates, 
                prefer="processes", verbose=verbose)

        rep_results = parallel(delayed(eval_evomodel)(
            EvoModelClass,
            model_args, fit_args) for s in\
                    range(fit.nb_replicates))
        #
        #result_data = dict(
        #        rep_results=rep_results, # rep for replicates
        #        )
        result_data["rep_results"] = rep_results

        #result_data["logl_data"] = logl_data

        dump(result_data, results_file,
                compress=stg.compress_files)

        # Writing a config file and package versions
        conf_file = output_path+"/{}_conf.ini".format(
                stg.job_name)
        if not os.path.isfile(conf_file):
            write_conf_packages(config, conf_file)

    if sim.sim_data:
        sim_params = result_data["sim_params"] 

    ## Report and plot results
    ## #######################
    scores = [result["fit_probs"] for\
            result in rep_results]

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

    ## Ploting results
    ## ###############
    # TODO: Add horizontal line for true logl
    if verbose: print("\nPlotting...")
    plt_elbo_ll_kl_rep_figure(
            the_scores,
            output_path+"/{}_rep_fig".format(stg.job_name),
            sizefont=plt.size_font,
            usetex=plt.plt_usetex,
            print_xtick_every=plt.print_xtick_every,
            title=None,
            plot_validation=False)

    if fit.save_fit_history or fit.save_val_history:
        hist = "fit" # [fit | val]
        estimates = aggregate_estimate_values(rep_results,
                "{}_estimates".format(hist))
        #print(estimates)
        #return a dictionary of dictionary of arrays

        ## Distance between estimated paramerters 
        ## and values given in the config file
        plot_fit_estim_dist(
                estimates, 
                sim_params,
                output_path+"/{}_{}_estim_dist".format(
                    stg.job_name, hist),
                sizefont=plt.size_font,
                usetex=plt.plt_usetex,
                print_xtick_every=plt.print_xtick_every,
                y_limits=[-0.1, 1.1],
                legend='upper right')

        ## Correlation between estimated paramerters 
        ## and values given in the config file
        plot_fit_estim_corr(
                estimates, 
                sim_params,
                output_path+"/{}_{}_estim_corr".format(
                    stg.job_name, hist),
                sizefont=plt.size_font,
                usetex=plt.plt_usetex,
                print_xtick_every=plt.print_xtick_every,
                y_limits=[-1.1, 1.1],
                legend='lower right')

    ## Generate report file from sampling step
    ## #######################################
    #if verbose: print("\nGenerate reports...")

    estim_samples = aggregate_sampled_estimates(
            rep_results, "samples")

    report_sampled_estimates(
            estim_samples,
            output_path+"/{}_estim_report.txt".format(
                stg.job_name))

    print("\nFin normale du programme\n")
