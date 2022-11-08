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
from nnTreeVB.utils import dict_to_numpy, dict_to_tensor

from nnTreeVB.reports import plot_elbo_ll_kl
from nnTreeVB.reports import aggregate_estimate_values
from nnTreeVB.reports import plot_fit_estim_distance
from nnTreeVB.reports import plot_fit_estim_correlation
from nnTreeVB.reports import plot_weights_grads_epochs
from nnTreeVB.reports import aggregate_sampled_estimates
from nnTreeVB.reports import report_sampled_estimates

from nnTreeVB.models.evo_models import compute_log_likelihood
from nnTreeVB.models.vb_models import VB_nnTree

import sys
import os
from os import makedirs
import time
from datetime import datetime
import random
import argparse
import copy

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

    ret["fit_probs"] = np.array([
            ret["elbos_list"],
            ret["lls_list"],
            ret["kls_list"]
            ])

    ## Sampling after fitting
    ## ########################
    ret["samples"] = e.sample(
            fit_args["X"],
            fit_args["X_counts"],
            elbo_type=fit_args["elbo_type"],
            nb_samples=fit_args["nb_samples"],
            alpha_kl=fit_args["alpha_kl"]
            )

    return ret


if __name__ == "__main__":

    # Parse command line
    ####################
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str,
            required=True)
    parser.add_argument('-s', '--seed', type=int)
    cmd_args = parser.parse_args()

    config_file = cmd_args.config_file.strip()
    seed = cmd_args.seed

    print("\nRunning {} with config file: {}".format(
        sys.argv[0], config_file), flush=True)

    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("\tSeed set to {}".format(seed))

    ## Parse config file
    ## #################
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

    if seed:
        config.set("settings", "seed", str(seed))

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
            stg.job_name)
    makedirs(output_path, mode=0o700, exist_ok=True)

    pg_path = os.path.join(output_path, "params_grads")
    makedirs(pg_path, mode=0o700, exist_ok=True)

    if verbose:
        print("\nExperiment output: {}".format(
            output_path))

    ## Get Fasta and tree file names
    ## #############################
    if sim.sim_data:
        # Files paths of simulated data
        # training sequences
        fasta_file = output_path+"/{}_input.fasta".format(
                stg.job_name)
        tree_file = output_path+"/{}_input.nwk".format(
                stg.job_name)
    else:
        # Files paths of given FASTA files
        fasta_file = io.seq_file
        tree_file = io.nwk_file

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
                    tree_obj) # numpy array

            # sim_params_np will be used to compare with
            # estimated parameters
 
            sim_params_np = dict_to_numpy(dict(
                    b=post_branches,
                    t=np.sum(post_branches, keepdims=1),
                    r=sim.sim_rates,
                    f=sim.sim_freqs,
                    k=sim.sim_kappa))

            sim_params_tensor = dict_to_tensor(sim_params_np,
                    device=device, dtype=torch.float32)

            for key in sim_params_tensor:
                sim_params_tensor[key] =\
                        sim_params_tensor[key].unsqueeze(0)

            result_data["sim_params"] = sim_params_np

        if verbose: print("\nLoading data to"\
                " TreeSeqCollection collection...")
        treeseqs = TreeSeqCollection(fasta_file, tree_file)
        tree_obj = treeseqs.tree # The tree is sorted

        # Transform fitting sequences
        x_motifs_cats = build_msa_categorical(treeseqs,
                nuc_cat=False)
        X = torch.from_numpy(x_motifs_cats.data).to(device)
        X, X_counts = X.unique(dim=0, return_counts=True)

        if sim.sim_data:
            with torch.no_grad():
                logl_data = (compute_log_likelihood(
                        copy.deepcopy(tree_obj),
                        X.unsqueeze(0),
                        sim.subs_model,
                        sim_params_tensor,
                        torch.tensor([sim.sim_freqs]).to(
                            device=device),
                        rescaled_algo=False, 
                        device=device)\
                                * X_counts).sum().cpu().numpy()

            result_data["logl_data"] = logl_data

            if verbose:
                print("\nLog likelihood of the data:"\
                        " {:.4f}".format(logl_data))

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

        # Fitting the parameters 
        if fit.K_grad_samples > 1:
            fit.grad_samples = [fit.grad_samples,
                    fit.K_grad_samples]
        else:
            fit.grad_samples = [fit.grad_samples]

        fit_args = {
                "X":X,
                "X_counts":X_counts,
                "verbose":not sim.sim_data,
                #"verbose":verbose,
                **fit.to_dict()
                }

        parallel = Parallel(n_jobs=stg.n_parallel, 
                prefer="processes", verbose=verbose)

        rep_results = parallel(delayed(eval_evomodel)(
            EvoModelClass,
            model_args, fit_args) for s in\
                    range(fit.nb_replicates))
        #
        result_data["rep_results"] = rep_results

        dump(result_data, results_file,
                compress=stg.compress_files)

        # Writing a config file and package versions
        conf_file = output_path+"/{}_conf.ini".format(
                stg.job_name)
        if not os.path.isfile(conf_file):
            write_conf_packages(config, conf_file)

    if sim.sim_data:
        sim_params_np = result_data["sim_params"] 

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
    
    logl_data = None
    if "logl_data" in result_data: 
        logl_data = result_data["logl_data"]

    plot_elbo_ll_kl(
            the_scores,
            output_path+"/{}_probs_fig".format(stg.job_name),
            line=logl_data,
            sizefont=plt.size_font,
            usetex=plt.plt_usetex,
            print_xtick_every=plt.print_xtick_every,
            title=None,
            plot_validation=False)

    if fit.save_fit_history:
        history = "fit" # [fit | val]
        estimates = aggregate_estimate_values(rep_results,
                "{}_estimates".format(history))
        #print(estimates)
        #return a dictionary of dictionary of arrays

        ## Distance between estimated paramerters 
        ## and values given in the config file
        plot_fit_estim_distance(
                estimates, 
                sim_params_np,
                output_path+"/{}_{}_estim_dist".format(
                    stg.job_name, history),
                sizefont=plt.size_font,
                usetex=plt.plt_usetex,
                print_xtick_every=plt.print_xtick_every,
                y_limits=[-0.1, 1.1],
                legend='upper right')

        ## Correlation between estimated paramerters 
        ## and values given in the config file
        plot_fit_estim_correlation(
                estimates, 
                sim_params_np,
                output_path+"/{}_{}_estim_corr".format(
                    stg.job_name, history),
                sizefont=plt.size_font,
                usetex=plt.plt_usetex,
                print_xtick_every=plt.print_xtick_every,
                y_limits=[-1.1, 1.1],
                legend='lower right')

    # Plot weights and grads for each learned distribution and 
    # for each replicate
    if fit.save_grad_stats and fit.save_weight_stats:
        # get the names of learned distributions
        distrs = [d for d in rep_results[0]["grad_stats"][0]]
        for n_rep in range(fit.nb_replicates):
            for dist_name in distrs:
                out_file = pg_path+"/{}_r{}_{}".format(
                        stg.job_name, n_rep, dist_name)

                plot_weights_grads_epochs(
                        rep_results[n_rep],
                        dist_name,
                        out_file,
                        epochs=slice(0, -1), # 0(,min_iter)
                        fig_size=(10, 3),
                        sizefont=plt.size_font
                        )

    ## Generate report file from sampling step
    ## #######################################
    if verbose: print("\nGenerate reports...")

    estim_samples = aggregate_sampled_estimates(
            rep_results, "samples")

    report_sampled_estimates(
            estim_samples,
            output_path+"/{}_estim_report.txt".format(
                stg.job_name))

    print("\nFin normale du programme\n")
