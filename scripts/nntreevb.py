#!/usr/bin/env python

from nnTreeVB import __version__ as _version
from nnTreeVB.data import evolve_seqs_full_homogeneity as\
        evolve_sequences
from nnTreeVB.data import simulate_tree
from nnTreeVB.data import build_tree_from_nwk
from nnTreeVB.data import get_postorder_branches
from nnTreeVB.data import get_postorder_branche_names
from nnTreeVB.data import TreeSeqCollection
from nnTreeVB.data import build_msa_categorical

from nnTreeVB.parse_config import parse_config

from nnTreeVB.utils import timeSince
from nnTreeVB.utils import write_conf_packages
from nnTreeVB.utils import update_sim_parameters
from nnTreeVB.utils import dict_to_numpy, dict_to_tensor
from nnTreeVB.utils import dump, load

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

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning) 
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings('ignore', 
        "r'All-NaN (slice|axis) encountered'")

__author__ = "amine remita"


"""
nntreevb.py is the main program that uses the VB_nnTree 
model with different substitution models (JC69, K80, HKY 
and GTR). The experiment of fitting the model can be done for
<fit.nb_rep_fit> times.
The program can use a sequence alignment from a Fasta file or
simulate a new sequence alignment using evolutionary parameters
defined in the config file.

Once the nnTreeVB package is installed, you can run nntreevb.py 
using this command line:

# nntreevb.py -c nntreevb_conf_template.ini [-s seed -j job_name]

where <nntreevb_conf_template.ini> is the config file, and 
seed (int) and job_name are optional

The program is adapted from evovgm.py ((c) remita 2022, 
MIT license)
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
    parser.add_argument('-j', '--job-name', type=str)
    parser.add_argument('--version', action='version',
                    version='nnTreeVB {version}'.format(
                        version=_version))

    cmd_args = parser.parse_args()

    config_file = cmd_args.config_file.strip()
    seed = cmd_args.seed
    job_name = cmd_args.job_name # see below for processing it

    print("\nRunning {} with config file {}".format(
        sys.argv[0], config_file), flush=True)

    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    ## Parse config file
    ## #################
    cfg_args, config = parse_config(config_file)

    io  = cfg_args.io
    dat = cfg_args.dat
    mdl = cfg_args.mdl
    fit = cfg_args.fit
    stg = cfg_args.stg
    plt = cfg_args.plt

    verbose = stg.verbose

    # Set the job name
    # ################
    if job_name is None:
        if str(io.job_name).lower() in ["auto", "none"]:
            now = datetime.now()
            io.job_name = now.strftime("%m%d%H%M%S")
        job_name = io.job_name

    config.set("io", "job_name", job_name)

    if verbose:
        print("\tJob name set to {}".format(job_name))
        print("\tVerbose set to {}".format(verbose))

    if seed:
        print("\tSeed set to {}".format(seed))
        config.set("settings", "seed", str(seed))

    # Computing device setting
    # ########################
    device = stg.device

    if "cuda" in device and\
            not torch.cuda.is_available():
        if verbose: 
            print("\nCuda is not available."\
                    " Changing device to 'cpu'")
        device = "cpu"

    elif "mps" in device and\
            not torch.backends.mps.is_available():
        if verbose: 
            print("\nMPS is not available."\
                    " Changing device to 'cpu'")
        device = "cpu"

    if verbose:
        print("\tDevice set to {}".format(device))

    config.set("settings", "device", device)
    device = torch.device(device)

    # dtype configuration
    # ########################
    np_dtype = np.float32
    torch_dtype = torch.float32

    if stg.dtype == "float64":
        np_dtype = np.float64
        torch_dtype = torch.float64

    if verbose:
        print("\tDtype set to {}".format(stg.dtype))

    ## output path 
    ## ###########
    output_path = os.path.join(io.output_path,
            job_name)
    makedirs(output_path, mode=0o700, exist_ok=True)

    pg_path = os.path.join(output_path, "params_grads")
    makedirs(pg_path, mode=0o700, exist_ok=True)

    if dat.sim_data:
        data_path = os.path.join(output_path, "data")
        makedirs(data_path, mode=0o700, exist_ok=True)

    if verbose:
        print("\nExperiment output: {}".format(
            output_path))

    nb_data = dat.nb_rep_data
    nb_fits = fit.nb_rep_fit

    t_seeds = [None] * nb_data
    # seeds for tree replicates, if None, each replicate will
    # have a different tree.

    if seed:
        t_seeds=[seed for i in range(nb_data)]
        if dat.sim_rep_trees:
            t_seeds=[seed+i for i in range(nb_data)]

    fasta_files = []
    tree_files = []

    ## Get Fasta and tree file names
    ## #############################
    if dat.sim_data:
        sim_prefix_file = os.path.join(data_path, 
                "{}_input_".format(job_name))
        # Files paths of simulated data
        # training sequences
        for i in range(nb_data):
            fasta_files.append(sim_prefix_file + \
                    "{}.fasta".format(i))
            tree_files.append(sim_prefix_file + \
                    "{}.nwk".format(i))

        config.set("data","seq_from_file", "True")
        config.set("data","nwk_from_file", "True")
        config.set("io", "seq_file", sim_prefix_file)
        config.set("io", "nwk_file", sim_prefix_file)

    else:
        # Files paths of given FASTA files
        fasta_files.append(io.seq_file)
        tree_files.append(io.nwk_file)

        if not os.path.isfile(fasta_files[0]):
            raise FileNotFoundError("Fasta file {} does not"\
                    " exist".format(fasta_files[0]))

        if not os.path.isfile(tree_files[0]):
            raise FileNotFoundError("Newick file {} does not"\
                    " exist".format(tree_files[0]))

        # Update file paths in config file
        config.set("io", "seq_file", fasta_files[0])
        config.set("io", "nwk_file", tree_files[0])

    config.set("io", "scores_from_file", "True")

    ## Loading results from file
    ## #########################
    results_file = output_path+"/{}_results.pkl".format(
            job_name)

    real_params_np = None
    post_branches = None
    post_branche_names = None

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
        if dat.sim_data:
 
            # Update sim params based on dat.subs_model
            # (useful to update rates using k (k80, hky))
            update_sim_parameters(dat)
            dat.real_params = True

            parallel = Parallel(n_jobs=nb_data, 
                prefer="processes", verbose=verbose)

            trf = all([os.path.isfile(f) for f in tree_files])
            if trf and dat.nwk_from_file:
                if verbose: print("\nExtracting simulated"\
                        " trees from ...")

                tree_data = parallel(delayed(
                    build_tree_from_nwk)(tree_files[i]) \
                            for i in range(nb_data))

                tree_objs = [d[0] for d in tree_data]
                taxa_list = [d[1] for d in tree_data]

            else:
                # if tree files are not given, or 
                # dat.nwk_from_file is false: simulate trees
                # using ete3 populate function
                if verbose:print("\nSimulating new trees...")

                tree_data = parallel(delayed(
                    simulate_tree)(dat.nb_taxa,
                        dat.sim_blengths[i], unroot=True,
                        seed=t_seeds[i])\
                                for i in range(nb_data))

                tree_objs = [d[0] for d in tree_data]
                taxa_list = [d[1] for d in tree_data]

                for i in range(nb_data):
                    tree_objs[i].write(outfile=tree_files[i],
                            format=1)

            #tree_nwk = tree_obj.write(format=1)
            tree_nwks = [tree_objs[i].write(format=1) for i\
                    in range(nb_data)]

            config.set("data", "real_params", "True")

            sqf= all([os.path.isfile(f) for f in fasta_files])
            if not sqf or not dat.seq_from_file:
                if verbose:
                    print("\nSimulating new sequences...")
 
                parallel = Parallel(n_jobs=nb_data, 
                    prefer="processes", verbose=verbose)

                # The order of freqs is different for pyvolve
                # A C G T
                sim_freqs_pyvolve = []
                for sim_freq in dat.sim_freqs:
                    sim_freqs_pyvolve.append([
                        sim_freq[0], 
                        sim_freq[2], 
                        sim_freq[1], 
                        sim_freq[3]])

                all_seqdicts = parallel(delayed(
                    evolve_sequences)(
                        tree_nwks[i],
                        fasta_file=None, # write internal seqs
                        nb_sites=dat.nb_sites,
                        subst_rates=dat.sim_rates[i],
                        state_freqs=sim_freqs_pyvolve[i],
                        return_anc=False,
                        seed=seed,
                        verbose=verbose) 
                    for i in range(nb_data))
 
                # Write fasta
                for i in range(nb_data):
                    seq_taxa = {s:all_seqdicts[i][s] for s in\
                            taxa_list[i]}
                    recs = [SeqRecord(Seq(seq_taxa[taxon]),
                        taxon, '', '') for taxon in seq_taxa]
                    SeqIO.write(recs, fasta_files[i], "fasta")

        if verbose: print("\nLoading data to"\
                " TreeSeqCollection collection...")

        treeseqs = [TreeSeqCollection(fasta_files[i],
            tree_files[i]) for i in range(nb_data)]

        # The tree is sorted
        tree_objs = [treeseqs[i].tree for i in range(nb_data)]

        post_branches = np.array([get_postorder_branches(
            tree_objs[i]) for i in range(nb_data)])
        # post_branches is a numpy array

        post_branche_names = [get_postorder_branche_names(
                tree_objs[i]) for i in range(nb_data)]

        result_data["b_names"] = post_branche_names

        # Transform fitting sequences
        x_data = [torch.from_numpy(build_msa_categorical(
            treeseqs[i], nuc_cat=False, 
            dtype=np_dtype).data).to(device) \
                    for i in range(nb_data)]
        x_patterns = [x_data[i].unique(dim=0, 
                return_counts=True) for i in range(nb_data)]

        if dat.real_params:
            # real_params_np will be used to compare with
            # estimated parameters

            real_params_np = dict_to_numpy(dict(
                b=post_branches,
                t=np.sum(post_branches, axis=1, keepdims=1),
                r=dat.sim_rates,
                f=dat.sim_freqs,
                k=dat.sim_kappa), dtype=np_dtype)

            result_data["real_params"] = real_params_np

            real_params_tensor = [dict_to_tensor(dict(
                b=post_branches[i],
                t=np.sum(post_branches[i], keepdims=1),
                r=dat.sim_rates[i],
                f=dat.sim_freqs[i],
                k=dat.sim_kappa[i]),
                device=device, dtype=torch_dtype)\
                        for i in range(nb_data)]

            for i in range(nb_data):
                for key in real_params_tensor[i]:
                    real_params_tensor[i][key] =\
                        real_params_tensor[i][
                                key].unsqueeze(0)

            # Compute log likelihood of the data given
            # real parameters
            logls = []

            for i in range(nb_data):
                with torch.no_grad():
                    X, X_counts = x_patterns[i] 
                    logl_data = (compute_log_likelihood(
                            copy.deepcopy(tree_objs[i]),
                            X.unsqueeze(0),
                            dat.subs_model,
                            real_params_tensor[i],
                            torch.tensor(
                                [dat.sim_freqs[i]]).to(
                                device=device,
                                dtype=torch_dtype),
                            rescaled_algo=True, 
                            device=device, dtype=torch_dtype)\
                                *X_counts).sum().cpu().numpy()

                    logls.append(logl_data)

                if verbose:
                    print("Log likelihood of the data {}""\
                        using input params: {:.4f}".format(
                                i, logl_data))

            result_data["logl_data"] = np.array(logls)

        # Writing a new config file and package versions
        # Could be used directly 
        conf_file = os.path.join(output_path,
                "{}_conf.ini".format(job_name))
        if not io.scores_from_file or \
                not os.path.isfile(conf_file):
            write_conf_packages(config, conf_file)

        if verbose: print()
        ## Evo model type
        ## ##############
        EvoModelClass = VB_nnTree

        model_arg = {
                "device":device,
                "dtype":torch_dtype,
                **mdl.to_dict()
                }

        fit_arg = {
                "verbose":verbose,
                **fit.to_dict()
                }

        model_args = []
        fit_args = []

        for i in range(nb_data):
            m_arg = copy.deepcopy(model_arg)
            m_arg["tree"] = tree_objs[i]
            model_args.append(m_arg)
            #
            f_arg = copy.deepcopy(fit_arg)
            f_arg["X"] = x_patterns[i][0]
            f_arg["X_counts"] = x_patterns[i][1]
            fit_args.append(f_arg)

        parallel = Parallel(n_jobs=stg.nb_parallel, 
                prefer="processes", verbose=verbose)

        rep_results = parallel(delayed(eval_evomodel)(
            EvoModelClass, model_args[i], fit_args[i]) \
                    for i in range(nb_data) \
                    for j in range(nb_fits))

        rep_results = [rep_results[i:i + nb_fits] for i in\
                range(0, len(rep_results), nb_fits)]
        #
        result_data["rep_results"] = rep_results

        dump(result_data, results_file,
                compress=stg.compress_files)

    if "real_params" in result_data and real_params_np is None:
        real_params_np = result_data["real_params"]

    if "b_names" in result_data and post_branche_names is None:
        post_branche_names = result_data["b_names"]

    #print(rep_results[0][0]["fit_estimates"][0].keys())
    #print(rep_results[0][0]["samples"].keys())

    ## Report and plot results
    ## #######################
    prob_scores = np.array([[rep["fit_probs"] for rep in d ]\
            for d in rep_results])
    #print("The scores {}".format(prob_scores.shape))

    ## Ploting results
    ## ###############
    if verbose: print("\nPlotting...")
 
    logl_data = None
    if "logl_data" in result_data and plt.logl_data: 
        logl_data = result_data["logl_data"]

    plot_elbo_ll_kl(
            prob_scores,
            output_path+"/{}_probs_fig".format(job_name),
            line=logl_data,
            sizefont=plt.size_font,
            usetex=plt.plt_usetex,
            print_xtick_every=plt.print_xtick_every,
            title=None,
            plot_validation=False)

    if fit.save_fit_history and real_params_np:
        history = "fit" # [fit | val]
        estimates = aggregate_estimate_values(rep_results,
                "{}_estimates".format(history))
        #print(estimates.keys())
        #return a dictionary of dictionary of arrays

        ## Distance between estimated paramerters 
        ## and values given in the config file
        plot_fit_estim_distance(
                estimates, 
                real_params_np,
                output_path+"/{}_{}_estim_dist".format(
                    job_name, history),
                scaled=False,
                sizefont=plt.size_font,
                usetex=plt.plt_usetex,
                print_xtick_every=plt.print_xtick_every,
                y_limits=[-0.1, None],
                legend='upper right')
        
        ## Distance between estimated paramerters 
        ## and values given in the config file
        ## scaled between 0 and 1
        plot_fit_estim_distance(
                estimates, 
                real_params_np,
                output_path+"/{}_{}_estim_scaled_dist".format(
                    job_name, history),
                scaled=True,
                sizefont=plt.size_font,
                usetex=plt.plt_usetex,
                print_xtick_every=plt.print_xtick_every,
                y_limits=[-0.1, 1.1],
                legend='upper right')

        ## Correlation between estimated paramerters 
        ## and values given in the config file
        plot_fit_estim_correlation(
                estimates, 
                real_params_np,
                output_path+"/{}_{}_estim_corr".format(
                    job_name, history),
                sizefont=plt.size_font,
                usetex=plt.plt_usetex,
                print_xtick_every=plt.print_xtick_every,
                y_limits=[-1.1, 1.1],
                legend='lower right')

    # Plot weights and grads for each learned distribution and 
    # for each replicate
    if fit.save_grad_stats and fit.save_weight_stats:
        # get the names of learned distributions
        distrs=[d for d in rep_results[0][0]["grad_stats"][0]]
        for n_d in range(nb_data):
            for n_f in range(nb_fits):
                for dist_name in distrs:
                    out_file = pg_path+\
                            "/{}+d{}_f{}+{}".format(
                                dist_name, n_d, n_f, job_name)

                    plot_weights_grads_epochs(
                            rep_results[n_d][n_f],
                            dist_name,
                            out_file,
                            epochs=slice(0, -1), #(0,min_iter)
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
                job_name),
            job_name=job_name,
            real_params=real_params_np,
            branch_names=post_branche_names)

    print("\nFin normale du programme\n")
