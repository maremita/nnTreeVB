#!/usr/bin/env python

from nnTreeVB.reports import plot_elbos_lls_kls
from nnTreeVB.reports import aggregate_estimate_values
from nnTreeVB.reports import aggregate_sampled_estimates
from nnTreeVB.reports import plot_fit_estim_distances
from nnTreeVB.reports import plot_fit_estim_correlations
from nnTreeVB.reports import summarize_sampled_estimates
from nnTreeVB.utils import dictLists2combinations

import sys
#import os
import os.path
from os import makedirs
import copy
import configparser
from datetime import datetime
import json
import argparse
from collections import defaultdict

import numpy as np
from joblib import dump, load

__author__ = "amine"

"""
python sum_plot_nntreevb_exps.py nntreevb_exps_config.ini\
        jobs_code
"""

if __name__ == '__main__':

    # Parse command line
    ####################
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str,
            required=True)
    parser.add_argument('-j', '--job-code', type=str,
            required=True)
    cmd_args = parser.parse_args()

    config_file = cmd_args.config_file
    jobs_code = cmd_args.job_code

    print("Summarizing {} experiments...\n".format(jobs_code))

    ## Fetch argument values from ini file
    ## ###################################
    config = configparser.ConfigParser(
            interpolation=\
                    configparser.ExtendedInterpolation())
 
    with open(config_file, "r") as cf:
        config.read_file(cf)

    max_iter = config.getint("evaluation", "max_iter")
    report_n_epochs = config.getint("evaluation",
            "report_n_epochs", fallback=max_iter)

    print_xtick_every = 100
    size_font = 16
    legend_elbo = 'lower right'
    legend_dist = False
    legend_corr = 'lower right'

    # Main output folder (contains configs, output and jobs)
    output_eval = config.get("evaluation", "output_eval")

    evaluations = json.loads(config.get("evaluation",
        "evaluations"))

    eval_codes = json.loads(config.get("evaluation",
        "eval_codes"))

    ## Output directories
    ## ##################
    output_dir = os.path.join(output_eval, "exp_outputs/",
            jobs_code)

    #config_dir = os.path.join(output_eval,"exp_configs/",
    #        jobs_code)

    output_sum = os.path.join(output_dir,"summarize")
    makedirs(output_sum, mode=0o700, exist_ok=True)

    #
    eval_combins = dictLists2combinations(evaluations)
    nb_combins = len(eval_combins)

    eval_code_combins = dictLists2combinations(eval_codes)
    # [(('d','jc69'), ('l','100'), ('t','8')), 
    #  (('d','jc69'), ('l','100'), ('t','16')), ... ]
    name_combins = [str(j)+"_"+"_".join(["".join(i)\
            for i in p])\
            for j, p in enumerate(eval_code_combins)]
    # ['0_djc69_l100_t8', '1_djc69_l100_t16', ... ]

    assert len(eval_combins) == len(name_combins)

    #print(name_combins)
    #sys.exit() 

    history = "fit" # [fit | val]
    prob_exps = dict()
    logl_data_exps = dict()
    estimates_exps = dict()
    sim_param_exps = dict()
    samples_exps = dict()

    for ind, eval_combin in enumerate(eval_combins):
        exp_name = "{}".format(name_combins[ind])
 
        res_file = os.path.join(output_dir, 
                "{}/{}_results.pkl".format(exp_name,
                    exp_name))

        result_data = load(res_file)
        rep_results=result_data["rep_results"]

        the_scores = [result["fit_probs"] for\
                result in rep_results]

        # get min number of epoch of all reps 
        # (maybe some reps stopped before max_iter)
        # to slice the list of epochs with the same length 
        # and be able to cast the list in ndarray
        the_scores=np.array(the_scores)[...,:report_n_epochs] 
        #print(the_scores.shape)
        # (nb_reps, nb_measures, nb_epochs)

        prob_exps[exp_name] = the_scores

        logl_data = None
        if "logl_data" in result_data:
            logl_data_exps[exp_name]=result_data["logl_data"]

        estimates_exps[exp_name] = aggregate_estimate_values(
                rep_results,
                "{}_estimates".format(history),
                report_n_epochs)

        sim_param_exps[exp_name] = result_data["sim_params"]

        samples_exps[exp_name] = aggregate_sampled_estimates(
                rep_results, "samples")

    for ind, eval_code in enumerate(eval_codes):
 
        output_case = os.path.join(output_sum, eval_code)
        makedirs(output_case, mode=0o700, exist_ok=True)

        # Get unique combinations for each ind
        combins = defaultdict(list)
        for exp_name in name_combins:
            xp = exp_name.split("_")
            xp.pop(0) # remove the indice of the combination
            combins["_".join([xp[i] for i,_ in\
                    enumerate(xp) if i!=ind])].append(
                            exp_name)
        #{'djc69_t8':['0_djc69_l100_t8','2_djc69_l1000_t8']...
        akey = list(combins.keys())[0]
        x_names = [c.split("_")[ind+1] for c in combins[akey]]

        # Get the prob results for each case
        prob_combins = {c:[prob_exps[x] for x in\
                combins[c]] for c in combins}

        # Get the logl of sim data for each case
        logl_data_combins = {c:[logl_data_exps[x] for x in\
                combins[c]] for c in combins}

        # Get the estimates results for each case
        estim_combins = {c:[estimates_exps[x] for x in\
                combins[c]] for c in combins}

        sample_combins = {c:[samples_exps[x] for x in\
                combins[c]] for c in combins}
        
        out_file = os.path.join(output_sum,
                "{}_sampling".format(eval_code))

        summarize_sampled_estimates(
            sample_combins,
            combins,
            x_names,
            sim_param_exps,
            out_file,
            logl_data_combins)

        # Plot estimate distances and correlations for each
        # unique case 

        #for combin in combins:

        #    # Probabilities (elbo, logl, kl)
        #    out_file = os.path.join(output_case,
        #            "{}_probs_fig_itr{}".format(combin, 
        #                report_n_epochs))

        #    plot_elbos_lls_kls(
        #            prob_combins[combin],
        #            combins[combin],
        #            out_file,
        #            lines=logl_data_combins[combin],
        #            sizefont=size_font,
        #            print_xtick_every=print_xtick_every,
        #            title=None,
        #            legend=legend_elbo,
        #            plot_validation=False)

        #    # Distances of estimates with sim params
        #    out_file = os.path.join(output_case,
        #            "{}_estim_dist_itr{}".format(combin, 
        #                report_n_epochs))

        #    plot_fit_estim_distances(
        #            estim_combins[combin],
        #            combins[combin],
        #            sim_param_exps,
        #            out_file,
        #            sizefont=size_font,
        #            print_xtick_every=print_xtick_every,
        #            y_limits=[-0.1, 1.1],
        #            usetex=False,
        #            legend=legend_dist,
        #            title=None)

        #    # Correlations of estimates with sim params
        #    out_file = os.path.join(output_case, 
        #            "{}_estim_corr_itr{}".format(combin,
        #                report_n_epochs))

        #    plot_fit_estim_correlations(
        #            estim_combins[combin],
        #            combins[combin],
        #            sim_param_exps,
        #            out_file,
        #            sizefont=size_font,
        #            print_xtick_every=print_xtick_every,
        #            y_limits=[-1.1, 1.1],
        #            usetex=False,
        #            legend=legend_corr,
        #            title=None)
