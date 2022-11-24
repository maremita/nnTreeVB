#!/usr/bin/env python

from nnTreeVB import __version__ as _version
from nnTreeVB.reports import aggregate_estimate_values
from nnTreeVB.reports import aggregate_sampled_estimates
from nnTreeVB.reports import summarize_sampled_estimates
from nnTreeVB.reports import plot_probs_distances_correlations
from nnTreeVB.utils import dictLists2combinations
from nnTreeVB.checks import check_verbose

import sys
import os.path
from os import makedirs
import configparser
from datetime import datetime
import json
import argparse
from collections import defaultdict

import numpy as np
from nnTreeVB.utils import load

from joblib import Parallel, delayed

__author__ = "amine"

"""
python nntreevb_sum_plot_exps.py -c nntreevb_conf_exps.ini\
        -j jobs_code
"""

if __name__ == '__main__':

    # Parse command line
    ####################
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str,
            required=True)
    parser.add_argument('-j', '--job-code', type=str,
            required=True)
    parser.add_argument('--version', action='version',
                    version='nnTreeVB {version}'.format(
                        version=_version))

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

    # Main output folder (contains configs, output and jobs)
    output_eval = config.get("evaluation", "output_eval")
    evaluations = json.loads(config.get("evaluation",
        "evaluations"))
    eval_codes = json.loads(config.get("evaluation",
        "eval_codes"))
    max_iter = config.getint("evaluation", "max_iter")
    report_n_epochs = config.getint("evaluation",
            "report_n_epochs", fallback=max_iter)
    nb_parallel = config.getint("settings", "nb_parallel",
            fallback=1)
    verbose = check_verbose(config.get("settings", "verbose",
        fallback=1))

    print_xtick_every = config.getint("plotting",
            "print_xtick_every", fallback=100)
    size_font = config.getint("plotting", "size_font",
            fallback=16)
    plt_usetex = config.getboolean("plotting",
            "plt_usetex", fallback=False)

    legend_elbo = 'lower right'
    legend_dist = False
    legend_corr = 'lower right'

    ## Output directories
    ## ##################
    output_dir = os.path.join(output_eval, "exp_outputs/",
            jobs_code)

    now_str = datetime.now().strftime("%H%M") 
    output_sum = os.path.join(output_dir,
            "summarize")
            #"summarize_{}".format(now_str))
    makedirs(output_sum, mode=0o700, exist_ok=True)

    #
    eval_combins = dictLists2combinations(evaluations)
    nb_combins = len(eval_combins)

    eval_code_combins = dictLists2combinations(eval_codes)
    # [(('d','jc69'), ('l','100'), ('t','8')), 
    #  (('d','jc69'), ('l','100'), ('t','16')), ... ]
    name_combins = ["_".join(["-".join(i) for i in p])\
            for p in eval_code_combins]
    # ['d-jc69_l-100_t-8', 'd-jc69_l-100_t-16', ... ]

    assert len(eval_combins) == len(name_combins)

    history = "fit" # [fit | val]
    prob_exps = dict()
    logl_data_exps = dict()
    estimates_exps = dict()
    real_param_exps = dict()
    samples_exps = dict()

    for ind, eval_combin in enumerate(eval_combins):
        exp_name = name_combins[ind]

        res_file = os.path.join(output_dir, 
                "{}/{}_results.pkl".format(exp_name,
                    exp_name))

        result_data = load(res_file)
        rep_results=result_data["rep_results"]

        #the_scores = [result["fit_probs"] for\
        #        result in rep_results]

        prob_scores = [[rep["fit_probs"] for rep in d ] \
                for d in rep_results]
        # get min number of epoch of all reps 
        # (maybe some reps stopped before max_iter)
        # to slice the list of epochs with the same length 
        # and be able to cast the list in ndarray
        prob_exps[exp_name] = np.array(
                prob_scores)[...,:report_n_epochs] 
        #print(prob_scores.shape)
        # (nb_data, nb_reps, nb_measures, nb_epochs)

        if "logl_data" in result_data:
            logl_data_exps[exp_name]=result_data["logl_data"]
        else:
            logl_data_exps[exp_name]=None

        estimates_exps[exp_name] = aggregate_estimate_values(
                rep_results,
                "{}_estimates".format(history),
                report_n_epochs)

        real_param_exps[exp_name] = result_data["real_params"]

        samples_exps[exp_name] = aggregate_sampled_estimates(
                rep_results, "samples")

    for ind, eval_code in enumerate(eval_codes):
        print("\nSummarizing {} results".format(eval_code))

        output_case = os.path.join(output_sum, eval_code)
        makedirs(output_case, mode=0o700, exist_ok=True)

        # Get unique combinations for each ind
        combins = defaultdict(list)
        for exp_name in name_combins:
            xp = exp_name.split("_")
            combins["_".join([xp[i] for i, _ in\
                    enumerate(xp) if i!=ind])].append(
                            exp_name)
        #{'d-jc69_t-8':['d-jc69_l-100_t-8',
        #    'd-jc69_l-1000_t-8']...
        akey = list(combins.keys())[0] # l-100_t-8_m-jc69
        x_names = [c.split("_")[ind] for c in combins[akey]]
        # example of x_names: ['d-jc69', 'd-gtr']

        # Get the prob results for each case
        prob_combins = {c:[prob_exps[x] for x in\
                combins[c]] for c in combins}

        # Get the logl of sim data for each case
        logl_data_combins = {c:[logl_data_exps[x] for x in\
                combins[c]] for c in combins}
        # If all values None, set logl_data_combins to None
        lvs = list(logl_data_combins.values())
        if lvs.count(None) == len(lvs): logl_data_combins=None

        # Get the estimates results for each case
        estim_combins = {c:[estimates_exps[x] for x in\
                combins[c]] for c in combins}

        sample_combins = {c:[samples_exps[x] for x in\
                combins[c]] for c in combins}

        print("\tSummarizing probs and samplinge stimates...")
        out_file = os.path.join(output_sum,
                "{}_sampling".format(eval_code))

        summarize_sampled_estimates(
            sample_combins,
            combins,
            x_names,
            real_param_exps,
            out_file,
            logl_data_combins)

        # Plot estimate distances and correlations for each
        # unique case 
        print("\tPloting...")

        parallel = Parallel(n_jobs=nb_parallel, 
                prefer="processes", verbose=verbose)

        parallel(delayed(plot_probs_distances_correlations)(
            probs_scores=prob_combins[combin],
            estim_scores=estim_combins[combin],
            exp_values=combins[combin],
            x_names=x_names,
            sim_param_exps=real_param_exps,
            #
            prob_out_file=os.path.join(output_case,
                "{}_probs_fig_itr{}".format(combin, 
                    report_n_epochs)),
            prob_lines=logl_data_combins[combin],
            prob_title=None,
            prob_legend=legend_elbo,
            #
            dist_out_file=os.path.join(output_case,
                "{}_estim_dist_itr{}".format(combin, 
                    report_n_epochs)),
            dist_y_limits=[-0.1, 1.1],
            dist_legend=legend_dist,
            dist_title=None,
            #
            corr_out_file=os.path.join(output_case, 
                "{}_estim_corr_itr{}".format(combin,
                    report_n_epochs)),
            corr_y_limits=[-1.1, 1.1],
            corr_legend=legend_corr,
            corr_title=None,
            #
            plot_validation=False,
            usetex=plt_usetex,
            sizefont=size_font,
            print_xtick_every=print_xtick_every) \
                    for combin in combins)
