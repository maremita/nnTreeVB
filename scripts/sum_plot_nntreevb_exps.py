#!/usr/bin/env python

from nnTreeVB.reports import aggregate_estimate_values
from nnTreeVB.reports import plot_fit_estim_distances
from nnTreeVB.reports import plot_fit_estim_correlations
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

    try:
        eval_codes = json.loads(config.get("evaluation",
            "eval_codes"))
    except:
        eval_codes = None

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

    if eval_codes:
        eval_code_combins = dictLists2combinations(eval_codes)
        # [(('d','jc69'), ('l','100'), ('t','8')), 
        #  (('d','jc69'), ('l','100'), ('t','16')), ... ]
        name_combins = [str(j)+"_"+"_".join(["".join(i)\
                for i in p])\
                for j, p in enumerate(eval_code_combins)]
        # ['0_djc69_l100_t8', '1_djc69_l100_t16', ... ]
    else:
        name_combins = [str(i) for i in range(nb_combins)]

    assert len(eval_combins) == len(name_combins)

    #print(name_combins)
    #sys.exit() 

    history = "fit" # [fit | val]
    estimate_exps = dict()
    sim_param_exps = dict()

    for ind, eval_combin in enumerate(eval_combins):
        exp_name = "{}".format(name_combins[ind])
 
        res_file = os.path.join(output_dir, 
                "{}/{}_results.pkl".format(exp_name, exp_name))

        result_data = load(res_file)
        rep_results=result_data["rep_results"]

        logl_data = None
        if "logl_data" in result_data:
            logl_data=result_data["logl_data"]

        estimates = aggregate_estimate_values(
                rep_results,
                "{}_estimates".format(history),
                report_n_epochs)

        estimate_exps[exp_name] = estimates
        sim_param_exps[exp_name] = result_data["sim_params"]

    for ind, eval_code in enumerate(eval_codes):
        
        output_case = os.path.join(output_sum, eval_code)
        makedirs(output_case, mode=0o700, exist_ok=True)

        # Get unique combinations for each ind
        combins = defaultdict(list)
        for exp_name in name_combins:
            xp = exp_name.split("_")
            xp.pop(0) # remove the indice of the combination
            combins["_".join([xp[i] for i,_ in\
                    enumerate(xp) if i!=ind])].append(exp_name)
        #{'djc69_t8':['0_djc69_l100_t8','2_djc69_l1000_t8']...

        # Get the estimates data for each case
        estim_combins = {c:[estimate_exps[x] for x in\
                combins[c]] for c in combins}

        # Plot estimate distances and correlations for each
        # unique case
 
        for combin in combins:
            out_file = os.path.join(output_case,
                    "{}_estim_dist_itr{}".format(combin, 
                        report_n_epochs))

            plot_fit_estim_distances(
                    estim_combins[combin],
                    combins[combin],
                    sim_param_exps,
                    out_file,
                    sizefont=size_font,
                    print_xtick_every=print_xtick_every,
                    y_limits=[-0.1, 1.1],
                    usetex=False,
                    legend=legend_dist,
                    title=None)

            out_file = os.path.join(output_case, 
                    "{}_estim_corr_itr{}".format(combin,
                        report_n_epochs))

            plot_fit_estim_correlations(
                    estim_combins[combin],
                    combins[combin],
                    sim_param_exps,
                    out_file,
                    sizefont=size_font,
                    print_xtick_every=print_xtick_every,
                    y_limits=[-1.1, 1.1],
                    usetex=False,
                    legend=legend_corr,
                    title=None)
