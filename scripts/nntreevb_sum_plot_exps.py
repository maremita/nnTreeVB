#!/usr/bin/env python

from nnTreeVB import __version__ as _version
from nnTreeVB.reports import aggregate_estimate_values
from nnTreeVB.reports import aggregate_sampled_estimates
from nnTreeVB.reports import summarize_sampled_estimates
from nnTreeVB.reports import plot_elbos_lls_kls
from nnTreeVB.reports import plot_fit_estim_statistics
from nnTreeVB.reports import compute_samples_statistics
from nnTreeVB.reports import violinplot_samples_statistics
from nnTreeVB.reports import plot_grouped_statistics
from nnTreeVB.utils import dictLists2combinations
from nnTreeVB.utils import str2list
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

    print("Summarizing {} experiments...".format(jobs_code),
            flush=True)

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
    save_fit_history = config.getboolean(
            "hyperparams", "save_fit_history", fallback=False)
    nb_parallel = config.getint("evaluation", "nb_parallel",
            fallback=4)
    verbose = check_verbose(config.get("settings", "verbose",
        fallback=1))
    print_xtick_every = config.getint("plotting",
            "print_xtick_every", fallback=100)
    size_font = config.getint("plotting", "size_font",
            fallback=16)
    plt_usetex = config.getboolean("plotting",
            "plt_usetex", fallback=False)

    # y limits
    y_limits = dict()
    y_limits["Dists"] = str2list(config.get("plotting",
        "ylim_dists", fallback="-0.01,None"), cast=float)
    y_limits["Scaled_dists"]= str2list(config.get("plotting",
        "ylim_scaled_dists", fallback="-0.01,1.01"),
        cast=float)
    y_limits["Corrs"] = str2list(config.get("plotting",
        "ylim_corrs", fallback="-1.01,1.01"), cast=float)
    y_limits["Ratios"] = str2list(config.get("plotting",
        "ylim_ratios", fallback="-0.01,None"), cast=float)

    # Legends positions
    legends = dict()
    legends["Elbos"] = config.get("plotting",
        "legend_elbos", fallback="best")
    legends["Dists"] = config.get("plotting",
        "legend_dists", fallback="best")
    legends["Scaled_dists"] = config.get("plotting",
        "legend_scaled_dists", fallback="best")
    legends["Corrs"] = config.get("plotting",
        "legend_corrs", fallback="lower right")
    legends["Ratios"] = config.get("plotting",
        "legend_ratios", fallback="best")

    ## Output directories
    ## ##################
    output_dir = os.path.join(output_eval, "exp_outputs/",
            jobs_code)

    now_str = datetime.now().strftime("%m%d%H%M") 
    output_sum = os.path.join(output_dir,
            "summarize_{}".format(now_str))
            #"summarize")
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
    metrics_exps = dict()

    for ind, eval_combin in enumerate(eval_combins):
        exp_name = name_combins[ind]
        #print(exp_name) # an atomic evaluation

        res_file = os.path.join(output_dir, 
                "{}/{}_results.pkl".format(exp_name,
                    exp_name))

        result_data = load(res_file)
        rep_results=result_data["rep_results"]

        # logl of real data
        if "logl_data" in result_data:
            logl_data_exps[exp_name] = \
                    result_data["logl_data"]
        else:
            logl_data_exps[exp_name]=None

        #
        if "real_params" in result_data:
            real_param_exps[exp_name] =\
                    result_data["real_params"]
        # 
        if save_fit_history:
            prob_scores = [[rep["fit_probs"] for rep in d]\
                    for d in rep_results]
            # get min number of epoch of all reps 
            # (maybe some reps stopped before max_iter)
            # to slice the list of epochs with the same length 
            # and be able to cast the list in ndarray
            prob_exps[exp_name] = np.array(
                    prob_scores)[...,:report_n_epochs] 
            #print(prob_scores.shape)
            # (nb_data, nb_reps, nb_measures, nb_epochs)

            estimates_exps[exp_name] =\
                    aggregate_estimate_values(
                        rep_results,
                        "{}_estimates".format(history),
                        report_n_epochs)
        #
        samples_exps[exp_name] = aggregate_sampled_estimates(
                rep_results, "samples")

        #
        metrics_exps[exp_name] = compute_samples_statistics(
                samples_exps[exp_name],
                real_param_exps[exp_name])

    for ind, eval_code in enumerate(eval_codes):
        print("\nSummarizing {} results".format(eval_code),
                flush=True)

        # Get unique combinations for each ind
        combins = defaultdict(list)
        for exp_name in name_combins:
            xp = exp_name.split("_")
            combins["_".join([xp[i] for i, _ in\
                    enumerate(xp) if i!=ind])].append(
                            exp_name)
        #print(combins)
        #{'d-jc69_t-8':['d-jc69_l-100_t-8',
        #    'd-jc69_l-1000_t-8']...
        a_key = list(combins.keys())[0] # l-100_t-8_m-jc69
        x_names = [c.split("_")[ind] for c in combins[a_key]]
        #print(x_names)
        # example of x_names: ['d-jc69', 'd-gtr']

        # Get second level combinations
        combins2 = dict()
        for exp_name in name_combins:
            xp = exp_name.split("_")

            name1 = "_".join([xp[i] 
                for i,_ in enumerate(xp) if i!=ind])
            # "l-500_t-4"
            name1_splt = name1.split("_") # ['l-500', 't-4']
            for j, v in enumerate(name1_splt):
                name2 = "_".join([name1_splt[i]
                    for i,_ in enumerate(name1_splt) if i!=j])

                if not name2 in combins2:
                    combins2[name2] = defaultdict(list)
                combins2[name2][v].append(exp_name)
        #print(combins2)

        # Get the logl of sim data for each case
        logl_data_combins = {c:[logl_data_exps[x] for x in\
                combins[c]] for c in combins}
        # If all values None, set logl_data_combins to None
        lvs = list(logl_data_combins.values())
        if lvs.count(None) == len(lvs): logl_data_combins=None

        parallel = Parallel(n_jobs=nb_parallel, 
                prefer="processes", verbose=verbose)

        if save_fit_history:
            output_fit = os.path.join(output_sum, 
                    eval_code,"fitting")
            makedirs(output_fit, mode=0o700, exist_ok=True)

            # Get the prob results for each case
            prob_combins = {c:[prob_exps[x] for x in\
                    combins[c]] for c in combins}

            # Plot elbos, logls and kls
            print("\tPloting elbos, logls and kls...", 
                    flush=True)

            parallel(delayed(plot_elbos_lls_kls)(
                prob_combins[combin],
                combins[combin],
                x_names,
                #
                out_file=os.path.join(output_fit,
                    "{}_estim_probs".format(combin)),
                lines=logl_data_combins[combin],
                title=None,
                legend=legends["Elbos"],
                #
                plot_validation=False,
                usetex=plt_usetex,
                sizefont=size_font,
                print_xtick_every=print_xtick_every) \
                        for combin in combins)

            # Get the estimates results for each case
            estim_combins = {c:[estimates_exps[x] for x in\
                    combins[c]] for c in combins}

            #Plot estimate distances and correlations for each
            # unique case 
            print("\tPlotting distances and correlations"
                    " of fit esimates history...", flush=True)

            parallel(delayed(plot_fit_estim_statistics)(
                estim_scores=estim_combins[combin],
                exp_values=combins[combin],
                x_names=x_names,
                sim_param_exps=real_param_exps,
                #
                dist_out_file=os.path.join(output_fit,
                    "{}_estim_dists".format(combin)),
                scaled_dist_out_file=os.path.join(output_fit,
                    "{}_estim_scaled_dists".format(combin)),
                dist_legend=legends["Dists"],
                dist_title=None,
                #
                corr_out_file=os.path.join(output_fit, 
                    "{}_estim_corrs".format(combin)),
                corr_legend=legends["Corrs"],
                corr_title=None,
                #
                usetex=plt_usetex,
                sizefont=size_font,
                print_xtick_every=print_xtick_every) \
                        for combin in combins)

        print("\tSummarizing probs and samplinge estimates...",
                flush=True)
        sample_combins = {c:[samples_exps[x] for x in\
                combins[c]] for c in combins}
        #print(sample_combins.keys())
        
        output_samples = os.path.join(output_sum, 
                "{}".format(eval_code), "sampling")
        makedirs(output_samples, mode=0o700, exist_ok=True)

        out_file = os.path.join(output_samples,
                "{}_samples".format(eval_code))

        summarize_sampled_estimates(
            sample_combins,
            combins,
            x_names,
            real_param_exps,
            out_file,
            logl_data_combins)
 
        print("\tPlotting distances and correlations of"
                " sampled estimates", flush=True)

        ## Violin plots
        output_violin = os.path.join(output_samples, 
                "violinplots")
        makedirs(output_violin, mode=0o700, exist_ok=True)

        metric_combins = {c:[metrics_exps[x] for x in\
                combins[c]] for c in combins}
        #print(metric_combins.keys())
        # ['l-500_t-4','l-500_t-8','l-1000_t-4','l-1000_t-8']

        parallel(delayed(
            violinplot_samples_statistics)(
                metric_scores=metric_combins[combin],
                exp_names=combins[combin],
                x_names=x_names,
                output_path=os.path.join(output_violin,
                    "{}_samples_".format(combin)),
                #
                y_limits={}, # use default values
                usetex=plt_usetex,
                sizefont=size_font) for combin in combins)

        # Line plots
        output_lines = os.path.join(output_samples, 
                "lineplots")
        makedirs(output_lines, mode=0o700, exist_ok=True)

        metric_combins2 = {n2:{v:[metrics_exps[x] 
            for x in combins2[n2][v]] 
            for v in combins2[n2]} 
            for n2 in combins2}
        #print(metric_combins2.keys())
        # ['t-4', 'l-500', 't-8', 'l-1000']

        parallel(delayed(
            plot_grouped_statistics)(
                metric_scores=metric_combins2[combin2],
                x_names=x_names,
                output_path=os.path.join(output_lines,
                    "{}_samples_".format(combin2)),
                #
                legends=legends,
                y_limits=y_limits,
                usetex=plt_usetex,
                sizefont=size_font) for combin2 in combins2)

    print("\nFin normale du programme\n", flush=True)
