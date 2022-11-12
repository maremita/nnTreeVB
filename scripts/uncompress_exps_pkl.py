#!/usr/bin/env python

from nnTreeVB.utils import dictLists2combinations

import sys
import os.path
import shutil
import configparser
import json
import argparse

from joblib import dump, load
from joblib import Parallel, delayed

__author__ = "amine"

"""
python uncompress_exps_pkl.py -c nntreevb_conf_exps.ini\
        -j jobs_code [-n n_parallel]
"""

def uncompress_pkls(eval_combin, exp_name, output_dir):

    run = 0
    output_path = os.path.join(output_dir, exp_name)

    res_file = os.path.join(output_path,
            "{}_results.pkl".format(exp_name))

    bkp_file = os.path.join(output_path,
            "{}_results_bkp.pkl".format(exp_name))

    if not os.path.isfile(res_file):
        print("{} doesn't exist!\n".format(res_file))
    else:
        # backup the file
        # shutil.copyfile(src, dst)
        shutil.copyfile(res_file, bkp_file)

        result_data = load(res_file)
        dump(result_data, res_file, compress=False)

        run = 1

    return run


if __name__ == '__main__':

    # Parse command line
    ####################
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str,
            required=True)
    parser.add_argument('-j', '--job-code', type=str,
            required=True)
    parser.add_argument('-n', '--n-parallel', type=int,
            required=False, default=1)
    cmd_args = parser.parse_args()

    config_file = cmd_args.config_file
    jobs_code = cmd_args.job_code
    n_parallel = cmd_args.n_parallel

    print("Uncompressing pkl files of {} ...\n".format(
         jobs_code))

    ## Fetch argument values from ini file
    ## ###################################
    config = configparser.ConfigParser(
            interpolation=\
                    configparser.ExtendedInterpolation())
 
    with open(config_file, "r") as cf:
        config.read_file(cf)
 
    # Main output folder (contains configs, output and jobs)
    output_eval = config.get("evaluation", "output_eval")

    ## Output directory
    output_dir = os.path.join(output_eval, "exp_outputs/",
            jobs_code)

    evaluations = json.loads(config.get("evaluation",
        "evaluations"))
    eval_codes = json.loads(config.get("evaluation",
        "eval_codes"))
    #
    eval_combins = dictLists2combinations(evaluations)

    eval_code_combins = dictLists2combinations(eval_codes)
    # [(('d','jc69'), ('l','100'), ('t','8')), 
    #  (('d','jc69'), ('l','100'), ('t','16')), ... ]
    name_combins = ["_".join(["-".join(i) for i in p])\
            for p in eval_code_combins]
    # ['d-jc69_l-100_t-8', 'd-jc69_l-100_t-16', ... ]

    assert len(eval_combins) == len(name_combins)

    # Start uncompressing files
    parallel = Parallel(n_jobs=n_parallel, 
            prefer="processes", verbose=1)

    n_runs = parallel(delayed(uncompress_pkls)(
        eval_combin, name_combins[i], output_dir) for\
                i, eval_combin in enumerate(eval_combins))

    n_runs = sum(n_runs)

    print("\n {} uncompressed file".format(n_runs))
    print("\nFin normal du programme {}".format(sys.argv[0]))
