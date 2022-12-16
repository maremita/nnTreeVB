#!/usr/bin/env python

from nnTreeVB import __version__ as _version
from nnTreeVB.utils import dictLists2combinations

import sys
import os
import os.path
from os import makedirs
import copy
import configparser
from datetime import datetime
import json
import argparse

__author__ = "amine"

"""
python nntreevb_write_run_exps.py -c nntreevb_conf_exps.ini\
        -s 42 -j jobs_code
"""

if __name__ == '__main__':

    sb_program = "slurm_exps.sh"
    program = "nntreevb.py"
    jobs_code = None

    # Parse command line
    ####################
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str,
            required=True)
    # TODO: do not require seed
    parser.add_argument('-s', '--seed', type=int,
            required=True)
    parser.add_argument('-j', '--jobs-code', type=str)
    parser.add_argument('--version', action='version',
                    version='nnTreeVB {version}'.format(
                        version=_version))

    cmd_args = parser.parse_args()

    config_file = cmd_args.config_file
    seed = cmd_args.seed
    jobs_code = cmd_args.jobs_code

    if jobs_code:
        scores_from_file = "True"
    else:
        now = datetime.now()
        jobs_code = now.strftime("%m%d%H%M")
        scores_from_file = "False"

    print("Runing {} experiments...\n".format(jobs_code), 
            flush=True)

    ## Fetch argument values from ini file
    ## ###################################
    config = configparser.ConfigParser(
            interpolation=\
                    configparser.ExtendedInterpolation())
 
    with open(config_file, "r") as cf:
        config.read_file(cf)
 
    # Get write_run_rep_exps.py specific sections:
    if config.has_section("slurm"):
        run_slurm = config.getboolean("slurm", "run_slurm")
        # if run_slurm=False: the script will launch the jobs
        # locally
        # elif run_slurm=True:it will launch the jobs on slurm

        account = config.get("slurm", "account")
        mail_user = config.get("slurm", "mail_user")

        # SLURM parameters
        cpus_per_task = config.getint("slurm",
                "cpus_per_task")
        gpus_per_node = config.getint("slurm",
                "gpus_per_node", fallback=0)
        exec_time = config.get("slurm", "exec_time")
        mem = config.get("slurm", "mem")
    else:
        # The script will launch the jobs locally;
        run_slurm = False

    run_jobs = config.getboolean("evaluation", "run_jobs")
    #if run_jobs=False: the script generates the nntreevb
    # config files but it won't launch the jobs.
    # If run_jobs=True: it runs the jobs

    # Main output folder (contains configs, output and jobs)
    output_eval = config.get("evaluation", "output_eval")

    max_iter = config.get("evaluation", "max_iter")

    nb_parallel = config.getint("evaluation", "nb_parallel",
            fallback=4)

    evaluations = json.loads(config.get("evaluation",
        "evaluations"))

    eval_codes = json.loads(config.get("evaluation",
        "eval_codes"))

    ## Remove slurm and evaluation sections from the config
    ## object to not include them in the config files of
    ## nntreevb.py

    if config.has_section("slurm"):
        config.remove_section('slurm')
    config.remove_section('evaluation')

    sim_data = config.getboolean("data", "sim_data",
            fallback=False)

    ## Output directories
    ## ##################
    output_dir = os.path.join(output_eval, "exp_outputs/",
            jobs_code)
    makedirs(output_dir, mode=0o700, exist_ok=True)

    config_dir = os.path.join(output_eval,"exp_configs/",
            jobs_code)
    makedirs(config_dir, mode=0o700, exist_ok=True)

    job_dir = os.path.join(output_eval, "exp_jobs/", 
            jobs_code)
    makedirs(job_dir, mode=0o700, exist_ok=True)

    ## Update options of config file
    ## #############################
    config.set("io", "output_path", output_dir)
    config.set("io", "scores_from_file", scores_from_file)
 
    config.set("hyperparams", "max_iter", max_iter)
    config.set("settings", "nb_parallel", str(nb_parallel))

    # config gpu
    set_gpu = ""

    if gpus_per_node > 0 and run_slurm:
        set_gpu = " --gpus-per-node={}".format(gpus_per_node)
        config.set("settings", "device", "cuda")

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

    # Dump the combinations into a file for post tracking
    eval_dict = {name_combins[i]:e for i, e\
            in enumerate(eval_combins)}

    eval_comb_file = os.path.join(output_dir,
            "eval_combinations.txt")

    with open(eval_comb_file, "w") as fh:
        json.dump(eval_dict, fh, indent=2)

    n_jobs = 0
    n_runs = 0
    # Start evaluations
    for ind, eval_combin in enumerate(eval_combins):
        # Create a new copy of config object to keep default
        # options
        cfg_eval = copy.deepcopy(config)

        #
        for eval_option in eval_combin:
            option, section = eval_option[0].split("@")
            cfg_eval.set(section, option, str(eval_option[1]))

        exp_name = "{}".format(name_combins[ind])
        cfg_eval.set("io", "job_name", exp_name)
 
        output_path = os.path.join(output_dir, exp_name)

        # Update file paths if sim_data is True
        if sim_data:
            data_path = os.path.join(output_path, "data")
            makedirs(data_path, mode=0o700, exist_ok=True)

            fasta_file = os.path.join(data_path,
                    "{}_input_".format(exp_name))
            tree_file = os.path.join(data_path,
                    "{}_input_".format(exp_name))

            cfg_eval.set("io", "seq_file", fasta_file)
            cfg_eval.set("io", "nwk_file", tree_file)

        # write it on a file
        config_file = os.path.join(config_dir,
                "{}.ini".format(exp_name))

        with open (config_file, "w") as fh:
            cfg_eval.write(fh)

        if run_slurm:
            s_error = os.path.join(job_dir,
                    "%j_"+exp_name+".err")
            s_output = os.path.join(job_dir, 
                    "%j_"+exp_name+".out")

            cmd = "sbatch"\
                    " --account={}"\
                    " --mail-user={}"\
                    " --job-name={}"\
                    " --export=PROGRAM={},"\
                    "CONF_file={},SEED={},"\
                    "{}"\
                    " --cpus-per-task={}"\
                    " --mem={}"\
                    " --time={}"\
                    " --error={}"\
                    " --output={}"\
                    " {}".format(
                            account,
                            mail_user,
                            exp_name,
                            program, config_file, seed,
                            set_gpu,
                            cpus_per_task,
                            mem,
                            exec_time, 
                            s_error,
                            s_output,
                            sb_program)
        else:
            ## ######################################
            ## WARNING: 
            ## The script will run several
            ## tasks locally in the background.
            ## Make sure that you have the necessary 
            ## resources on your machine.
            ## It can crash the system.
            ## You can modify the grid of parameters
            ## to be evaluated in the config file to 
            ## reduce the number of scenarios.
            ## ######################################
            s_error = os.path.join(job_dir,
                    exp_name+".err")
            s_output = os.path.join(job_dir, 
                    exp_name+".out")

            cmd = "{} -c {} -s {} 2>{} >{} &".format(program,
                    config_file, seed, s_error, s_output)

        res_file = os.path.join(output_path,
                "{}_results.pkl".format(exp_name))

        n_jobs += 1

        if not os.path.isfile(res_file):
            print("\n", exp_name, flush=True)
            if run_jobs:
                print(cmd, flush=True)
                os.system(cmd)
                n_runs += 1

    print("\n {}/{} launched jobs".format(n_runs, n_jobs))
    print("\nFin normal du programme {}".format(sys.argv[0]))
