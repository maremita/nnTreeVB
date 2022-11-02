#!/usr/bin/env python

from nnTreeVB.utils import dictLists2combinations

#import sys
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
python write_run_nntreevb_exps.py nntreevb_exps_config.ini\
        jobs_code
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
    parser.add_argument('-s', '--seed', type=int,
            required=True)
    parser.add_argument('-j', '--job-code', type=str)
    cmd_args = parser.parse_args()

    config_file = cmd_args.config_file
    seed = cmd_args.seed
    jobs_code = cmd_args.job_code

    if jobs_code:
        scores_from_file = "True"
    else:
        now = datetime.now()
        jobs_code = now.strftime("%m%d%H%M")
        scores_from_file = "False"

    print("Runing {} experiments...\n".format(jobs_code))

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
    # needs to be str
    n_reps = config.get("evaluation", "nb_replicates")

    evaluations = json.loads(config.get("evaluation",
        "evaluations"))

    try:
        eval_codes = json.loads(config.get("evaluation",
            "eval_codes"))
    except:
        eval_codes = None

    ## Remove slurm and evaluation sections from the config
    ## object to not include them in the config files of
    ## nntreevb.py

    if config.has_section("slurm"):
        config.remove_section('slurm')
    config.remove_section('evaluation')

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
    config.set("hyperparams", "nb_replicates", n_reps)

    # config gpu
    set_gpu = ""
    config.set("settings", "device", "cpu")

    if gpus_per_node > 0:
        set_gpu = " --gpus-per-node={}".format(gpus_per_node)
        config.set("settings", "device", "cuda")

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

    # Dump the combinations into a file for post tracking
    eval_dict = {name_combins[i]:e for i, e\
            in enumerate(eval_combins)}

    with open(output_dir+"/eval_combinations.txt", "w") as fh:
        json.dump(eval_dict, fh, indent=2)

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
        cfg_eval.set("settings", "job_name", exp_name)

        # write it on a file
        config_file = os.path.join(config_dir,
                "{}.ini".format(exp_name))

        with open (config_file, "w") as fh:
            cfg_eval.write(fh)

        if run_slurm:
            s_error = os.path.join(job_dir,
                    exp_name+"_%j.err")
            s_output = os.path.join(job_dir, 
                    exp_name+"_%j.out")

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
            cmd = "{} -c {} -s {} &".format(program,
                    config_file, seed)

        res_file = output_dir+ "{}/{}_results.pkl".format(
                exp_name, exp_name)

        if not os.path.isfile(res_file):
            print("\n", exp_name)
            if run_jobs:
                print(cmd)
                os.system(cmd)
