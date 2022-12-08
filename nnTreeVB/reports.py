from nnTreeVB.utils import compute_corr
from nnTreeVB.utils import compute_estim_stats
from nnTreeVB import __version__ as _version

from collections import defaultdict
import os.path

import numpy as np
import pandas as pd

from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist

import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

__author__ = "amine remita"


estim_list = ["b", "t", "b1", "r", "f", "k"]

stat_names = ["mean", "ci_min", "ci_max", "var", "min", "max"]

estim_names = {
        "b":"Branch lengths",
        "t":"Tree length",
        "r":"Substitution rates", 
        "f":"Relative frequencies",
        "k":"Kappa"}

estim_colors = { 
        "b":"#226E9C",
        "t":"#595F73", #2a9d8f
        "r":"#D12959", 
        "f":"#40AD5A",
        "k":"#FFAA00"}

prob_names = {
        "elbo":"ELBO", 
        "logl":"LogL",
        "logprior":"LogPrior",
        "logq":"LogQ",
        "kl_qprior":"KL_QPrior"}

rates_list = ["AG", "AC", "AT", "GC", "GT", "CT"]
freqs_list = ["A", "G", "C", "T"]

line_color = "#c13e4c"
elbo_color = "#3EC1B3"  # green
ll_color =   "#226E9C"  # darker blue
kl_color =   "#7C1D69"  # pink

elbo_color_v = "#6BE619"
ll_color_v =   "#009ADE" # light blue
kl_color_v =   "#AF58BA"  # light pink

def plot_weights_grads_epochs(
        data,
        module_name,
        out_file=False,
        epochs=slice(0,-1),
        fig_size=(10, 4),
        sizefont=16,
        usetex=False,
        legend='best'):

    fig_format= "png"
    fig_dpi = 150

    if out_file:
        fig_file = out_file+"."+fig_format

    plt.rcParams.update({'font.size':sizefont,
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    params = list(data["grad_stats"][0][module_name].keys())
    nb_params = len(params)

    w_means = defaultdict(list)
    w_vars = defaultdict(list)

    for e in data["weight_stats"]:
        submodule = e[module_name]

        for param in submodule:
            w_means[param].append(submodule[param]["mean"])
            w_vars[param].append(submodule[param]["var"])

    g_means = defaultdict(list)
    g_vars = defaultdict(list)

    for e in data["grad_stats"]: # e for epoch, is a dict
        submodule = e[module_name]

        for param in submodule:
            g_means[param].append(submodule[param]["mean"])
            g_vars[param].append(submodule[param]["var"])

    f, axs = plt.subplots(nb_params, 2, 
            figsize=(fig_size[0], fig_size[1]*nb_params))

    if len(axs.shape)>1:
        axs = np.concatenate(axs)

    nb_iters = len(data["grad_stats"][epochs])
    start = epochs.start + 1
    end = epochs.stop
    if end == -1: end = nb_iters
    x = [j for j in range(start, end+1)]

    ind = 0
    for i, param in enumerate(params): 
        # weights
        w_m = np.array(w_means[param])[epochs]
        w_s = np.sqrt(np.array(w_vars[param]))[epochs]

        axs[ind].plot(x, w_m, color="green")

        axs[ind].fill_between(x,
                w_m-w_s,
                w_m+w_s,
                color="green",
                alpha=0.2)

        axs[ind].grid(zorder=-1)
        axs[ind].set_ylabel(param)
        #axs[ind].legend()

        # gradients
        g_m = np.array(g_means[param])[epochs]
        g_s = np.sqrt(np.array(g_vars[param]))[epochs]

        axs[ind+1].plot(x, g_m, color="red")

        axs[ind+1].fill_between(x,
            g_m-g_s,
            g_m+g_s,
            alpha=0.2, color="red")

        axs[ind+1].grid(zorder=-1)
        #axs[ind+1].legend()
        
        if ind < 2:
            axs[ind].set_title("Parameters")
            axs[ind+1].set_title("Gradients")

        if ind >= nb_params-1:
            axs[ind].set_xlabel("Iterations")
            axs[ind+1].set_xlabel("Iterations")

        ind +=2

    plt.suptitle(module_name)
 
    if out_file:
        plt.savefig(fig_file, bbox_inches="tight", 
                format=fig_format, dpi=fig_dpi)
    else:
        plt.show()

    plt.close(f)


def plot_elbo_ll_kl(
        scores,
        out_file,
        line=None,
        sizefont=16,
        usetex=False,
        print_xtick_every=10,
        legend='best',
        title=None,
        plot_validation=False):

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    kl_fit_finite = np.isfinite(scores[...,2,:]).all()
    kl_val_finite = False
    if plot_validation:
        kl_val_finite = np.isfinite(scores[...,5,:]).all()

    plt.rcParams.update({'font.size':sizefont, 
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    f, ax = plt.subplots(figsize=(8, 5))
    
    if kl_fit_finite or kl_val_finite:
        ax2 = ax.twinx()

    nb_iters = scores.shape[-1]
    x = [j for j in range(1, nb_iters+1)]

    ax.set_rasterization_zorder(0)

    sshp = scores.shape
    mx = tuple([i for i in range(len(sshp))\
            if i < -2%len(sshp)])

    ax.plot(x, scores[...,0,:].mean(mx),
            "-", color=elbo_color, 
            label="ELBO", zorder=6) # ELBO train
    
    ax.fill_between(x,
            scores[...,0,:].mean(mx)-scores[...,0,:].std(mx), 
            scores[...,0,:].mean(mx)+scores[...,0,:].std(mx), 
            color=elbo_color,
            alpha=0.2, zorder=5, interpolate=True)

    ax.plot(x, scores[...,1,:].mean(mx), "-", color=ll_color,
            label="LogL", zorder=4) # LL train

    ax.fill_between(x,
            scores[...,1,:].mean(mx)-scores[...,1,:].std(mx), 
            scores[...,1,:].mean(mx)+scores[...,1,:].std(mx),
            color=ll_color,
            alpha=0.2, zorder=3, interpolate=True)
    
    if kl_fit_finite:
        ax2.plot(x, scores[...,2,:].mean(mx), "-",
                color=kl_color, 
                label="KL_qp",
                zorder=4) # KL train

        ax2.fill_between(x,
                scores[...,2,:].mean(mx)\
                        -scores[...,2,:].std(mx),
                scores[...,2,:].mean(mx)\
                        +scores[...,2,:].std(mx), 
                color=kl_color,
                alpha=0.2, zorder=-6, interpolate=True)

    # plot validation
    if plot_validation:
        ax.plot(x, scores[...,3,:].mean(mx), "-.",
                color=elbo_color_v,
                label="ELBO_val", zorder=2) # ELBO val

        ax.fill_between(x,
                scores[...,3,:].mean(mx)\
                        -scores[...,3,:].std(mx), 
                scores[...,3,:].mean(mx)\
                        +scores[...,3,:].std(mx), 
                color=elbo_color_v,
                alpha=0.1, zorder=1, interpolate=True)

        ax.plot(x, scores[...,4,:].mean(mx), "-.",
                color=ll_color_v,
                label="LogL_val", zorder=0) # LL val
 
        ax.fill_between(x,
                scores[...,4,:].mean(mx)\
                        -scores[...,4,:].std(mx), 
                scores[...,4,:].mean(mx)\
                        +scores[...,4,:].std(mx), 
                color=ll_color_v,
                alpha=0.1, zorder=2, interpolate=True)

        if kl_val_finite:
            ax2.plot(x, scores[...,5,:].mean(mx), "-.",
                    color=kl_color_v,
                    label="KL_qp_val", zorder=2) # KL val
        
            ax2.fill_between(x,
                    scores[...,5,:].mean(mx)\
                            -scores[...,5,:].std(mx), 
                    scores[...,5,:].mean(mx)\
                            +scores[...,5,:].std(mx), 
                    color= kl_color_v,
                    alpha=0.1, zorder=1, interpolate=True)

    if line is not None:
        ml = line.mean(0)
        sl = line.std(0)
        ax.plot(x, np.zeros_like(x)+ml, label="Real LogL",
                color=line_color, linestyle='-')
        ax.fill_between(x, ml-sl, ml+sl,
                    color= line_color,
                    alpha=0.1, zorder=0, interpolate=True)

    #ax.set_zorder(ax2.get_zorder()+1) 
    if kl_fit_finite or kl_val_finite:
        ax.set_frame_on(False)

    ax.set_ylim([None, 0])
    ax.set_xticks([t for t in range(1, nb_iters+1) if t==1 or\
            t % print_xtick_every==0])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("ELBO and Log Likelihood")

    if kl_fit_finite or kl_val_finite:
        ax2.set_ylabel("KL(q|prior)")

    ax.grid(zorder=-1)
    ax.grid(zorder=-1, visible=True, which='minor', alpha=0.1)
    ax.minorticks_on()

    if legend:
        handles,labels = [],[]
        for ax in f.axes:
            for h,l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        plt.legend(handles, labels, loc=legend, framealpha=1,
                facecolor="white", fancybox=True)

    if title:
        plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)

def plot_elbos_lls_kls(
        exp_scores,
        exp_values,
        x_names,
        out_file,
        lines=None,
        sizefont=14,
        usetex=False,
        print_xtick_every=20,
        title=None,
        legend='best',
        plot_validation=False):

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    if not isinstance(exp_scores, np.ndarray):
        exp_scores = np.array(exp_scores)

    nb_evals = len(exp_scores)

    f, axs = plt.subplots(1, nb_evals,
            figsize=(7*nb_evals, 5))

    if nb_evals == 1: axs = [axs]

    plt.rcParams.update({'font.size':sizefont, 
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.07, hspace=0.1)

    nb_iters = exp_scores[0].shape[-1]
    x = [j for j in range(1, nb_iters+1)]
 
    for i, exp_value in enumerate(exp_values):
        scores = exp_scores[i,...]

        sshp = scores.shape
        mx = tuple([i for i in range(len(sshp))\
                if i < -2%len(sshp)])

        kl_fit_finite = np.isfinite(scores[...,2,:]).all()
        kl_val_finite = False
        if plot_validation:
            kl_val_finite = np.isfinite(scores[...,5,:]).all()

        if kl_fit_finite or kl_val_finite:
            ax2 = axs[i].twinx()

        # ELBO train
        axs[i].plot(x, scores[...,0,:].mean(mx), "-",
                color=elbo_color,  label="ELBO", zorder=6) 

        axs[i].fill_between(x,
                scores[...,0,:].mean(mx)\
                        -scores[...,0,:].std(mx), 
                scores[...,0,:].mean(mx)\
                        +scores[...,0,:].std(mx), 
                color=elbo_color,
                alpha=0.2, zorder=5, interpolate=True)

        # LL train
        axs[i].plot(x, scores[...,1,:].mean(mx), "-",
                color=ll_color, label="LogL", zorder=4) 

        axs[i].fill_between(x,
                scores[...,1,:].mean(mx)\
                        -scores[...,1,:].std(mx), 
                scores[...,1,:].mean(mx)\
                        +scores[...,1,:].std(mx),
                color=ll_color,
                alpha=0.2, zorder=3, interpolate=True)

        if kl_fit_finite:
            # KL train
            ax2.plot(x, scores[...,2,:].mean(mx), "-",
                    color=kl_color, label="KL_qp", zorder=4)

            ax2.fill_between(x,
                    scores[...,2,:].mean(mx)\
                            -scores[...,2,:].std(mx),
                    scores[...,2,:].mean(mx)\
                            +scores[...,2,:].std(mx),
                    color=kl_color,
                    alpha=0.2, zorder=-6, interpolate=True)

        # plot validation
        if plot_validation:
            axs[i].plot(x, scores[...,3,:].mean(mx), "-.",
                    color=elbo_color_v,
                    label="ELBO_val", zorder=2) # ELBO val

            axs[i].fill_between(x,
                    scores[...,3,:].mean(mx)\
                            -scores[...,3,:].std(mx),
                    scores[...,3,:].mean(mx)\
                            +scores[...,3,:].std(mx),
                    color=elbo_color_v,
                    alpha=0.1, zorder=1, interpolate=True)

            axs[i].plot(x, scores[...,4,:].mean(mx), "-.",
                    color=ll_color_v,
                    label="LogL_val", zorder=0) # LL val

            axs[i].fill_between(x,
                    scores[...,4,:].mean(mx)\
                            -scores[...,4,:].std(mx),
                    scores[...,4,:].mean(mx)\
                            +scores[...,4,:].std(mx),
                    color=ll_color_v,
                    alpha=0.1, zorder=2, interpolate=True)

            if kl_fit_finite:
                ax2.plot(x, scores[...,5,:].mean(mx), "-.", 
                        color=kl_color_v,
                        label="KL_qp_val", zorder=2) # KL val
                
                ax2.fill_between(x,
                        scores[...,5,:].mean(mx)\
                                -scores[...,5,:].std(mx),
                        scores[...,5,:].mean(mx)\
                                +scores[...,5,:].std(mx),
                        color= kl_color_v,
                        alpha=0.1, zorder=1, interpolate=True)

        #if lines:
        #    axs[i].axhline(y=lines[i], color=line_color,
        #            linestyle='-')

        if lines is not None:
            ml = lines[i].mean(0)
            sl = lines[i].std(0)

            axs[i].plot(x, np.zeros_like(x)+ml,
                    label="Real LogL",
                    color=line_color, linestyle='-')
            axs[i].fill_between(x, ml-sl, ml+sl,
                        color= line_color,
                        alpha=0.1, zorder=0, interpolate=True)

        #axs[i].set_zorder(ax2.get_zorder()+1)
        if kl_fit_finite or kl_val_finite:
            axs[i].set_frame_on(False)

        #axs[i].set_title(x_names[i].split("-")[1])
        axs[i].set_title(x_names[i])
        #axs[i].set_ylim([None, 0])
        #axs[i].set_ylim([-10000, 0])
        axs[i].set_ylim([np.min(np.ma.masked_invalid(
            exp_scores[...,0,:].flatten())), 0])
        axs[i].set_xticks([t for t in range(1, nb_iters+1) if\
                t==1 or t % print_xtick_every==0])
        axs[i].set_xlabel("Iterations")
        axs[i].grid(zorder=-2)
        axs[i].grid(zorder=-2, visible=True, which='minor',
                alpha=0.1)
        axs[i].minorticks_on()

        if kl_fit_finite:
            ax2.set_ylim([ 
                np.min(np.ma.masked_invalid(
                    exp_scores[...,2,:].flatten())),
                np.max(np.ma.masked_invalid(
                    exp_scores[...,2,:].flatten()))])

        if i != 0:
            axs[i].set(yticklabels=[])
        else:
            axs[i].set_ylabel("ELBO and Log Likelihood")

        if kl_fit_finite:
            if i != nb_evals - 1:
                ax2.set(yticklabels=[])
            else:
                ax2.set_ylabel("KL(q|prior)")

    if legend:
        handles,labels = [],[]
        for ax in f.axes:
            for h,l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        plt.legend(handles, labels, loc=legend, framealpha=1,
                facecolor="white", fancybox=True)

    if title:
        plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)

def plot_fit_estim_distance(
        scores,
        sim_params,
        out_file,
        scaled=False,
        sizefont=16,
        usetex=False,
        print_xtick_every=10,
        y_limits=[0., None],
        legend='upper right',
        title=None):
    """
    scores here is a dictionary of estimate dictionary of stats
    mean, var...
    Each array has the shape:(nb_data, nb_fits, nb_epochs, 
    *estim_shape)
    """

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    plt.rcParams.update({'font.size':sizefont,
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    f, ax = plt.subplots(figsize=(8, 5))

    an_estimate = list(scores.keys())[0]

    nb_data = scores[an_estimate]["mean"].shape[0]
    nb_iters = scores[an_estimate]["mean"].shape[-2]

    x = [j for j in range(1, nb_iters+1)]

    for ind, name in enumerate(scores):
        if name in estim_names:
            estim_scores = scores[name]["mean"]
            # print(name, estim_scores.shape)
            # [nb_data, nb_fit_reps, nb_epochs, estim_shape]
            sim_param = sim_params[name].reshape(nb_data,
                    1, 1, -1)

            # eucl dist
            dists = np.linalg.norm(sim_param - estim_scores,
                    axis=-1)
            # ratio
            #dists = np.squeeze(estim_scores/sim_param, -1)

            if scaled:
                dists = 1-(1/(1+dists))
            #print(name, dists.shape)
            # [nb_data, nb_fit_reps, nb_epochs]

            m = dists.mean((0,1))
            s = dists.std((0,1))

            ax.plot(x, m, "-", color=estim_colors[name],
                    label=estim_names[name])

            ax.fill_between(x, m-s, m+s, 
                    color=estim_colors[name],
                    alpha=0.2, interpolate=True)
        
    ax.set_xticks([t for t in range(1, nb_iters+1) if t==1 or\
            t % print_xtick_every==0])
    ax.set_ylim(y_limits)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Euclidean distance")
    ax.grid(zorder=-1)
    ax.grid(zorder=-1, visible=True, which='minor', alpha=0.1)
    ax.minorticks_on()

    if legend:
        handles,labels = [],[]
        for ax in f.axes:
            for h,l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        plt.legend(handles, labels, loc=legend, framealpha=1,
                facecolor="white", fancybox=True)

    if title:
        plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)

def plot_fit_estim_distances(
        exp_scores,
        exp_values,
        x_names,
        sim_param_exps,
        out_file,
        scaled=False,
        sizefont=14,
        y_limits=[0., None],
        usetex=False,
        print_xtick_every=20,
        legend='best',
        title=None):

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    nb_evals = len(exp_scores)

    f, axs = plt.subplots(1, nb_evals,
            figsize=(7*nb_evals, 5))

    if nb_evals == 1: axs = [axs]

    plt.rcParams.update({'font.size':sizefont,
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.07, hspace=0.1)

    an_estimate = list(exp_scores[0].keys())[0]
    nb_data = exp_scores[0][an_estimate]["mean"].shape[0]
    nb_iters = exp_scores[0][an_estimate]["mean"].shape[-2]
    x = [j for j in range(1, nb_iters+1)]

    for i, exp_value in enumerate(exp_values):
        scores = exp_scores[i]
        sim_params = sim_param_exps[exp_value]

        for ind, name in enumerate(scores):
            if name in estim_names:
                estim_scores = scores[name]["mean"]
                sim_param = sim_params[name].reshape(nb_data,
                        1, 1, -1)
                #print(name, estim_scores.shape)

                # eucl dist
                dists = np.linalg.norm(
                        sim_param - estim_scores, axis=-1)
                #print(name, dists.shape)

                if scaled:
                    dists = 1-(1/(1+dists))

                m = dists.mean((0,1))
                s = dists.std((0,1))

                axs[i].plot(x, m, "-", 
                        color=estim_colors[name],
                        label=estim_names[name])

                axs[i].fill_between(x, m-s, m+s, 
                        color=estim_colors[name],
                        alpha=0.2, interpolate=True)

        axs[i].set_title(x_names[i])
        axs[i].set_xticks([t for t in range(1, nb_iters+1) if\
                t==1 or t % print_xtick_every==0])
        axs[i].set_ylim(y_limits)
        axs[i].set_xlabel("Iterations")
        axs[i].grid(zorder=-2)
        axs[i].grid(zorder=-2, visible=True, which='minor',
                alpha=0.1)
        axs[i].minorticks_on()
 
        if i != 0:
            axs[i].set(yticklabels=[])
        else:
            axs[i].set_ylabel("Euclidean distance")

    if legend:
        handles,labels = [],[]
        for ax in f.axes:
            for h,l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        plt.legend(handles, labels, loc=legend, framealpha=1,
                facecolor="white", fancybox=True)

    if title:
        plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)

def violinplot_sampled_estim_statistics(
        sample_scores,
        exp_names,
        x_names,
        sim_param_exps,
        output_path,
        sizefont=14,
        usetex=False):

    unique_names = []
    for exp_scores in sample_scores:
        unique_names.extend(exp_scores[0].keys())
    unique_names = set(unique_names)

    nb_data = len(sample_scores[0])

    # aggregate data
    estim_dict = dict()
    for estim_name in estim_names:
        if estim_name in unique_names:
            if estim_name in ["b", "r", "f"]:
                col_index = pd.MultiIndex.from_product(
                    [x_names,['dists','scaled_dists','corrs']])

                df = pd.DataFrame(columns=col_index)

                for c, exp_scores in \
                        enumerate(sample_scores):

                    # exp_scores is a list with nb_data 
                    # elements. Each element contains a 
                    # dictionary for estimates [b, r, f..]
                    # that have values of shape
                    # [nb_rep, nb_sample,shape of estimate]
                    #print(estim_name)
                    #print(exp_scores[0][estim_name].shape)
                    if estim_name in exp_scores[0]:
                        scores = np.array([exp_scores[d][
                            estim_name] 
                            for d in range(nb_data)])
                        #print(estim_name, scores.shape)
                        #[nb_data, nb_fit_reps, 
                        # nb_samples, estim_shape]

                        sim_param = sim_param_exps[
                            exp_names[c]][estim_name]
                        #print("sim_param {}".format(
                        #    sim_param.shape))
                        #[nb_data, estim shape]

                        # euclidean distance
                        dists = np.linalg.norm(
                            sim_param.reshape(
                                nb_data,1,1,-1) - scores,
                            axis=-1)

                        df[x_names[c],"dists"]=dists.mean(
                                (0,1))

                        # Scaled distance within range 0,1
                        scaled_dists = 1-(1/(1+dists))

                        df[x_names[c], "scaled_dists"] = \
                                scaled_dists.mean((0,1))

                        # correlation
                        corrs, pvals = compute_corr(
                            sim_param, scores) 
                        #print("corrs", corrs.shape)
                        #[nb_data, nb_fit_reps, nb_samples]

                        df[x_names[c],"corrs"]=corrs.mean(
                                (0,1))

            elif estim_name in ["t", "k"]:
                col_index = pd.MultiIndex.from_product(
                        [x_names, ['Ratio']])

                df = pd.DataFrame(columns=col_index)

                for c, exp_scores in \
                        enumerate(sample_scores):
                    if estim_name in exp_scores[0]:
                        scores = np.array([exp_scores[d][
                            estim_name] 
                            for d in range(nb_data)])
                        #print(estim_name, scores.shape)

                        sim_param = sim_param_exps[
                            exp_names[c]][estim_name]

                        ratio = np.squeeze(
                                scores/sim_param.reshape(
                                    nb_data,1,1,-1), -1)

                        df[x_names[c], "Ratio"] = \
                                ratio.mean((0,1))

            estim_dict[estim_name] = df

    y_limits = {
            "dists": [-0.1, None],
            "scaled_dists": [-0.1, 1.1],
            "corrs": [-1.1, 1.1],
            "ratios": [-0.1, None]
            }

    # plotting
    for estim_name in estim_dict:
        df = estim_dict[estim_name]
        for stat in ['dists','scaled_dists','corrs','ratios']:
            if stat in df.columns.get_level_values(1):
                #print(estim_name, stat)
                out_file = output_path+estim_name+"_"+stat
                title=os.path.basename(out_file)

                x_df = df.iloc[:,
                        df.columns.get_level_values(1)==stat]
                x_df.columns = x_df.columns.droplevel(1)
                #print(estim_name, stat, "\n", x_df.describe())
                # TODO Check if x_df is empty
                # https://stackoverflow.com/a/72086939
                x_df = x_df.replace([np.inf, -np.inf],
                        np.nan).dropna(axis=1)

                if not x_df.empty:
                    violinplot_from_dataframe(
                            x_df,
                            out_file,
                            y_limit=y_limits[stat],
                            sizefont=sizefont,
                            usetex=usetex,
                            title=title)

def violinplot_from_dataframe(
        df,
        out_file,
        y_limit=[0, None],
        sizefont=14,
        usetex=False,
        title=None):

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    f, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme()
    plt.rcParams.update({'font.size':sizefont,
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.07, hspace=0.1)

    sns.violinplot(data=df, palette="husl")

    plt.ylim(*y_limit)

    if title:
        plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)

def plot_fit_estim_correlation(
        scores,
        sim_params,
        out_file,
        sizefont=16,
        usetex=False,
        print_xtick_every=10,
        y_limits=[-1., 1.],
        legend='lower right',
        title=None):
        
    """
    scores here is a dictionary of estimate arrays.
    Each array has the shape:(nb_data, nb_fits, nb_epochs, 
    *estim_shape)
    """

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    f, ax = plt.subplots(figsize=(8, 5))

    plt.rcParams.update({'font.size':sizefont,
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    an_estimate = list(scores.keys())[0]
    nb_iters = scores[an_estimate]["mean"].shape[-2]
    nb_data = scores[an_estimate]["mean"].shape[0]
    x = [j for j in range(1, nb_iters+1)]

    for ind, name in enumerate(scores):
        if name in estim_names:
            estim_scores = scores[name]["mean"]
            sim_param = sim_params[name]
            #print(name, estim_scores.shape)
            # [nb_data, nb_fit_reps, nb_epochs, estim_shape]
            #print(name, sim_param.shape)
            # [nb_data, estim_shape]

            # pearson correlation coefficient
            corrs,_ = compute_corr(sim_param.reshape(
                nb_data, -1), estim_scores)
            #print(name, corrs.shape)
            # [nb_data, nb_fit_reps, nb_epochs]

            if np.isfinite(corrs).any():
                m = corrs.mean((0,1)) # average by replicates
                s = corrs.std((0,1))  # std by replicates
                ax.plot(x, m, "-", color=estim_colors[name],
                        label=estim_names[name])

                ax.fill_between(x, m-s, m+s, 
                        color=estim_colors[name],
                        alpha=0.2, interpolate=True)

    ax.set_xticks([t for t in range(1, nb_iters+1) if t==1 or\
            t % print_xtick_every==0])
    ax.set_ylim(y_limits)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Correlation coefficient")
    ax.grid(zorder=-1)
    ax.grid(zorder=-1, visible=True, which='minor', alpha=0.1)
    ax.minorticks_on()

    if legend:
        handles,labels = [],[]
        for ax in f.axes:
            for h,l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        plt.legend(handles, labels, loc=legend, framealpha=1,
                facecolor="white", fancybox=True)

    if title:
        plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)

def plot_fit_estim_correlations(
        exp_scores,
        exp_values,
        x_names,
        sim_param_exps,
        out_file,
        sizefont=14,
        y_limits=[0., None],
        usetex=False,
        print_xtick_every=20,
        legend='best',
        title=None):

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    nb_evals = len(exp_scores)

    f, axs = plt.subplots(1, nb_evals, 
            figsize=(7*nb_evals, 5))

    if nb_evals == 1: axs = [axs]

    plt.rcParams.update({'font.size':sizefont,
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.07, hspace=0.1)

    an_estimate = list(exp_scores[0].keys())[0]
    nb_iters = exp_scores[0][an_estimate]["mean"].shape[-2]
    x = [j for j in range(1, nb_iters+1)]

    for i, exp_value in enumerate(exp_values):
        scores = exp_scores[i]
        sim_params = sim_param_exps[exp_value]

        for ind, name in enumerate(scores):
            if name in estim_names:
                estim_scores = scores[name]["mean"]
                sim_param = sim_params[name]
                #print(name, estim_scores.shape)
                #print(name, sim_param.shape)

                # pearson correlation coefficient
                corrs,_ = compute_corr(sim_param, estim_scores)
                #print(name, corrs.shape)

                if np.isfinite(corrs).any():
                    m = corrs.mean((0,1))
                    s = corrs.std((0,1))

                    axs[i].plot(x, m, "-",
                            color=estim_colors[name],
                            label=estim_names[name])

                    axs[i].fill_between(x, m-s, m+s, 
                            color=estim_colors[name],
                            alpha=0.2, interpolate=True)

        axs[i].set_title(x_names[i])
        axs[i].set_xticks([t for t in range(1, nb_iters+1) if\
                t==1 or t % print_xtick_every==0])
        axs[i].set_ylim(y_limits)
        axs[i].set_xlabel("Iterations")
        axs[i].grid(zorder=-2)
        axs[i].grid(zorder=-2, visible=True, which='minor',
                alpha=0.1)
        axs[i].minorticks_on()
 
        if i != 0:
            axs[i].set(yticklabels=[])
        else:
            axs[i].set_ylabel("Correlation coefficient")

    if legend:
        handles,labels = [],[]
        for ax in f.axes:
            for h,l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        plt.legend(handles, labels, loc=legend, framealpha=1,
                facecolor="white", fancybox=True)

    if title:
        plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)

def plot_fit_estim_statistics(
        estim_scores,
        exp_values,
        x_names,
        sim_param_exps,
        #
        dist_out_file,
        scaled_dist_out_file,
        dist_legend,
        dist_title,
        #
        corr_out_file,
        corr_legend,
        corr_title,
        #
        usetex=False,
        sizefont=14,
        print_xtick_every=100):

    # Distances of estimates with sim params
    plot_fit_estim_distances(
            estim_scores,
            exp_values,
            x_names,
            sim_param_exps,
            dist_out_file,
            scaled=False,
            sizefont=sizefont,
            print_xtick_every=print_xtick_every,
            y_limits=[-0.1, None],
            usetex=usetex,
            legend=dist_legend,
            title=dist_title)

    # Scaled distances of estimates with sim params
    plot_fit_estim_distances(
            estim_scores,
            exp_values,
            x_names,
            sim_param_exps,
            scaled_dist_out_file,
            scaled=True,
            sizefont=sizefont,
            print_xtick_every=print_xtick_every,
            y_limits=[-0.1, 1.1],
            usetex=usetex,
            legend=dist_legend,
            title=dist_title)

    # Correlations of estimates with sim params
    plot_fit_estim_correlations(
            estim_scores,
            exp_values,
            x_names,
            sim_param_exps,
            corr_out_file,
            sizefont=sizefont,
            print_xtick_every=print_xtick_every,
            y_limits=[-1.1, 1.1],
            usetex=usetex,
            legend=corr_legend,
            title=corr_title)

def aggregate_estimate_values(
        rep_results,
        key, #fit_estimates
        report_n_epochs=False,
        ):

    #return a dictionary of arrays
    estimates = defaultdict(dict)

    # List (nb_data) of List (nb_fits) of list (nb_epochs)
    # of dictionaries (estimate names) of
    # dicitonaries (estimate stats)
    #estim_reps = [result[key] for result in rep_results]
    estim_reps = [[rep[key] for rep in d ] \
            for d in rep_results]

    param_names = estim_list
    names = param_names+[
            "a", "x", 
            "a_hamming", "a_euclidean",
            "x_hamming", "x_euclidean"]

    nb_data = len(rep_results)
    nb_fits = len(rep_results[0])

    if report_n_epochs:
        nb_epochs = report_n_epochs
    else:
        nb_epochs = len(estim_reps[0][0])

    #print(list(estim_reps[0][0][0].keys()))

    for name in names:
        if name in estim_reps[0][0][0]:
            #print(name)

            estim_stats = estim_reps[0][0][0][name]

            for stat_name in stat_names:
                if stat_name in estim_stats:
                    estim = estim_stats[stat_name]

                if name in param_names:
                    shape = list(estim.flatten().shape)
                else:
                    shape = list(estim.shape)
                #print(name, shape)

                estimates[name][stat_name] = np.zeros((
                    nb_data,
                    nb_fits,
                    nb_epochs,
                    *shape))
                #print(estimates[name][stat_name].shape)

    for i, d_replicat in enumerate(estim_reps):
        # list of reps
        for j, f_replicat in enumerate(d_replicat):
            #print("f_replicat {}".format(type(f_replicat)))
            for k in range(nb_epochs): # list of epochs
                epoch = f_replicat[k]
                for name in estimates:
                    for stat_name in estimates[name]:
                        if isinstance(epoch[name][stat_name],
                                torch.Tensor):
                            estimation = epoch[name][
                                    stat_name].cpu().detach(
                                            ).numpy()
                        elif isinstance(epoch[name][stat_name],
                                np.ndarray): 
                            estimation = epoch[name][stat_name]
                        else:
                            raise ValueError("{} is not"\
                                    " tensor or array in"\
                                    " {}".format(name, key))

                        if name in param_names:
                            #print(name, estimation.shape)
                            estimation = estimation.flatten()

                        estimates[name][stat_name][i,j,k]\
                                = estimation
                        #print(name, estimation.shape)

    return estimates 

def aggregate_sampled_estimates(
        rep_results,
        key):

    estim_reps = [[rep[key] for rep in d ] \
            for d in rep_results]


    nb_data = len(estim_reps)
    nb_fits = len(estim_reps[0])

    all_estimates = [] 
    for i in range(nb_data): 
        estimates = defaultdict(list)
        for j in range(nb_fits):
            samples = estim_reps[i][j]

            for name in samples:
                sample = samples[name]
                estimates[name].append(sample)

        for name in estimates:
            estimates[name] = np.stack(estimates[name],axis=0)

        all_estimates.append(dict(estimates))

    return all_estimates

def report_sampled_estimates(
        estimates,
        out_file,
        job_name=None,
        real_params=None,
        branch_names=None):

    nb_data = len(estimates)

    pkg_name = __name__.split(".")[0]
    chaine =  "{} {} estimations\n".format(pkg_name, _version)
    chaine += "##########################\n\n"

    if job_name:
        chaine += "Job: {}\n\n".format(job_name)

    chaine += "{}\n".format("#"*70)
    for i in range(nb_data):
        estim_data = estimates[i]

        chaine += "## Data replicate {} {}\n".format(i,"#"*50)
        chaine += "{}\n".format("#"*70)

        chaine += "\n## Log probabilities and KLs\n"
        for prob_name in prob_names:
            prob = estim_data[prob_name]

            # TODO Get the full values to compute other 
            # statistics
            if len(prob.shape) > 1: prob = prob.mean()
            chaine +="{}\t{:.5f}\n".format(
                    prob_names[prob_name], prob.mean())

        for name in estim_names:
            if name in estim_data:
                chaine += "\n## "+estim_names[name] + "\n"

                estimate = estim_data[name]
                # Average by fit replicates
                estimate_avrg_reps = estimate.mean(0)
                param_dim = estimate_avrg_reps.shape[-1]
                #print("param_dim {}".format(param_dim))

                # Statistics by samples (already averaged by
                # replicates)
                estimate_stats = compute_estim_stats(
                        estimate_avrg_reps,
                        confidence=0.95, axis=0)
                
                real_flg = real_params is not None\
                        and name in real_params

                # Names of stat columns
                chaine += "   "
                for stat_name in estimate_stats:
                    chaine += "\t"+stat_name

                if real_flg:
                    chaine += "\tReal"
                chaine += "\n"

                for dim in range(param_dim):
                    if name == "r":
                        the_name = rates_list[dim]
                    elif name == "f":
                        the_name = freqs_list[dim]
                    elif name == "b":
                        if branch_names and\
                                branch_names[i][dim]:
                            the_name = branch_names[i][dim]
                        else:
                            the_name = name + str(dim+1)
                    else:
                        the_name = name

                    chaine += the_name
 
                    for stat_name in estimate_stats:
                        stats = estimate_stats[stat_name]
                        chaine +="\t{:.5f}".format(
                                stats[dim].item())
        
                    if real_flg:
                        real_val = real_params[name][i][dim]
                        chaine +="\t{:.5f}".format(real_val)

                    chaine += "\n"
                chaine += "\n"
            
                # Compute distance and correlation
                if real_flg:
                    sim_param = real_params[name][i]
                    estim_mean = estimate_stats["mean"]

                    distance = np.linalg.norm(
                            sim_param.reshape(1, 1, -1) -\
                            estimate, axis=-1)
                    # distance shape [nb_rep_fit, nb_samples]
                    chaine += "\tEuclidean distance: "\
                            "Mean {:.5f}, STD {:.5f}\n".format(
                                    distance.mean(),
                                    distance.std())

                    corrs, pvals = compute_corr(
                            sim_param.reshape(1, -1),
                            np.expand_dims(estimate, axis=0))
                    # corrs shape [1, nb_rep_fit, nb_samples]
                    chaine += "\tCorrelation and p-value:"\
                        " {:.5f}, {:.5e}\n".format(
                                np.mean(corrs),
                                np.mean(pvals))

        chaine += "\n{}\n".format("#"*70)
    chaine += "END OF REPORT"
    chaine += "\n{}".format("#"*70)

    with open(out_file, "w") as fh:
        fh.write(chaine)

def summarize_sampled_estimates(
        sample_combins,
        combins,
        x_names,
        sim_param_exps,
        out_file,
        logl_data_combins=None
        ):

    probs_dict = dict()
    estim_dict = dict()
    row_index = [c_name for c_name in combins]

    unique_names = []
    for c_name in combins:
        for exp_scores in sample_combins[c_name]:
            unique_names.extend(exp_scores[0].keys())
    unique_names = set(unique_names)
    #print(unique_names)
    # {'r', 'b', 'logprior', 'logl', 'elbo', 'f', 'logq',...}

    nb_data = len(sample_combins[row_index[0]][0])

    for p_name in prob_names:
        if p_name in unique_names:
            col_index = pd.MultiIndex.from_product(
                    [x_names, ['Mean','STD']])
 
            logl_real = False
            if p_name == "logl" and logl_data_combins != None:
                logl_real = True
                col_index = pd.MultiIndex.from_product(
                        [x_names, ['Real','Mean','STD']])

            df = pd.DataFrame("-", index=row_index,
                    columns=col_index)

            for c_name in combins:
                for c, exp_scores in \
                        enumerate(sample_combins[c_name]):
                    if p_name in exp_scores[0]:

                        scores = np.array(
                            [exp_scores[d][p_name]\
                                for d in range(nb_data)])
                        # print(p_name, scores.shape)
                        # [nb_data, nb_fit_reps]

                        df[x_names[c], "Mean"].loc[c_name] =\
                                scores.mean().item()
                        df[x_names[c], "STD"].loc[c_name] =\
                                np.ma.masked_invalid(
                                    scores).std().item()

                        if logl_real and\
                            logl_data_combins[c_name] != None:
                            df[x_names[c], "Real"].loc[c_name]\
                                = logl_data_combins[c_name][
                                        c].mean()

            probs_dict[prob_names[p_name]] = df

    write_dict_dfs(probs_dict, out_file+"_probs.txt")

    for estim_name in estim_names:
        if estim_name in unique_names:
            if estim_name in ["b", "r", "f"]:
                col_index = pd.MultiIndex.from_product(
                    [x_names, ['dists','scaled_dists',
                        'corrs','pvals'], ['Mean', 'STD']])

                df = pd.DataFrame("-", index=row_index,
                        columns=col_index)

                for c_name in combins:
                    exp_names = combins[c_name]
                    #print(exp_names)

                    for c, exp_scores in \
                            enumerate(sample_combins[c_name]):

                        # exp_scores is a list with nb_data 
                        # elements. Each element contains a 
                        # dictionary for estimates [b, r, f..]
                        # that have values of shape
                        # [nb_rep, nb_sample,shape of estimate]
                        #print(estim_name)
                        #print(exp_scores[0][estim_name].shape)
                        if estim_name in exp_scores[0]:
                            scores = np.array([exp_scores[d][
                                estim_name] 
                                for d in range(nb_data)])
                            #print(estim_name, scores.shape)
                            #[nb_data, nb_fit_reps, 
                            # nb_samples, estim_shape]

                            sim_param = sim_param_exps[
                                exp_names[c]][estim_name]
                            #print("sim_param {}".format(
                            #    sim_param.shape))
                            #[nb_data, estim shape]

                            # euclidean distance
                            dists = np.linalg.norm(
                                sim_param.reshape(
                                    nb_data,1,1,-1) - scores,
                                axis=-1)

                            df[x_names[c],"dists", "Mean"].loc[
                                    c_name] = dists.mean()
                            df[x_names[c],"dists", "STD"].loc[
                                    c_name] = dists.std()

                            # Scaled distance within range 0,1
                            scaled_dists = 1-(1/(1+dists))

                            df[x_names[c],"scaled_dists",
                                    "Mean"].loc[c_name]=\
                                            scaled_dists.mean()
                            df[x_names[c],"scaled_dists",
                                    "STD"].loc[c_name]=\
                                            scaled_dists.std()

                            # correlation
                            corrs, pvals = compute_corr(
                                sim_param, scores) 
                            #print("corrs", corrs.shape)
                            #[nb_data, nb_fit_reps, nb_samples]

                            df[x_names[c],"corrs", "Mean"].loc[
                                    c_name] = corrs.mean()
                            df[x_names[c],"corrs", "STD"].loc[
                                    c_name] = corrs.std()
                            df[x_names[c],"pvals","Mean"].loc[
                                    c_name] = "{:.5e}".format(
                                            pvals.mean())
                            df[x_names[c],"pvals","STD"].loc[
                                    c_name] = "{:.5e}".format(
                                            pvals.std())

            elif estim_name in ["t", "k"]:
                col_index = pd.MultiIndex.from_product(
                        [x_names, ['Mean','STD','Real']])

                df = pd.DataFrame("", index=row_index,
                        columns=col_index)

                for c_name in combins:
                    exp_names = combins[c_name]
                    for c, exp_scores in \
                            enumerate(sample_combins[c_name]):
                        if estim_name in exp_scores[0]:
                            scores = np.array([exp_scores[d][
                                estim_name] 
                                for d in range(nb_data)])
                            #print(estim_name, scores.shape)

                            sim_param = sim_param_exps[
                                exp_names[c]][estim_name]

                            df[x_names[c],"Mean"].loc[
                                    c_name] = scores.mean()
                            df[x_names[c],"STD"].loc[
                                    c_name] = scores.std()
                            df[x_names[c],"Real"].loc[
                                    c_name] = sim_param.mean()

            estim_dict[estim_names[estim_name]] = df

    write_dict_dfs(estim_dict, out_file+"_estim.txt")

def write_dict_dfs(dict_df, filename):
    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None):
        pd.options.display.float_format = '{:.5f}'.format
        with open(filename, "w") as fh:
            for code in dict_df:
                fh.write("## {}".format(code))
                fh.write("\n\n")
                #fh.write(dict_df[code].to_csv(
                #    float_format='%.3f'))
                #fh.write("\n")
                #fh.write("\n")
                df = dict_df[code]
                #if "pval" in df.columns.get_level_values(1):
                #    df = df.drop("pval", axis=1, level=1)
                fh.write(df.style.to_latex())
                fh.write("\n")
                fh.write(dict_df[code].to_string(
                    justify="left"))
                fh.write("\n")
                fh.write("\n")
                fh.write("#" * 63)
                fh.write("\n")
                fh.write("\n")
