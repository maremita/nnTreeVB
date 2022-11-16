from nnTreeVB.utils import compute_corr
from nnTreeVB.utils import compute_estim_stats

from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist

import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
        "t":"#226E9C",
        "r":"#D12959", 
        "f":"#40AD5A",
        "k":"#FFAA00"}

prob_names = {"elbo":"ELBO", 
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

    kl_fit_finite = np.isfinite(scores[:,2,:]).all()
    kl_val_finite = False
    if plot_validation:
        kl_val_finite = np.isfinite(scores[:,5,:]).all()

    plt.rcParams.update({'font.size':sizefont, 
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    f, ax = plt.subplots(figsize=(8, 5))
    
    if kl_fit_finite or kl_val_finite:
        ax2 = ax.twinx()

    nb_iters = scores.shape[2] 
    x = [j for j in range(1, nb_iters+1)]

    ax.set_rasterization_zorder(0)

    ax.plot(x, scores[:,0,:].mean(0), "-", color=elbo_color, 
            label="ELBO", zorder=6) # ELBO train
    
    ax.fill_between(x,
            scores[:,0,:].mean(0)-scores[:,0,:].std(0), 
            scores[:,0,:].mean(0)+scores[:,0,:].std(0), 
            color=elbo_color,
            alpha=0.2, zorder=5, interpolate=True)

    ax.plot(x, scores[:,1,:].mean(0), "-", color=ll_color,
            label="LogL", zorder=4) # LL train

    ax.fill_between(x,
            scores[:,1,:].mean(0)-scores[:,1,:].std(0), 
            scores[:,1,:].mean(0)+scores[:,1,:].std(0),
            color=ll_color,
            alpha=0.2, zorder=3, interpolate=True)
    
    if kl_fit_finite:
        ax2.plot(x, scores[:,2,:].mean(0), "-", color=kl_color,
            label="KL_qp", zorder=4) # KL train

        ax2.fill_between(x,
                scores[:,2,:].mean(0)-scores[:,2,:].std(0), 
                scores[:,2,:].mean(0)+scores[:,2,:].std(0), 
                color=kl_color,
                alpha=0.2, zorder=-6, interpolate=True)

    # plot validation
    if plot_validation:
        ax.plot(x, scores[:,3,:].mean(0), "-.",
                color=elbo_color_v,
                label="ELBO_val", zorder=2) # ELBO val

        ax.fill_between(x,
                scores[:,3,:].mean(0)-scores[:,3,:].std(0), 
                scores[:,3,:].mean(0)+scores[:,3,:].std(0), 
                color=elbo_color_v,
                alpha=0.1, zorder=1, interpolate=True)

        ax.plot(x, scores[:,4,:].mean(0), "-.",
                color=ll_color_v,
                label="LogL_val", zorder=0) # LL val
 
        ax.fill_between(x,
                scores[:,4,:].mean(0)-scores[:,4,:].std(0), 
                scores[:,4,:].mean(0)+scores[:,4,:].std(0), 
                color=ll_color_v,
                alpha=0.1, zorder=2, interpolate=True)

        if kl_val_finite:
            ax2.plot(x, scores[:,5,:].mean(0), "-.",
                    color=kl_color_v,
                    label="KL_qp_val", zorder=2) # KL val
        
            ax2.fill_between(x,
                    scores[:,5,:].mean(0)-scores[:,5,:].std(0), 
                    scores[:,5,:].mean(0)+scores[:,5,:].std(0), 
                    color= kl_color_v,
                    alpha=0.1, zorder=1, interpolate=True)

    if line:
        ax.axhline(y=line, color=line_color, linestyle='-')

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

        kl_fit_finite = np.isfinite(scores[:,2,:]).all()
        kl_val_finite = False
        if plot_validation:
            kl_val_finite = np.isfinite(scores[:,5,:]).all()

        if kl_fit_finite or kl_val_finite:
            ax2 = axs[i].twinx()

        # ELBO train
        axs[i].plot(x, scores[:,0,:].mean(0), "-",
                color=elbo_color,  label="ELBO", zorder=6) 

        axs[i].fill_between(x,
                scores[:,0,:].mean(0)-scores[:,0,:].std(0), 
                scores[:,0,:].mean(0)+scores[:,0,:].std(0), 
                color=elbo_color,
                alpha=0.2, zorder=5, interpolate=True)

        # LL train
        axs[i].plot(x, scores[:,1,:].mean(0), "-",
                color=ll_color, label="LogL", zorder=4) 

        axs[i].fill_between(x,
                scores[:,1,:].mean(0)-scores[:,1,:].std(0), 
                scores[:,1,:].mean(0)+scores[:,1,:].std(0),
                color=ll_color,
                alpha=0.2, zorder=3, interpolate=True)

        if kl_fit_finite:
            # KL train
            ax2.plot(x, scores[:,2,:].mean(0), "-",
                    color=kl_color, label="KL_qp", zorder=4)

            ax2.fill_between(x,
                    scores[:,2,:].mean(0)-scores[:,2,:].std(0),
                    scores[:,2,:].mean(0)+scores[:,2,:].std(0),
                    color=kl_color,
                    alpha=0.2, zorder=-6, interpolate=True)

        # plot validation
        if plot_validation:
            axs[i].plot(x, scores[:,3,:].mean(0), "-.",
                    color=elbo_color_v,
                    label="ELBO_val", zorder=2) # ELBO val

            axs[i].fill_between(x,
                    scores[:,3,:].mean(0)-scores[:,3,:].std(0),
                    scores[:,3,:].mean(0)+scores[:,3,:].std(0),
                    color=elbo_color_v,
                    alpha=0.1, zorder=1, interpolate=True)

            axs[i].plot(x, scores[:,4,:].mean(0), "-.",
                    color=ll_color_v,
                    label="LogL_val", zorder=0) # LL val

            axs[i].fill_between(x,
                    scores[:,4,:].mean(0)-scores[:,4,:].std(0),
                    scores[:,4,:].mean(0)+scores[:,4,:].std(0),
                    color=ll_color_v,
                    alpha=0.1, zorder=2, interpolate=True)

            if kl_fit_finite:
                ax2.plot(x, scores[:,5,:].mean(0), "-.", 
                        color=kl_color_v,
                        label="KL_qp_val", zorder=2) # KL val
                
                ax2.fill_between(x,
                        scores[:,5,:].mean(0)\
                                -scores[:,5,:].std(0),
                        scores[:,5,:].mean(0)\
                                +scores[:,5,:].std(0),
                        color= kl_color_v,
                        alpha=0.1, zorder=1, interpolate=True)

        if lines:
            axs[i].axhline(y=lines[i], color=line_color,
                    linestyle='-')

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
        sizefont=16,
        usetex=False,
        print_xtick_every=10,
        y_limits=[0., None],
        legend='upper right',
        title=None):
    """
    scores here is a dictionary of estimate arrays.
    Each array has the shape:(nb_reps, nb_epochs, *estim_shape)
    """

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    plt.rcParams.update({'font.size':sizefont,
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    f, ax = plt.subplots(figsize=(8, 5))

    nb_iters = scores["b"]["mean"].shape[1]
    x = [j for j in range(1, nb_iters+1)]

    for ind, name in enumerate(scores):
        if name in estim_names:
            estim_scores = scores[name]["mean"]
            sim_param = sim_params[name].reshape(1, 1, -1)

            # eucl dist
            dists = np.linalg.norm(
                    sim_param - estim_scores, axis=-1)
            #print(name, dists.shape)

            m = dists.mean(0)
            s = dists.std(0)

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

    nb_iters = exp_scores[0]["b"]["mean"].shape[1]
    x = [j for j in range(1, nb_iters+1)]

    for i, exp_value in enumerate(exp_values):
        scores = exp_scores[i]
        sim_params = sim_param_exps[exp_value]

        for ind, name in enumerate(scores):
            if name in estim_names:
                estim_scores = scores[name]["mean"]
                sim_param = sim_params[name].reshape(1,1,-1)
                #print(name, estim_scores.shape)

                # eucl dist
                dists = np.linalg.norm(
                        sim_param - estim_scores, axis=-1)
                #print(name, dists.shape)

                m = dists.mean(0)
                s = dists.std(0)

                axs[i].plot(x, m, "-", 
                        color=estim_colors[name],
                        label=estim_names[name])

                axs[i].fill_between(x, m-s, m+s, 
                        color=estim_colors[name],
                        alpha=0.2, interpolate=True)
    
        #axs[i].set_title(x_names[i].split("-")[1])
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
    Each array has the shape:(nb_reps,nb_epochs,*estim_shape)
    """

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    f, ax = plt.subplots(figsize=(8, 5))

    plt.rcParams.update({'font.size':sizefont,
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    nb_iters = scores["b"]["mean"].shape[1]
    x = [j for j in range(1, nb_iters+1)]

    # Don't compute correlation if vector has the same values
    skip = []
    for name in sim_params:
        if np.all(sim_params[name]==sim_params[name][0]):
            skip.append(name)

    for ind, name in enumerate(scores):
        if name in estim_names and name not in skip:
            estim_scores = scores[name]["mean"]
            sim_param = sim_params[name]
            #print(name, estim_scores.shape)

            # pearson correlation coefficient
            corrs = compute_corr(sim_param, estim_scores)
            #print(name, corrs.shape)

            m = corrs.mean(0) # average by replicate
            s = corrs.std(0)  # std by replicate
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

    nb_iters = exp_scores[0]["b"]["mean"].shape[1]
    x = [j for j in range(1, nb_iters+1)]

    for i, exp_value in enumerate(exp_values):
        scores = exp_scores[i]
        sim_params = sim_param_exps[exp_value]

        # Don't compute correlation if vector has the
        # same values
        skip = []
        for name in sim_params:
            if np.all(sim_params[name]==sim_params[name][0]):
                skip.append(name)

        for ind, name in enumerate(scores):
            if name in estim_names and name not in skip:
                estim_scores = scores[name]["mean"]
                sim_param = sim_params[name]
                #print(name, estim_scores.shape)

                # pearson correlation coefficient
                corrs = compute_corr(sim_param, estim_scores)
                #print(name, corrs.shape)

                m = corrs.mean(0)
                s = corrs.std(0)

                axs[i].plot(x, m, "-",
                        color=estim_colors[name],
                        label=estim_names[name])

                axs[i].fill_between(x, m-s, m+s, 
                        color=estim_colors[name],
                        alpha=0.2, interpolate=True)

        #axs[i].set_title(x_names[i].split("-")[1])
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

def aggregate_estimate_values(
        rep_results,
        key, #val_hist_estim
        report_n_epochs=False,
        ):

    #return a dictionary of arrays
    estimates = defaultdict(dict)

    # List (nb_reps) of list (nb_epochs) of dictionaries
    # (estimate names) of dicitonaries (estimate stats)
    estim_reps = [result[key] for result in rep_results]

    param_names = estim_list
    names = param_names+[
            "a", "x", 
            "a_hamming", "a_euclidean",
            "x_hamming", "x_euclidean"]

    nb_reps = len(rep_results)

    if report_n_epochs:
        nb_epochs = report_n_epochs
    else:
        nb_epochs = len(estim_reps[0])

    #print(list(estim_reps[0][0].keys()))

    for name in names:
        if name in estim_reps[0][0]:
            #print(name)

            estim_stats = estim_reps[0][0][name]

            for stat_name in stat_names:
                if stat_name in estim_stats:
                    estim = estim_stats[stat_name]

                if name in param_names:
                    shape = list(estim.flatten().shape)
                else:
                    shape = list(estim.shape)
                #print(name, shape)

                estimates[name][stat_name] = np.zeros((
                    nb_reps,
                    nb_epochs,
                    *shape))
                #print(estimates[name][stat_name].shape)

    for i, replicat in enumerate(estim_reps): # list of reps
        #print("replicat {}".format(type(replicat)))
        for j in range(nb_epochs): # list of epochs
            epoch = replicat[j]
            #print("epoch {}".format(type(epoch)))
            for name in estimates:
                for stat_name in estimates[name]:
                    if isinstance(epoch[name][stat_name],
                            torch.Tensor):
                        estimation =epoch[name][stat_name].cpu(
                                ).detach().numpy()
                    elif isinstance(epoch[name][stat_name],
                            np.ndarray): 
                        estimation = epoch[name][stat_name]
                    else:
                        raise ValueError("{} is not tensor"\
                                " or array in {}".format(
                                    name, key))

                    if name in param_names:
                        #print(name, estimation.shape)
                        estimation = estimation.flatten()

                    estimates[name][stat_name][i,j]=estimation
                    #print(name, estimation.shape)

    return estimates 

def aggregate_sampled_estimates(
        rep_results,
        key):

    estim_reps = [result[key] for result in rep_results]

    estimates = defaultdict(list)
    nb_reps = len(estim_reps)

    for r in range(nb_reps):
        samples = estim_reps[r]

        for name in samples:
            sample = samples[name]
            estimates[name].append(sample)

    for name in estimates:
        estimates[name] = np.stack(estimates[name], axis=0)

    return dict(estimates)

def report_sampled_estimates(
        estimates,
        out_file,
        real_params=None,
        branch_names=None):

    pkg_name = __name__.split(".")[0]
    chaine =  "{} estimations\n".format(pkg_name)
    chaine += "####################\n\n"

    chaine += "## Log probabilities and KLs\n"
    for prob_name in prob_names:
        prob = estimates[prob_name]

        # TODO Get the full values to compute other statistics
        if len(prob.shape) > 1: prob = prob.mean()
        chaine +="{}\t\t{:.4f}\n".format(prob_names[prob_name],
                prob.mean())

    chaine += "\n"

    for name in estim_names:
        if name in estimates:
            chaine += "## "+estim_names[name] + "\n"

            # Average by replicates
            estimate = estimates[name].mean(0)
            param_dim = estimate.shape[-1]
            #print("param_dim {}".format(param_dim))

            #means = estimate.mean(0)

            # Statistics by samples (already averaged by
            # replicates)
            estimate_stats = compute_estim_stats(
                    estimate, confidence=0.95, axis=0)

            # Names of stat columns
            chaine += "   "
            for stat_name in estimate_stats:
                chaine += "\t"+stat_name

            if real_params and name in real_params:
                chaine += "\tReal"
            chaine += "\n"

            for dim in range(param_dim):
                if name == "r":
                    the_name = rates_list[dim]
                elif name == "f":
                    the_name = freqs_list[dim]
                else:
                    if branch_names and branch_names[dim]:
                        the_name = branch_names[dim]
                    else:
                        the_name = name + str(dim+1)

                chaine += the_name
                
                for stat_name in estimate_stats:
                    stats = estimate_stats[stat_name]
                    chaine +="\t{:.4f}".format(
                            stats[dim].item())
    
                if real_params and name in real_params:
                    chaine +="\t{:.4f}".format(
                            real_params[name][dim].item())

                chaine += "\n"
            chaine += "\n"

    with open(out_file, "w") as fh:
        fh.write(chaine)

def summarize_sampled_estimates(
        sample_combins,
        combins,
        x_names,
        sim_param_exps,
        out_file,
        logl_data_combins = None
        ):

    probs_dict = dict()
    estim_dict = dict()
    row_index = [c_name for c_name in combins]

    unique_names = []
    for c_name in combins:
        for exp_scores in sample_combins[c_name]:
            unique_names.extend(exp_scores.keys())
    unique_names = set(unique_names)
    #print(unique_names)
    # {'r', 'b', 'logprior', 'logl', 'elbo', 'f', 'logq',...}

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
                for i, exp_scores in \
                        enumerate(sample_combins[c_name]):
                    if p_name in exp_scores:
                        scores = exp_scores[p_name]

                        df[x_names[i], "Mean"].loc[c_name] =\
                                scores.mean().item()
                        df[x_names[i], "STD"].loc[c_name] =\
                                np.ma.masked_invalid(
                                        scores).std().item()

                        if logl_real:
                            df[x_names[i], "Real"].loc[c_name]\
                                    = logl_data_combins[
                                            c_name][i].item()

            probs_dict[prob_names[p_name]] = df

    write_dict_dfs(probs_dict, out_file+"_probs.txt")

    for estim_name in estim_names:
        if estim_name in unique_names:
            if estim_name in ["b", "r", "f"]:
                col_index = pd.MultiIndex.from_product(
                        [x_names, ['Dist', 'Corr', 'Pval']])

                df = pd.DataFrame("-", index=row_index,
                        columns=col_index)

                for c_name in combins:
                    exp_names = combins[c_name]
     
                    for i, exp_scores in \
                            enumerate(sample_combins[c_name]):

                        if estim_name in exp_scores:
                            scores=exp_scores[estim_name].mean(
                                    (0,1))
                            sim_param = sim_param_exps[
                                    exp_names[i]][estim_name]

                            # eucl distance
                            dist = np.linalg.norm(
                                    sim_param - scores, 
                                    axis=-1).mean()
     
                            # correlation
                            corr = [np.nan, np.nan]
                            if len(np.unique(sim_param)) > 1:
                                corr = pearsonr(sim_param, 
                                        scores)

                            df[x_names[i],"Dist"].loc[c_name]=\
                                    dist
                            df[x_names[i],"Corr"].loc[c_name]=\
                                    corr[0]
                            df[x_names[i],"Pval"].loc[c_name]=\
                                    corr[1]

            elif estim_name in ["t", "k"]:
                col_index = pd.MultiIndex.from_product(
                        [x_names, ['Value']])

                df = pd.DataFrame("", index=row_index,
                        columns=col_index)

                for c_name in combins:
                    #exp_names = combins[c_name]
                    for i, exp_scores in \
                            enumerate(sample_combins[c_name]):
                        if estim_name in exp_scores:
                            scores=exp_scores[estim_name].mean(
                                    (0,1))
                            df[x_names[i],"Value"].loc[c_name]\
                                    =scores.mean()

            estim_dict[estim_names[estim_name]] = df

    write_dict_dfs(estim_dict, out_file+"_estim.txt")

def write_dict_dfs(dict_df, filename):
    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None):
        pd.options.display.float_format = '{:.3f}'.format
        with open(filename, "w") as fh:
            for code in dict_df:
                fh.write(code)
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
                fh.write(dict_df[code].to_string())
                fh.write("\n")
                fh.write("\n")
                fh.write("#" * 63)
                fh.write("\n")
                fh.write("\n")
