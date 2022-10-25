from collections import defaultdict
import numpy as np

from nnTreeVB.utils import compute_corr
from nnTreeVB.utils import compute_estim_stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch

__author__ = "amine remita"


estim_names = ["b", "t", "b1", "r", "f", "k"]
stat_names = ["mean", "cimin", "cimax", "var", "min", "max"]

def plot_grads_weights_epochs(
        data,
        module_name,
        out_file=False,
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
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    w_means = []
    w_vars = []
    g_means = []
    g_vars = []
 
    for e in data["grad_stats"]:
        g_means.append(e[module_name]["mean"])
        g_vars.append(e[module_name]["var"])

    for e in data["weight_stats"]:
        w_means.append(e[module_name]["mean"])
        w_vars.append(e[module_name]["var"])

    g_means = np.array(g_means)
    g_stds = np.sqrt(np.array(g_vars))
    w_means = np.array(w_means)
    w_stds = np.sqrt(np.array(w_vars))

    f, axs = plt.subplots(1, 2, figsize=fig_size)

    nb_iters = len(g_means)
    x = [j for j in range(1, nb_iters+1)]
    
    axs[0].plot(x, g_means, label='gradients', color="red")
    axs[0].fill_between(x,
        g_means-g_stds,
        g_means+g_stds,
        alpha=0.2, color="red")

    axs[0].grid(zorder=-1)
    axs[0].legend()

    axs[1].plot(x, w_means, label='weights')
    axs[1].fill_between(x,
            w_means-w_stds,
            w_means+w_stds,
            alpha=0.2)

    axs[1].grid(zorder=-1)
    axs[1].legend()

    plt.suptitle(module_name)
 
    if out_file:
        plt.savefig(fig_file, bbox_inches="tight", 
                format=fig_format, dpi=fig_dpi)
    else:
        plt.show()

    plt.close(f)


def plt_elbo_ll_kl_rep_figure(
        scores,
        out_file,
        sizefont=16,
        usetex=False,
        print_xtick_every=10,
        legend='best',
        title=None,
        plot_validation=False):

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    plt.rcParams.update({'font.size':sizefont, 
        'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    f, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    nb_iters = scores.shape[2] 
    x = [j for j in range(1, nb_iters+1)]

    ax.set_rasterization_zorder(0)
 
    elbo_color = "#E9002D"  #sharop red
    ll_color =   "#226E9C"  # darker blue
    kl_color =   "#7C1D69"  # pink

    elbo_color_v = "tomato"
    ll_color_v =   "#009ADE" # light blue
    kl_color_v =   "#AF58BA"  # light pink
    
    # plot means
    ax.plot(x, scores[:,0,:].mean(0), "-", color=elbo_color, 
            label="ELBO", zorder=6) # ELBO train
    ax.plot(x, scores[:,1,:].mean(0), "-", color=ll_color,
            label="LogL", zorder=4) # LL train
    ax2.plot(x, scores[:,2,:].mean(0), "-", color=kl_color,
            label="KL_qp", zorder=4) # KL train

    # plot stds 
    ax.fill_between(x,
            scores[:,0,:].mean(0)-scores[:,0,:].std(0), 
            scores[:,0,:].mean(0)+scores[:,0,:].std(0), 
            color=elbo_color,
            alpha=0.2, zorder=5, interpolate=True)

    ax.fill_between(x,
            scores[:,1,:].mean(0)-scores[:,1,:].std(0), 
            scores[:,1,:].mean(0)+scores[:,1,:].std(0),
            color=ll_color,
            alpha=0.2, zorder=3, interpolate=True)

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
        ax.plot(x, scores[:,4,:].mean(0), "-.",
                color=ll_color_v,
                label="LogL_val", zorder=0) # LL val
        ax2.plot(x, scores[:,5,:].mean(0), "-.",
                color=kl_color_v,
                label="KL_qp_val", zorder=2) # KL val
        
        ax.fill_between(x,
                scores[:,3,:].mean(0)-scores[:,3,:].std(0), 
                scores[:,3,:].mean(0)+scores[:,3,:].std(0), 
                color=elbo_color_v,
                alpha=0.1, zorder=1, interpolate=True)

        ax.fill_between(x,
                scores[:,4,:].mean(0)-scores[:,4,:].std(0), 
                scores[:,4,:].mean(0)+scores[:,4,:].std(0), 
                color=ll_color_v,
                alpha=0.1, zorder=2, interpolate=True)

        ax2.fill_between(x,
                scores[:,5,:].mean(0)-scores[:,5,:].std(0), 
                scores[:,5,:].mean(0)+scores[:,5,:].std(0), 
                color= kl_color_v,
                alpha=0.1, zorder=1, interpolate=True)

    #ax.set_zorder(ax2.get_zorder()+1)
    ax.set_frame_on(False)

    ax.set_ylim([None, 0])
    ax.set_xticks([t for t in range(1, nb_iters+1) if t==1 or\
            t % print_xtick_every==0])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("ELBO and Log Likelihood")
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
        #plt.legend(handles, labels, bbox_to_anchor=(1.105, 1), 
        #        loc='upper left', borderaxespad=0.)
        plt.legend(handles, labels, loc=legend, framealpha=1,
                facecolor="white", fancybox=True)

    if title:
        plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)

def plot_fit_estim_dist(
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

    params = {
            "b":"Branch lengths",
            "t":"Tree length",
            "r":"Substitution rates", 
            "f":"Relative frequencies",
            "k":"Kappa"}

    colors = { 
            "b":"#226E9C",
            "t":"#226E9C",
            "r":"#D12959", 
            "f":"#40AD5A",
            "k":"#FFAA00"}

    for ind, name in enumerate(scores):
        if name in params:
            estim_scores = scores[name]["mean"]
            sim_param = sim_params[name].reshape(1, 1, -1)

            # eucl dist
            dists = np.linalg.norm(
                    sim_param - estim_scores, axis=-1)
            #print(name, dists.shape)

            m = dists.mean(0)
            s = dists.std(0)

            ax.plot(x, m, "-", color=colors[name],
                    label=params[name])

            ax.fill_between(x, m-s, m+s, 
                    color=colors[name],
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
        #plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), 
        #        loc='upper left', borderaxespad=0.)
        plt.legend(handles, labels, loc=legend, framealpha=1,
                facecolor="white", fancybox=True)

    if title:
        plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)


def plot_fit_estim_corr(
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
    Each array has the shape:(nb_reps, nb_epochs, *estim_shape)
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

    params = {
            "b":"Branch lengths",
            "t":"Tree length",
            "r":"Substitution rates", 
            "f":"Relative frequencies"}

    colors = { 
            "b":"#226E9C",
            "r":"#D12959", 
            "f":"#40AD5A",
            "k":"#FFAA00"}

    # Don't compute correlation if vector has the same values
    skip = []
    for name in sim_params:
        if np.all(sim_params[name]==sim_params[name][0]):
            skip.append(name)

    for ind, name in enumerate(scores):
        if name in params and name not in skip:
            estim_scores = scores[name]["mean"]
            sim_param = sim_params[name]
            #print(name, estim_scores.shape)

            # pearson correlation coefficient
            corrs = compute_corr(sim_param, estim_scores)
            #print(name, corrs.shape)

            m = corrs.mean(0)
            s = corrs.std(0)
            ax.plot(x, m, "-", color=colors[name],
                    label=params[name])

            ax.fill_between(x, m-s, m+s, 
                    color=colors[name],
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
        #plt.legend(handles, labels, bbox_to_anchor=(1.01, 1), 
        #        loc='upper left', borderaxespad=0.)
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

    param_names = estim_names
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
        out_file
        ):

    param_names = {
            "b":"Branch lengths",
            "t":"Tree length",
            "r":"Substitution rates", 
            "f":"Relative frequencies",
            "k":"Kappa"}

    rates = ["AG", "AC", "AT", "GC", "GT", "CT"]
    freqs = ["A", "G", "C", "T"]

    prob_names = {"elbo":"ELBO", 
            "logl":"LogL",
            "logprior":"LogPrior",
            "logq":"LogQ",
            "kl_qprior":"KL_QPrior"}

    pkg_name = __name__.split(".")[0]
    chaine =  "{} estimations\n".format(pkg_name)
    chaine += "####################\n\n"

    chaine += "Log probabilities and KLs\n"
    for prob_name in prob_names:
        prob = estimates[prob_name]

        # TODO Get the full values to compute other statistics
        if len(prob.shape) > 1: prob = prob.mean()
        chaine +="{}\t\t{:.4f}\n".format(prob_names[prob_name],
                prob.mean())

    chaine += "\n"

    for name in param_names:
        #var_flag = False
        if name in estimates:
            chaine += param_names[name] + "\n"
 
            # Average by replicates
            estimate = estimates[name].mean(0)
            param_dim = estimate.shape[-1]
            print("param_dim {}".format(param_dim))

            #means = estimate.mean(0)

            # Statistics by samples (already averaged by
            # replicates)
            estimate_stats = compute_estim_stats(
                    estimate, confidence=0.95, axis=0)

            # Names of stat columns
            chaine += "   "
            for stat_name in estimate_stats:
                chaine += "\t"+stat_name
            chaine += "\n"

            for dim in range(param_dim):
                if name == "r":
                    the_name = rates[dim]
                elif name == "f":
                    the_name = freqs[dim]
                else:
                    # TODO Get branch names
                    the_name = name + str(dim+1)

                chaine += the_name
                
                for stat_name in estimate_stats:
                    stats = estimate_stats[stat_name]
                    chaine +="\t{:.4f}".format(
                            stats[dim].item())
                chaine += "\n"
            chaine += "\n"

    with open(out_file, "w") as fh:
        fh.write(chaine)
