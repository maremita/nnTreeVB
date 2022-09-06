import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

__author__ = "amine remita"


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
