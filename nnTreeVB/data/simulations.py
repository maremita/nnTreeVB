from .utils import set_postorder_ranks

from pyvolve import read_tree, Model, Partition, Evolver
from ete3 import Tree

import torch

__author__ = "amine"


__all__ = [
        "evolve_seqs_full_homogeneity",
        "simulate_tree"
        ]


def simulate_tree(
        nb_taxa, 
        branch_dists,
        unroot=True,
        seed=None):

    taxa_names = ["T"+str(i) for i in list(range(0, nb_taxa))]
    post_branches = []

    t = Tree()
    t.populate(nb_taxa,
            names_library=taxa_names,
            random_branches=False)

    if unroot: t.unroot()

    t.sort_descendants()

    if len(branch_dists) == 1:
        # use the same distribution to sample internal edges
        branch_dists.append(branch_dists[0])

    # Populate branches 
    with torch.no_grad():
        for node in t.traverse("postorder"):
            if node.is_leaf():
                node.dist = branch_dists[0].sample().item()
                post_branches.append(node.dist)
            elif not node.is_root():
                node.dist = branch_dists[1].sample().item()
                post_branches.append(node.dist)
            else:
                node.dist = 0.

    # Add postorder ranking of nodes:
    t, taxa, nodes = set_postorder_ranks(t)

    return t, post_branches, taxa, nodes

def evolve_seqs_full_homogeneity(
        nwk_tree,
        nb_sites=10, 
        fasta_file=False,
        subst_rates=None,
        state_freqs=None,
        return_anc=False,
        seed=None,
        verbose=False):

    """
    Order of GTR rates:
    AG AC AT GC GT CT

    Order of relative frequencies:
    A C G T
    """

    gtr_r = ["AG", "AC", "AT", "GC", "GT", "CT"]
 
    # Evolve sequences
    if verbose: print("Evolving new sequences with the"\
            " amazing Pyvolve for {}".format(fasta_file))
    pytree = read_tree(tree=nwk_tree)

    parameters = None
    if subst_rates is not None or state_freqs is not None:
        parameters = dict()

        if subst_rates is not None:
            rs = subst_rates
            if isinstance(subst_rates, float):
                rs = {g:subst_rates for g in gtr_r}

            elif isinstance(subst_rates, list):
                assert len(subst_rates) == 6
                rs = {g:float(r) for g, r in zip(gtr_r,
                    subst_rates)}

            parameters.update(mu=rs)

        if state_freqs is not None:
            # A C G T
            fs = state_freqs
            if isinstance(state_freqs, float):
                fs = [state_freqs]*4

            elif isinstance(state_freqs, list):
                assert len(state_freqs) == 4

            parameters.update(state_freqs=fs)

    m = Model("nucleotide", parameters=parameters) 

    p = Partition(size=nb_sites, models=m)
    e = Evolver(partitions=p, tree=pytree)

    e(seqfile=fasta_file, infofile=False, ratefile=False,
            write_anc=True, seed=seed)

    seqdict = e.get_sequences(anc=return_anc)

    return seqdict
