from .utils import set_postorder_ranks

import random 

from pyvolve import read_tree, Model, Partition, Evolver
from ete3 import Tree

import numpy as np
import torch

__author__ = "amine"


__all__ = [
        "evolve_seqs_full_homogeneity",
        "simulate_tree"
        ]


def simulate_tree(
        nb_taxa, 
        branch_lens,
        unroot=True):

    taxa_names = ["T"+str(i) for i in list(range(0, nb_taxa))]

    t = Tree()
    t.populate(nb_taxa,
            names_library=taxa_names,
            random_branches=False)

    if unroot and nb_taxa>2: t.unroot()

    t.sort_descendants()

    # Add postorder ranking of nodes and return
    t, taxa, interns = set_postorder_ranks(t)

    # Populate branches using postrank attribute 
    for node in t.traverse("postorder"):
        if not node.is_root():
            node.dist = branch_lens[node.postrank]
        else:
            node.dist = 0.
    
    return t, taxa, interns

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
    if verbose: 
        print("Evolving sequences with the amazing Pyvolve")

    pytree = read_tree(tree=nwk_tree)

    parameters = None
    if subst_rates is not None or state_freqs is not None:
        parameters = dict()

        if subst_rates is not None:
            rs = subst_rates
            if isinstance(subst_rates, float):
                rs = {g:subst_rates for g in gtr_r}

            elif isinstance(subst_rates,
                    (list, np.ndarray, torch.Tensor)):
                assert len(subst_rates) == 6
                rs = {g:float(r) for g, r in zip(gtr_r,
                    subst_rates)}

            else:
                raise ValueError("Rates vector must be "\
                        "float, list, Tensor or ndarray ")

            parameters.update(mu=rs)

        if state_freqs is not None:
            # A C G T
            if isinstance(state_freqs, float):
                fs = [state_freqs] * 4

            elif isinstance(state_freqs,
                    (list, np.ndarray, torch.Tensor)):
                assert len(state_freqs) == 4
                fs = [float(s) for s in state_freqs]

            else:
                raise ValueError("Freqs vector must be "\
                        "float, list, Tensor or ndarray ")

            parameters.update(state_freqs=fs)

    m = Model("nucleotide", parameters=parameters) 

    p = Partition(size=nb_sites, models=m)
    e = Evolver(partitions=p, tree=pytree)

    e(seqfile=fasta_file, infofile=False, ratefile=False,
            write_anc=True, seed=seed)

    seqdict = e.get_sequences(anc=return_anc)

    return seqdict
