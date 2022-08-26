from pyvolve import read_tree, Model, Partition, Evolver

def build_star_tree(b_lengths):
    bls = b_lengths.split(",")

    newick = "("
    for i, bl in enumerate(bls):
        newick += "t{}:{}".format(i+1, bl)
        if i<len(bls)-1: newick += ","
    newick += ");"

    return newick

def evolve_seqs_full_homogeneity(
        nwk_tree,
        nb_sites=10, 
        fasta_file=False,
        subst_rates=None,
        state_freqs=None,
        return_anc=True,
        seed=None,
        verbose=False):

    gtr_r = ["AG", "AC", "AT", "GC", "GT", "CT"]
 
    # Evolve sequences
    if verbose: print("Evolving new sequences with the amazing "\
            "Pyvolve for {}".format(fasta_file))
    tree = read_tree(tree=nwk_tree)

    parameters = None
    if subst_rates is not None or state_freqs is not None:
        parameters = dict()

        if subst_rates is not None:
            rs = subst_rates
            if isinstance(subst_rates, float):
                rs = {g:subst_rates for g in gtr_r}

            elif isinstance(subst_rates, list):
                assert len(subst_rates) == 6
                rs = {g:float(r) for g, r in zip(gtr_r, subst_rates)}

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
    e = Evolver(partitions=p, tree=tree)

    e(seqfile=fasta_file, infofile=False, ratefile=False,
            write_anc=True, seed=seed)

    seqdict = e.get_sequences(anc=return_anc)

    return seqdict["root"],\
            [seqdict[s] for s in seqdict if s != "root"]
