from collections import OrderedDict
from ete3 import Tree

import numpy as np

__author__ = "amine"

__all__ = [
        "build_nwk_star_tree",
        "build_tree_from_nwk",
        "set_postorder_ranks",
        "get_postorder_branches",
        "get_postorder_branche_names"
        ]

def build_nwk_star_tree(b_lengths):
    bls = b_lengths.split(",")

    newick = "("
    for i, bl in enumerate(bls):
        newick += "t{}:{}".format(i+1, bl)
        if i<len(bls)-1: newick += ","
    newick += ");"

    return newick

# inspired from
# https://github.com/zcrabbit/vbpi-nf/blob/main/code/treeManipulation.py#L10 
def set_postorder_ranks(tree):
    """
    Add postrank attribute that contains the post order 
    traversal based rank (here, the order is changed so leaves
    are ranked before internal nodes)

    Example:
    ########

    nw = "((b,f)e,(a,c)x);"
    t = Tree(nw, format=1)
    t.sort_descendants()

    print(t)

    #      /-a
    #   /-|
    #  |   \-c
    #--|
    #  |   /-b
    #   \-|
    #      \-f

    print(t.write(format=1))
    # ((a:1,c:1)x:1,(b:1,f:1)e:1);

    _ = set_postorder_ranks(t)

    print({n.name:n.postrank for n in t.traverse("postorder")})
    #{'a': 0, 'c': 1, 'x': 4, 'b': 2, 'f': 3, 'e':5, 'root':6}

    """

    leaves = OrderedDict()
    internals = OrderedDict() # without root

    tree.sort_descendants()
    
    i, j, k = 0, len(tree), 0
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.postrank, i = i, i+1
            node.ancestral_postrank = -1

            leaves[node.name] = node.dist

        else:
            node.postrank, j = j, j+1
            node.ancestral_postrank, k = k, k+1

            if node.is_root():
                node.name = "root"
            else:
                if node.name == "":
                    node.name = "N" + \
                            str(node.ancestral_postrank)
                internals[node.name] = node.dist

    return tree, leaves, internals

def build_tree_from_nwk(nwk_tree):

    tree = Tree(nwk_tree, format=1)
    tree.sort_descendants()

    return set_postorder_ranks(tree)

def get_postorder_branches(tree):
    """
    Get branch lengths vector using postrank attribute
    """

    post_branches = np.zeros(len(tree.get_descendants()))

    for node in tree.traverse("postorder"):
        if not node.is_root():
            post_branches[node.postrank] = node.dist

    return post_branches

def get_postorder_branche_names(tree):
    """
    Get branch names using postrank attribute
    """

    post_branches = [""]*len(tree.get_descendants())

    for node in tree.traverse("postorder"):
        if not node.is_root():
            post_branches[node.postrank] = node.name

    return post_branches
