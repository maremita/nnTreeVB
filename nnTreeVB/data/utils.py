from collections import OrderedDict
from ete3 import Tree

__author__ = "amine"

__all__ = [
        "build_nwk_star_tree",
        "build_tree_from_nwk",
        "set_postorder_ranks"
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
    leaves = OrderedDict()
    internals = OrderedDict() # without root

    i, j, k= 0, len(tree), 0
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
                node.name = "N"+str(node.ancestral_postrank)
                internals[node.name] = node.dist

    return tree, leaves, internals

def build_tree_from_nwk(nwk_tree):

    tree = Tree(nwk_tree, format=1)
    tree.sort_descendants()

    return set_postorder_ranks(tree)

def get_postorder_branches(tree):
    post_branches = []

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            post_branches.append(node.dist)
        elif not node.is_root():
            post_branches.append(node.dist)

    return post_branches
