from collections import OrderedDict
from ete3 import Tree

__author__ = "amine"

__all__ = [
        "build_nwk_star_tree",
        "build_tree_from_nwk"
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
def build_tree_from_nwk(nwk_tree):

    tree = Tree(nwk_tree, format=1)

    leaves = OrderedDict()
    internals = OrderedDict()

    i, j, k= 0, len(tree), 0
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            leaves[node.name] = node.dist
            node.postrank, i = i, i+1
            node.ancestral_postrank = -1

        else:
            if node.is_root():
                node.name = "root"
            else:
                internals[node.name] = node.dist

            node.postrank, j = j, j+1
            node.ancestral_postrank, k = k, k+1

    return tree, leaves, internals
