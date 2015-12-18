import networkx as nx
from cStringIO import StringIO
from Bio import Phylo
import matplotlib.pyplot as plt
import random
import logging
from tqdm import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import single, to_tree
from scipy.spatial.distance import pdist
import trees
from trees.tree import Tree, TreeNode, TreeLeaf

import seaborn as sns

sns.set_style('white')

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--tree', default=None)

args = argparser.parse_args()

dataset = trees.data.load('zoo')
X, y = dataset.X, dataset.y

animal_list = [
    'lark',
    'pheasant',
    'swan',
    'octopus',
    'scorpion',
    'housefly',
    'gorilla',
    'dolphin',
    'sealion',
    'platypus',
    'calf',
    'pony',
    'tortoise',
    'carp',
    'tuna',
    'stingray',
]

def plot_tree(final_tree, name):
    final_tree = final_tree.copy()
    for node in final_tree.dfs():
        if node.is_leaf():
            node.point = y[node.point]

    final_tree = final_tree.induced_subtree(animal_list)

    plt.figure()
    newick = final_tree.to_newick()
    tree = Phylo.read(StringIO(newick), 'newick')

    Phylo.draw_graphviz(tree, prog='neato')
    plt.savefig("%s.png" % name, dpi=200, bbox_inches='tight')
X += np.random.normal(scale=0.01, size=X.shape)

pca = PCA(2)
pca.fit(X)

# X = pca.transform(X)
N, D = X.shape

C = pdist(X)
tree = to_tree(single(C))

def construct_node(snode):
    if snode.left is None and snode.right is None:
        return TreeLeaf(snode.get_id())
    node = TreeNode()
    node.add_child(construct_node(snode.left))
    node.add_child(construct_node(snode.right))
    return node

root = construct_node(tree)
linkage_tree = Tree(root=root)
plot_tree(linkage_tree, 'linkage_induced')


if args.tree:
    with open(args.tree, 'r') as fp:
        ddt_tree = Tree.from_newick(fp.read())
    plot_tree(ddt_tree, 'ddt_induced')


