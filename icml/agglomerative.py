from cStringIO import StringIO
from Bio import Phylo
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import single, to_tree, ward, average
from scipy.spatial.distance import pdist
import trees
from trees.tree import Tree, TreeNode, TreeLeaf
from util import plot_tree, dist
from master_tree import make_master

import seaborn as sns

sns.set_style('white')

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', default='zoo')
argparser.add_argument('--subset', default=150, type=int)
args = argparser.parse_args()

dataset = trees.data.load(args.dataset)
X, y = dataset.X, dataset.y


dataset_name = args.dataset

if dataset_name == 'mnist' or dataset_name == 'iris' or dataset_name == '20news':
    np.random.seed(0)
    idx = np.random.permutation(xrange(X.shape[0]))[:args.subset]
    X = X[idx]
    y = y[idx]
if dataset_name == 'mnist' or dataset_name == '20news':
    pca = PCA(10)
    X = pca.fit_transform(X)
if dataset_name == 'zoo':
    # pca = PCA(10)
    # X = pca.fit_transform(X)
    X += np.random.normal(size=X.shape) * 0.01

master_tree = make_master(X, y, args.dataset)
N, D = X.shape

C = pdist(X)

def make_tree(X, C, method='single'):
    if method == 'single':
        tree = to_tree(single(C))
    elif method == 'ward':
        tree = to_tree(ward(X))
    elif method == 'average':
        tree = to_tree(average(C))
    return Tree(root=construct_node(tree))

def construct_node(snode):
    if snode.left is None and snode.right is None:
        return TreeLeaf(snode.get_id())
    node = TreeNode()
    node.add_child(construct_node(snode.left))
    node.add_child(construct_node(snode.right))
    return node

single_tree = make_tree(X, C, method='single')
avg = make_tree(X, C, method='average')
ward_tree = make_tree(X, C, method='ward')

print "Single Tree:", dist(master_tree, single_tree)
print "Average Tree:", dist(master_tree, avg)
print "Ward Tree:", dist(master_tree, ward_tree)

# plt.figure()
# plot_tree(single_tree, y)
# plt.figure()
# plot_tree(ward_tree, y)
# plt.show()
