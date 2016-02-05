from Bio import Phylo
from cStringIO import StringIO
import numpy as np
import logging
# logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import seaborn as sns
from trees.dasgupta import DasguptaTree
from trees.mcmc import SPRSampler
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from itertools import combinations

from trees.data import load
from sklearn.decomposition import PCA

data = load('zoo')
X, y = data.X, data.y

pca = PCA(10)
X = pca.fit_transform(X)
X += np.random.normal(size=X.shape) * 0.01

N = X.shape[0]
np.random.seed(0)
# idx = np.random.permutation(np.arange(N))[:20]

# X = X[idx]
# y = np.array(y)
# y = y[idx]

D = 1.0 / squareform(pdist(X))

tree = DasguptaTree(D, 1)
tree.initialize_assignments(xrange(X.shape[0]))

sampler = SPRSampler(tree, X)

ordering = np.loadtxt('ordering.txt', dtype='|S8')
ordering_idx = [y.index(a) for a in ordering]

def plot_tree(tree):
    final_tree = tree.copy()
    for node in final_tree.dfs():
        if node.is_leaf():
            node.point = y[node.point]

    newick = final_tree.to_newick()
    tree = Phylo.read(StringIO(newick), 'newick')
    Phylo.draw_graphviz(tree, prog='neato')
    plt.show()

def iterate(n):
    trees = []
    for i in tqdm(xrange(n)):
        sampler.sample()
        trees.append(sampler.tree)
    return trees

def get_tree_distance(u, v):
    i = 0
    if u == v:
        return 0
    while u[i] == v[i]:
        i += 1
    return len(u[i:]) + len(v[i:])

def create_depth_matrix(tree):
    mat = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(i, N):
            u, v = tree.point_index(i), tree.point_index(j)
            if u is None or v is None:
                mat[i, j] = np.inf
                mat[j, i] = np.inf
            else:
                mat[i, j] = get_tree_distance(u, v)
                mat[j, i] = get_tree_distance(u, v)
    return mat

def get_three(combo):
    a, b, c = combo
    return ((a, b, c), (b, c, a), (c, a, b))

def get_variance(trees, idx):
    combos = list(combinations(idx, 3))
    trees = [tree.induced_subtree(idx) for tree in trees]
    sats = []
    for tree in tqdm(trees):
        sat = []
        for c1, c2, c3 in map(get_three, combos):
            if tree.verify_constraint(c1):
                sat.append([1, 0, 0])
            elif tree.verify_constraint(c2):
                sat.append([0, 1, 0])
            else:
                sat.append([0, 0, 1])
        sats.append(sat)
    return np.array(sats), combos

def measure_variance(iters, idx, a=0.1):
    trees = iterate(iters)
    sats, combos = get_variance(trees, idx)
    means = sats.mean(axis=0)
    means = (means + a) / 1.3
    logs = np.log(means) / np.log(3.0)
    entropy = - (means * logs).sum(axis=1)
    return [[y[i] for i in c] for c in np.array(combos)[entropy.argsort()[::-1]]], vars
