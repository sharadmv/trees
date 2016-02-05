from Bio import Phylo
from cStringIO import StringIO
import numpy as np
# logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import seaborn as sns
from trees.mcmc import MetropolisHastingsSampler
from trees.ddt import *
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from itertools import combinations
import cPickle as pickle
import random

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

N, D = X.shape
df = Inverse(c=1)

lm = GaussianLikelihoodModel(sigma=np.eye(D) / 4.0, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()

tree = DirichletDiffusionTree(df=df, likelihood_model=lm)
sampler = MetropolisHastingsSampler(tree, X)
sampler.initialize_assignments()

D = 1.0 / squareform(pdist(X))

def plot_tree(tree):
    final_tree = tree.copy()
    for node in final_tree.dfs():
        if node.is_leaf():
            node.point = y[node.point]

    newick = final_tree.to_newick()
    tree = Phylo.read(StringIO(newick), 'newick')
    Phylo.draw_graphviz(tree, prog='neato')
    plt.show()

def iterate(sampler, n):
    costs, trees = [], []
    for i in tqdm(xrange(n)):
        sampler.sample()
        trees.append(sampler.tree)
        costs.append(sampler.tree.marg_log_likelihood())
    return trees, costs

def get_tree_distance(u, v):
    i = 0
    if u == v:
        return 0
    while u[i] == v[i]:
        i += 1
    return len(u[i:]) + len(v[i:])

def create_depth_matrix(tree):
    points = list(tree.root.points())
    N = len(points)
    mat = np.zeros((N, N))
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            u, v = tree.point_index(p1), tree.point_index(p2)
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

def get_variance(trees, idx, a=0.01):
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
    sats = np.array(sats)
    means = sats.mean(axis=0)
    means = (means + a) / 1.3
    logs = np.log(means) / np.log(3.0)
    entropy = -(means * logs).sum(axis=1)
    return np.array(combos)[entropy.argsort()[::-1]]

def get_constraint(tree, constraint):
    a, b, c = constraint
    if tree.verify_constraint((a, b, c)):
        return (a, b, c)
    if tree.verify_constraint((a, c, b)):
        return (a, c, b)
    if tree.verify_constraint((b, c, a)):
        return (b, c, a)

def get_vars(sampler, iters, K=10, N=10):
    trees, _ = iterate(sampler, iters)
    sub_idx = []
    subtrees = []
    depths = []

    points = sampler.tree.root.points()
    for i in tqdm(xrange(N)):
        idx = random.sample(points, K)
        sub_idx.append(idx)
        subtree = []
        depth = []
        for t in tqdm(trees, nested=True):
            st = t.induced_subtree(idx)
            subtree.append(st)
            depth.append(create_depth_matrix(st))
        subtrees.append(subtree)
        depths.append(depth)
    depths = np.array(depths)
    std = depths.std(axis=1)
    vars = []
    triu = np.triu_indices(K)
    for i in xrange(N):
        vars.append(std[i][triu].max())
    return np.array(vars), np.array(sub_idx), trees

if __name__ == "__main__":
    vars, idx, trees = get_vars(sampler, 1000, K=5, N=20)
