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

if __name__ == "__main__":
    with open('scripts/zoo.tree', 'rb') as fp:
        master_tree = pickle.load(fp)
    points = master_tree.root.points()

    tree1 = DirichletDiffusionTree(df=df, likelihood_model=lm)
    sampler1 = MetropolisHastingsSampler(tree1, X)
    sampler1.initialize_assignments()
    sampler1.tree = sampler1.tree.induced_subtree(points)

    tree2 = DirichletDiffusionTree(df=df, likelihood_model=lm)
    sampler2 = MetropolisHastingsSampler(tree2, X)
    sampler2.initialize_assignments()
    sampler2.tree = sampler2.tree.induced_subtree(points)

    all_constraints = list(master_tree.generate_constraints())

    np.random.seed(0)
    np.random.shuffle(all_constraints)
    test_constraints = all_constraints[:10000]

    satisfied = [set(), set()]

    iterate(sampler1, 100)
    iterate(sampler2, 100)

    sampler_costs = [None, None]
    scores = [[], []]
    for val, sampler in enumerate([sampler1, sampler2]):
        sampler_cost = []
        for i in xrange(100):
            trees, costs = iterate(sampler, 1000)
            sampler_cost.extend(costs)

            if val == 0:
                constraint = random.choice(all_constraints)
                while constraint in satisfied[val]:
                    constraint = random.choice(all_constraints)
            elif val == 1:
                idx = np.random.permutation(list(points))[:20]
                combos = get_variance(trees, idx)
                constraint = get_constraint(master_tree, combos[0])
                id = 1
                while constraint in satisfied[val]:
                    constraint = get_constraint(combos[id])
                    id += 1
            print "Adding", val, constraint
            sampler.add_constraint(constraint)
            satisfied[val].add(constraint)
            scores[val].append(float(sampler.tree.score_constraints(test_constraints))
                                / len(test_constraints))
        sampler_costs[val] = sampler_cost

    plt.plot(sampler_costs[0], label='random')
    plt.plot(sampler_costs[1], label='entropy')
    plt.show()
