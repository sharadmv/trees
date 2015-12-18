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
import trees
from trees.ddt import DirichletDiffusionTree, Inverse, GaussianLikelihoodModel
from trees.mcmc import MetropolisHastingsSampler
from trees.util import plot_tree_2d
from sklearn.decomposition import PCA

import seaborn as sns

sns.set_style('white')
import cPickle as pickle

dataset = trees.data.load('zoo')
X, y = dataset.X, dataset.y
X += np.random.normal(scale=0.01, size=X.shape)

pca = PCA(2)
pca.fit(X)

# X = pca.transform(X)
N, D = X.shape

df = Inverse(c=0.9)

lm = GaussianLikelihoodModel(sigma=np.eye(D) / 4.0, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()

model = DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=[])
sampler = MetropolisHastingsSampler(model, X)
sampler.initialize_assignments()

likelihoods = []
fontsize = 18

def iterate(n_iters):
    for i in tqdm(xrange(n_iters)):
        sampler.sample()
        likelihoods.append(sampler.tree.marg_log_likelihood())

    plt.figure()
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Data Log Likelihood", fontsize=fontsize)
    plt.plot(likelihoods)
    plt.legend(loc='best', fontsize=12)

    plt.savefig('unconstrained-likelihoods.png', bbox_inches='tight')


    final_tree = sampler.tree.copy()

    plt.figure()
    plot_tree_2d(final_tree, X, pca)

    for node in final_tree.dfs():
        if node.is_leaf():
            node.point = y[node.point]

    plt.figure()
    newick = final_tree.to_newick()
    tree = Phylo.read(StringIO(newick), 'newick')

    Phylo.draw_graphviz(tree, prog='neato')
    plt.savefig('unconstrained-tree.png', bbox_inches='tight')
    graph = Phylo.to_networkx(tree)
    with open('unconstrained-tree.nwk', 'w') as fp:
        print >>fp, newick,
    nx.write_dot(graph, 'unconstrained-tree.dot')
    plt.show()
