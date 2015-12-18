from cStringIO import StringIO
from Bio import Phylo
import matplotlib
import matplotlib.pyplot as plt
import random
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import trees
from trees.ddt import DirichletDiffusionTree, Inverse, GaussianLikelihoodModel
from trees.mcmc import MetropolisHastingsSampler
matplotlib.rcParams.update({'font.size': 48})
from trees.util import plot_tree_2d

import seaborn as sns

sns.set_style('white')
from tqdm import tqdm
from sklearn.decomposition import PCA
import cPickle as pickle

pca = PCA(2)
dataset = trees.data.load('zoo')
X, y = dataset.X, dataset.y
X += np.random.normal(scale=0.01, size=X.shape)
pca.fit(X)
N, D = X.shape
with open('../../scripts/zoo.tree', 'rb') as fp:
    master_tree = pickle.load(fp)

master_constraints = list(master_tree.generate_constraints())
random.seed(0)
random.shuffle(master_constraints)
train_constraints, test_constraints = master_constraints[:200], master_constraints[200:]

test_constraints = test_constraints[:10000]

df = Inverse(c=0.9)

lm = GaussianLikelihoodModel(sigma=np.eye(D) / 4.0, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()

models = {
    'No constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=[]),
    # '10 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints[:10]),
    '50 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints[:50]),
    '100 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints[:100]),
    # '150 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints[:150]),
    '200 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints),
}

samplers = {
    a : MetropolisHastingsSampler(d, X) for a, d in models.iteritems()
}

for sampler in samplers.values():
    sampler.initialize_assignments()


score_every = 1000

def iterate(n_iters):
    scores = {a: [] for a in samplers}
    likelihoods = {a: [] for a in samplers}
    for i in tqdm(xrange(n_iters)):
        for name, sampler in samplers.items():
            sampler.sample()
            likelihoods[name].append(sampler.tree.marg_log_likelihood())
            if i % score_every == 0:
                scores[name].append(float(sampler.tree.score_constraints(test_constraints))
                                    / len(test_constraints))
    for name, sampler in samplers.items():
        scores[name].append(float(sampler.tree.score_constraints(test_constraints))
                            / len(test_constraints))
    return scores, likelihoods

n_iters = 100000
scores, likelihoods = iterate(n_iters)
fontsize = 16

plt.figure()
plt.ylim([0, 1])
plt.xlim([0, n_iters])
plt.xlabel("Iterations", fontsize=fontsize)
plt.ylabel("Constraint Score", fontsize=fontsize)
for name, score in scores.items():
    plt.plot(np.arange(0, n_iters + score_every, score_every), score, label=name)
plt.legend(loc='best', fontsize=12)
plt.savefig('offline-scores.png', bbox_inches='tight')

plt.figure()
plt.xlim([0, n_iters])
# plt.ylim(ymin=-400)
plt.xlabel("Iterations", fontsize=fontsize)
plt.ylabel("Data Log Likelihood", fontsize=fontsize)
for name, likelihood in likelihoods.items():
    plt.plot(likelihood, label=name)
plt.legend(loc='best', fontsize=12)
plt.savefig('offline-likelihoods.png', bbox_inches='tight')

for type, model in models.items():
    final_tree = model.copy()

    plt.figure()
    plot_tree_2d(final_tree, X, pca)

    for node in final_tree.dfs():
        if node.is_leaf():
            node.point = y[node.point]

    newick = final_tree.to_newick()
    tree = Phylo.read(StringIO(newick), 'newick')

    Phylo.draw_graphviz(tree, prog='neato')
    plt.savefig('tree-%s.png' % type, bbox_inches='tight')
plt.show()

