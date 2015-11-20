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

import seaborn as sns

sns.set_style('white')
from tqdm import tqdm
from sklearn.decomposition import PCA
import cPickle as pickle

X, y = trees.data.load('zoo')
pca = PCA(n_components=2)
X = pca.fit_transform(X)
X += np.random.normal(scale=0.05, size=X.shape)
idx = xrange(20)
X = X[idx]
N, D = X.shape
with open('../../scripts/zoo.tree', 'rb') as fp:
    master_tree = pickle.load(fp).induced_subtree(idx)
master_constraints = list(master_tree.generate_constraints())
random.seed(0)
random.shuffle(master_constraints)
train_constraints, test_constraints = master_constraints[:200], master_constraints[200:]

df = Inverse(c=1)

lm = GaussianLikelihoodModel(sigma=np.cov(X.T) / 4.0, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()

models = {
    'No constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=[]),
    '10 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints[:10]),
    # '50 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints[:50]),
    '100 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints[:100]),
    # '150 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints[:150]),
    '200 constraints': DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints),
}

samplers = {
    a : MetropolisHastingsSampler(d, X) for a, d in models.iteritems()
}

for sampler in samplers.values():
    sampler.initialize_assignments()

def iterate(n_iters):
    scores = {a: [] for a in samplers}
    likelihoods = {a: [] for a in samplers}
    for i in tqdm(xrange(n_iters)):
        for name, sampler in samplers.items():
            sampler.sample()
            likelihoods[name].append(sampler.tree.marg_log_likelihood())
            scores[name].append(float(sampler.tree.score_constraints(test_constraints))
                                / len(test_constraints))
    return scores, likelihoods

scores, likelihoods = iterate(10000)
fontsize = 16

plt.figure()
plt.ylim([0, 1])
plt.xlim([0, 10000])
plt.xlabel("Iterations", fontsize=fontsize)
plt.ylabel("Constraint Score", fontsize=fontsize)
for name, score in scores.items():
    plt.plot(score, label=name)
plt.legend(loc='best', fontsize=12)
plt.savefig('offline-scores.png', bbox_inches='tight')

plt.figure()
plt.xlim([0, 10000])
plt.ylim(ymin=-400)
plt.xlabel("Iterations", fontsize=fontsize)
plt.ylabel("Data Log Likelihood", fontsize=fontsize)
for name, likelihood in likelihoods.items():
    plt.plot(likelihood, label=name)
plt.legend(loc='best', fontsize=12)

plt.savefig('offline-likelihoods.png', bbox_inches='tight')
