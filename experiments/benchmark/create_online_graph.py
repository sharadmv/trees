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

import seaborn as sns

sns.set_style('white')
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

model = DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=[])
sampler = MetropolisHastingsSampler(model, X)
sampler.initialize_assignments()

constraint_add = 200
constraint_index = 0
n_iters = 40000

likelihoods = []
scores = []
for i in tqdm(xrange(n_iters + constraint_add)):
    if i != 0 and i % constraint_add == 0:
        sampler.add_constraint(train_constraints[constraint_index])
        constraint_index += 1
    sampler.sample()
    likelihoods.append(sampler.tree.marg_log_likelihood())
    scores.append(float(sampler.tree.score_constraints(test_constraints))
                        / len(test_constraints))

fontsize = 18
plt.figure()
plt.xlim([0, n_iters + constraint_add])
plt.ylim([0, 1])
plt.xlabel("Iterations", fontsize=fontsize)
plt.ylabel("Constraint Score", fontsize=fontsize)
plt.plot(scores)
plt.legend(loc='best', fontsize=12)
plt.savefig('online-scores.png', bbox_inches='tight')

plt.figure()
plt.xlim([0, n_iters + constraint_add])
plt.xlabel("Iterations", fontsize=fontsize)
plt.ylabel("Data Log Likelihood", fontsize=fontsize)
plt.plot(likelihoods)
plt.legend(loc='best', fontsize=12)

plt.savefig('online-likelihoods.png', bbox_inches='tight')

plt.show()
