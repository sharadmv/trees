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
sns.despine()
import cPickle as pickle

X, y = trees.data.load('zoo')
X += np.random.normal(scale=0.05, size=X.shape)
N, D = X.shape
print N, D
with open('../../scripts/zoo.tree', 'rb') as fp:
    master_tree = pickle.load(fp)
with open('../../scripts/zoo.constraints', 'rb') as fp:
    master_constraints = pickle.load(fp)

random.seed(0)
random.shuffle(master_constraints)

train_percentage = 0.001

training_size = int(train_percentage * len(master_constraints))

print "Training constraints:", training_size

train_constraints, test_constraints = master_constraints[:training_size], master_constraints[training_size:]

test_constraints = test_constraints[:10000]

df = Inverse(c=1)

lm = GaussianLikelihoodModel(sigma=np.cov(X.T) / 2.0, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()

model = DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=train_constraints)
sampler = MetropolisHastingsSampler(model, X)
sampler.initialize_assignments()

n_iters = 2000000

scores = []
for i in tqdm(xrange(n_iters)):
    sampler.sample()
    if i % 100 == 0:
        wat = sampler.tree.score_constraints(train_constraints)
        assert wat == len(train_constraints), (wat, len(train_constraints))
        scores.append(float(sampler.tree.score_constraints(test_constraints))
                            / len(test_constraints))

fontsize = 18
plt.figure()
plt.xlim([0, n_iters/ 100])
plt.ylim([0, 1])
plt.xlabel("Iterations", fontsize=fontsize)
plt.ylabel("Constraint Score", fontsize=fontsize)
plt.plot(scores)
plt.legend(loc='best', fontsize=12)
plt.savefig("wat.png")

plt.figure()
plt.xlim([0, n_iters])
plt.xlabel("Iterations", fontsize=fontsize)
plt.ylabel("Data Log Likelihood", fontsize=fontsize)
plt.plot(sampler.likelihoods)
plt.savefig("wat2.png")
