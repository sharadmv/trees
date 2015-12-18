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

with open('../../scripts/zoo.tree', 'rb') as fp:
    master_tree = pickle.load(fp)

master_constraints = list(master_tree.generate_constraints())
random.seed(0)
random.shuffle(master_constraints)
train_constraints, test_constraints = master_constraints[:200], master_constraints[200:]
test_constraints = test_constraints[:10000]

df = Inverse(c=0.9)

lm = GaussianLikelihoodModel(sigma=np.eye(D) / 4.0, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()

model = DirichletDiffusionTree(df=df, likelihood_model=lm, constraints=[])
sampler = MetropolisHastingsSampler(model, X)
sampler.initialize_assignments()


constraint_add = 500
constraint_index = 0
n_iters = 100000
score_every = 1000

likelihoods = []
scores = []
for i in tqdm(xrange(n_iters + constraint_add)):
    if i != 0 and i % constraint_add == 0:
        sampler.add_constraint(train_constraints[constraint_index])
        constraint_index += 1
    sampler.sample()
    likelihoods.append(sampler.tree.marg_log_likelihood())
    if i % score_every == 0:
        scores.append(float(sampler.tree.score_constraints(test_constraints))
                            / len(test_constraints))

scores.append(float(sampler.tree.score_constraints(test_constraints))
                    / len(test_constraints))

fontsize = 18
plt.figure()
plt.xlim([0, n_iters + constraint_add])
plt.ylim([0, 1])
plt.xlabel("Iterations", fontsize=fontsize)
plt.ylabel("Constraint Score", fontsize=fontsize)
plt.plot(np.arange(0, n_iters + constraint_add + score_every, score_every), scores)
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


final_tree = sampler.tree.copy()

plt.figure()
plot_tree_2d(final_tree, X, pca)

for node in final_tree.dfs():
    if node.is_leaf():
        node.point = y[node.point]

newick = final_tree.to_newick()
tree = Phylo.read(StringIO(newick), 'newick')

plt.figure()
Phylo.draw_graphviz(tree, prog='neato')
plt.savefig('tree.png', bbox_inches='tight')
graph = Phylo.to_networkx(tree)
with open('tree.nwk', 'w') as fp:
    print >>fp, newick,
nx.write_dot(graph, 'tree.dot')
plt.show()
