import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import trees
from trees.ddt import *
import mpld3
import seaborn as sns
sns.set_style('white')
from tqdm import tqdm
from sklearn.decomposition import PCA

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--iterations', type=int, default=1000)
argparser.add_argument('--N', type=int, default=20)

args = argparser.parse_args()

X, y = trees.data.load('zoo')
pca = PCA(n_components=2)
X = pca.fit_transform(X)
X += np.random.normal(scale=.15, size=X.shape)
X = X[0:args.N]
y = y[0:args.N]
N, D = X.shape

df = Inverse(c=5)

lm = GaussianLikelihoodModel(sigma=np.cov(X.T) / 2.0, sigma0=np.eye(D) / 10.0, mu0=X.mean(axis=0)).compile()
sampler = MetropolisHastingsSampler(DirichletDiffusionTree(df, lm), X)
sampler.initialize_assignments()

def iterate(n_iters):
    lls = []
    for i in tqdm(xrange(n_iters)):
        sampler.sample()
        lls.append(sampler.ddt.marg_log_likelihood())
    return lls

iterate(args.iterations)
