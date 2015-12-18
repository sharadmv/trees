import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import logging
logging.basicConfig(level=logging.INFO)
import cPickle as pickle
import numpy as np
from sklearn.decomposition import PCA
from path import Path
from tqdm import tqdm

from trees.data import load
from trees.interact import Database, Interactor
from trees.ddt import GaussianLikelihoodModel, DirichletDiffusionTree, Inverse
from trees.mcmc import MetropolisHastingsSampler

mnist = load('mnist')
database = Database('mnist')
interactor = Interactor(mnist, database)

X, y = mnist.X, mnist.y

X = X.astype(np.float32)
X /= 255.0

X -= X.mean(axis=0)

logging.debug("Finding PCA...")
pca_path = Path("pca.pkl")
if pca_path.exists():
    with open(pca_path, 'rb') as fp:
        pca = pickle.load(fp)
    X = pca.transform(X)
else:
    pca = PCA(15)
    X = pca.fit_transform(X)
    with open('pca.pkl', 'wb') as fp:
        pickle.dump(pca, fp)

idx = np.arange(X.shape[0])

subset_path = Path("subset.pkl")
if subset_path.exists():
    with open(subset_path, 'rb') as fp:
        idx = pickle.load(fp)
else:
    np.random.seed(1337)
    idx = np.random.permutation(idx)[:1000]
    with open(subset_path, 'wb') as fp:
        pickle.dump(idx, fp)

idx_map = {}

for i, v in enumerate(idx):
    idx_map[v] = i

X = X[idx]
y = y[idx]

constraints = set()

for constraint in interactor.current_interactions:
    a, b, c = constraint
    if a not in idx_map or b not in idx_map or c not in idx_map:
        continue
    constraints.add(tuple(map(lambda x: idx_map[x], constraint)))

logging.info("Interactions: %s", constraints)
logging.info("Interactions: %u", len(constraints))

N, D = X.shape

df = Inverse(c=2)
cov = np.cov(X.T) / 4.0 + np.eye(D) * 0.001
lm = GaussianLikelihoodModel(sigma=cov, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()

model = DirichletDiffusionTree(df=df,
                               likelihood_model=lm,
                               constraints=constraints)
sampler = MetropolisHastingsSampler(model, X)
sampler.initialize_assignments()

def iterate(iters):
    for i in tqdm(xrange(iters)):
        sampler.sample()
