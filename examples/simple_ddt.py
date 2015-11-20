import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from trees.util import plot_tree, plot_tree_2d
from trees.ddt import DirichletDiffusionTree, Inverse, GaussianLikelihoodModel
from trees.mcmc import MetropolisHastingsSampler
from tqdm import tqdm


if __name__ == "__main__":
    D = 2
    N = 100
    X = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D), size=N).astype(np.float32)
    df = Inverse(c=1)
    lm = GaussianLikelihoodModel(sigma=np.eye(D) / 4.0, mu0=np.zeros(D), sigma0=np.eye(D))
    ddt = DirichletDiffusionTree(df=df,
                                 likelihood_model=lm)
    mh = MetropolisHastingsSampler(ddt, X)
    mh.initialize_assignments()

    for _ in tqdm(xrange(1000)):
        mh.sample()

    plt.figure()
    plt.plot(mh.likelihoods)

    plt.figure()
    plot_tree(mh.tree)

    plt.figure()
    plot_tree_2d(mh.tree, X)

    plt.show()
