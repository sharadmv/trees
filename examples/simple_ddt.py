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
    X = np.matrix([
        [0, 0],
        [0, 1],
        [1, 1]
    ]).astype(np.float32)
    N, D = X.shape
    y = ['wat', 'who', 'where']
    df = Inverse(c=1)
    lm = GaussianLikelihoodModel(sigma=np.eye(D), mu0=np.zeros(D), sigma0=np.eye(D))
    ddt = DirichletDiffusionTree(df=df,
                                 likelihood_model=lm)
    mh = MetropolisHastingsSampler(ddt, X)
    mh.initialize_assignments()

    # plot_tree_2d(mh.tree, X)
    # plt.show()

    # for _ in tqdm(xrange(10000)):
        # mh.sample()

    # plot_tree_2d(mh.tree, X)
    # plt.show()
