import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from trees.util import plot_tree
from trees.ddt import DirichletDiffusionTree, Inverse, GaussianLikelihoodModel

import matplotlib.pyplot as plt

if __name__ == "__main__":
    X = np.matrix([
        [0, 0],
        [1, 1],
        [2, 2]
    ])
    N, D = X.shape
    y = ['wat', 'who', 'where']
    df = Inverse(c=1)
    lm = GaussianLikelihoodModel(sigma=np.eye(D), mu0=np.zeros(D), sigma0=np.eye(D))
    ddt = DirichletDiffusionTree(df=df,
                                 likelihood_model=lm)
    ddt.initialize_from_data(X)
    a, p = ddt.sample_assignment()
    #plot_tree(ddt)
