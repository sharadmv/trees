import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def generate_dataset(D, N, K, mu0, sigma0, sigma):

    z = np.random.randint(K, size=N)
    mu = np.random.multivariate_normal(mean=mu0, cov=sigma0, size=K)
    X = np.zeros((N, D))
    for i in xrange(N):
        X[i] = np.random.multivariate_normal(mean=mu[z[i]], cov=sigma)
    return X, z

def plot_dataset(X, z, K):
    colors = sns.color_palette(n_colors=K)
    for k in xrange(K):
        x, y = X[z == k].T
        plt.scatter(x, y, color=colors[k])
