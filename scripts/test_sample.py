import timeit
from trees.ddt import *
import numpy as np
import scipy.stats as stats

mean = np.zeros(2)
cov = np.eye(2)

lm = GaussianLikelihoodModel(sigma=np.eye(2), sigma0=np.eye(2), mu0=np.zeros(2)).compile()
num_samples = 100000000

def numpy_sample():
    for _ in xrange(num_samples):
        np.random.multivariate_normal(mean, cov)

def scipy_sample():
    for _ in xrange(num_samples):
        stats.multivariate_normal(mean, cov).rvs()

def theano_sample():
    for _ in xrange(num_samples):
        lm.sample(mean, cov)
if __name__ == '__main__':
    print timeit.timeit('numpy_sample', setup="from __main__ import numpy_sample")
    print timeit.timeit('scipy_sample', setup="from __main__ import scipy_sample")
    print timeit.timeit('theano_sample', setup="from __main__ import theano_sample")

