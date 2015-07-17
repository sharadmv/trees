import scipy.stats as stats

class GaussianParameterProcess(object):

    def __init__(self, mu0, sigma0, sigma):
        self.parameters = {}
        self.mu0, self.sigma0 = mu0, sigma0
        self.sigma = sigma

    def get_parameter(self, index):
        if index not in self.parameters:
            self.parameters[index] = stats.multivariate_normal(mean=self.mu0, cov=self.sigma0).rvs()
        return self.parameters[index]

    def log_likelihood(self, x, index):
        mu = self.get_parameter(index)
        return stats.multivariate_normal(mean=mu, cov=self.sigma).logpdf(x)
