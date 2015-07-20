import numpy as np
import scipy.stats as stats
from ..distribution import Distribution

class GaussianParameterProcess(Distribution):

    def __init__(self, mu0, sigma0, sigmat, sigma, eta=0.9):
        self.parameters = {}
        self.mu0, self.sigma0 = mu0, sigma0
        self.sigmat = sigmat
        self.sigma = sigma

        self.sigma0inv = np.linalg.inv(self.sigma0)
        self.sigmatinv = np.linalg.inv(self.sigmat)
        self.sigmainv = np.linalg.inv(self.sigma)
        self.eta = eta

        self.sigma0inv_mu0 = np.dot(self.sigma0inv, self.mu0)

    def generate(self, parameter=None):
        if parameter is None:
            parameter = self.mu0
        return stats.multivariate_normal(mean=self.eta * parameter, cov=self.sigma0).rvs()

    def prior_log_likelihood(self, x):
        return stats.multivariate_normal(mean=self.mu0, cov=self.sigma0).logpdf(x)

    def transition_log_likelihood(self, mu1, mu2):
        return stats.multivariate_normal(mean=self.eta * mu1, cov=self.sigmat).logpdf(mu2)

    def data_log_likelihood(self, x, mu):
        return stats.multivariate_normal(mean=mu, cov=self.sigma).logpdf(x)

    def sample_one(self, parameter):
        return stats.multivariate_normal(mean=parameter, cov=self.sigma).rvs()

    def sample_posterior(self, data, children, parent):
        Nd, _ = data.shape
        if len(children.shape) == 1:
            Nc = 0
        else:
            Nc, _ = children.shape

        sigma0inv = self.sigma0inv
        sigma0inv_mu0 = self.sigma0inv_mu0
        if parent is not None:
            sigma0inv = self.sigmatinv
            sigma0inv_mu0 = np.dot(self.sigmatinv, self.eta * parent)

        sigman = np.linalg.inv(sigma0inv + Nd * self.sigmainv + Nc * self.sigmatinv)
        children_mean = 0
        if Nc > 0:
            children_mean = np.dot(self.sigmatinv, children.sum(axis=0))
        mun = np.dot(sigman, sigma0inv_mu0 +
                     np.dot(self.sigmainv, data.sum(axis=0)) +
                     children_mean)
        return stats.multivariate_normal(mean=mun, cov=sigman).rvs()
