import numpy as np
from theanify import theanify, Theanifiable
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T

class LikelihoodModel(Theanifiable):

    def __init__(self, **parameters):
        super(LikelihoodModel, self).__init__()
        for param in self.get_parameters():
            assert param in parameters, "Missing parameter: %s" % param
        self.parameters = parameters

    def __getattr__(self, key):
        if key in self.parameters:
            return self.parameters[key]


class GaussianLikelihoodModel(LikelihoodModel):

    def __init__(self, **parameters):
        super(GaussianLikelihoodModel, self).__init__(**parameters)
        self.sigma0inv = np.linalg.inv(self.sigma0)
        self.D = self.sigma.shape[0]
        self.compile()

    def transition_probability(self, parent, child):
        child_latent, child_time = child.get_state('latent_value'), child.get_state('time')
        if parent is None:
            return self.calculate_transition(child_latent, self.mu0, child_time, -1)
        parent_latent, parent_time = parent.get_state('latent_value'), parent.get_state('time')
        assert parent_time < child_time, (parent_time, child_time)
        return self.calculate_transition(child_latent, parent_latent, child_time, parent_time)

    @theanify(T.dvector('state'), T.dvector('parent'), T.dscalar('time'), T.dscalar('parent_time'))
    def calculate_transition(self, state, parent, time, parent_time):
        sigma = (time - parent_time) * self.sigma
        mu = parent

        logdet = T.log(T.nlinalg.det(sigma))
        delta = state - mu
        pre = -(self.D / 2.0 * np.log(2 * np.pi) + 1/2.0 * logdet)
        return pre + -0.5 * (T.dot(delta, T.dot(T.nlinalg.matrix_inverse(sigma), delta)))

    @theanify(T.dvector('mean'), T.dmatrix('cov'))
    def sample(self, mean, cov):
        e, v = T.nlinalg.eigh(cov)
        x = RandomStreams().normal(size=(self.D,))
        x = T.dot(x, T.sqrt(e)[:, None] * v)
        return x + mean

    def sample_transition(self, node, parent):
        children = node.children
        time = node.get_state('time')
        if parent is None:
            mu0 = self.mu0
            sigma0 = self.sigma0
            sigma0inv = self.sigma0inv
        else:
            mu0 = parent.get_state('latent_value')
            sigma0 = self.sigma * (time - parent.get_state('time'))
            sigma0inv = np.linalg.inv(sigma0)

        mus = [c.get_state('latent_value') for c in children]

        sigmas = [self.sigma * (c.get_state('time') - time) for c in children]

        sigmas_inv = [np.linalg.inv(s) for s in sigmas]

        sigman = np.linalg.inv(sigma0inv + sum(sigmas_inv))
        mun = np.dot(sigman, np.dot(sigma0inv, mu0) + sum([np.dot(a, b) for a, b in zip(sigmas_inv, mus)]))
        return self.sample(mun, sigman)

    def get_parameters(self):
        return {"sigma", "sigma0", "mu0"}

