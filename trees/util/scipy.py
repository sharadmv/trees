from ..distribution import Distribution

class ScipySampler(Distribution):

    def __init__(self, rv, parameters={}, seed=None):
        super(ScipySampler, self).__init__(parameters=parameters, seed=seed)
        self.rv = rv

    def sample(self, n_samples):
        return self.rv.rvs(size=n_samples)

    def sample_one(self):
        return self.rv.rvs()

    def parameters(self):
        return []
