import numpy as np

class Distribution(object):

    def __init__(self, parameters={}, seed=None):
        np.random.seed(seed)
        for parameter_name in self.parameters():
            assert parameter_name in parameters, "Parameter<%s> not provided." % parameter_name

        self.parameters = parameters

    def get_parameter(self, name):
        return self.parameters[name]

    def set_parameter(self, name, value):
        self.parameters[name] = value

    def sample_one(self):
        raise NotImplementedError

    def sample(self, n_samples):
        return [self.sample_one() for _ in xrange(n_samples)]

    def parameters(self):
        raise NotImplementedError
