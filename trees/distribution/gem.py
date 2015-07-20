import numpy as np
import scipy.stats as stats

from ..util import ScipySampler, DistributionStream
from ..distribution import Distribution

class GEM(Distribution):

    def __getitem__(self, key):
        return self.get_weight(key)

    def sample_one(self):
        return self.uniform_index(np.random.random())[0]

    def uniform_index(self, u):
        location = 0
        i = -1
        while location < u:
            i += 1
            weight = self.get_weight(i)
            location += weight
        return i, location

    def get_weight(self, index):
        raise NotImplementedError

    def get_parameters(self):
        return {"a"}

class TruncatedGEM(GEM):

    def __init__(self, *args, **kwargs):
        super(TruncatedGEM, self).__init__(*args, **kwargs)

        a = self.get_parameter("a")
        max_length = self.get_parameter("max_length")
        self.betas = np.random.beta(1, a, size=max_length)
        self.weights = np.zeros(max_length)

        prev = 1
        for i in xrange(max_length - 1):
            self.weights[i] = self.betas[i] * prev
            prev *= (1 - self.betas[i])
        self.weights[max_length - 1] = 1 - np.sum(self.weights)
        np.testing.assert_equal(1, np.sum(self.weights))

    def get_weight(self, index):
        max_length = self.get_parameter("max_length")
        if index >= max_length:
            raise Exception("index desired greater than max_length, %u" % max_length)
        return self.weights[index]

    def get_parameters(self):
        return {"a", "max_length"}

class LazyGEM(GEM):

    def __init__(self, *args, **kwargs):
        super(LazyGEM, self).__init__(*args, **kwargs)

        a = self.get_parameter("a")
        self.betas = DistributionStream(ScipySampler(stats.beta(1, a)))
        self.weights = np.zeros(0)
        self.index = -1
        self.prev = 1.0

    def get_weight(self, index):
        if index > self.index:
            diff = index - self.index
            weights = np.zeros(diff)
            self.weights = np.concatenate([self.weights, weights])
            for i in xrange(self.index + 1, index + 1):
                self.weights[i] = self.betas[i] * self.prev
                self.prev *= (1 - self.betas[i])
                self.index += 1
        return self.weights[index]
