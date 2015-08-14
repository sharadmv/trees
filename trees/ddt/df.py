import theano.tensor as T
from theanify import theanify, Theanifiable

class DivergenceFunction(Theanifiable):

    def __init__(self, **parameters):
        super(DivergenceFunction, self).__init__()
        for param in self.get_parameters():
            assert param in parameters, 'Missing parameter %s' % param
        self.parameters = parameters
        self.compile()

    def __getattr__(self, key):
        if key in self.parameters:
            return self.parameters[key]

    def divergence(self, t):
        raise NotImplementedError

    def cumulative_divergence(self, t):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

class Inverse(DivergenceFunction):

    @theanify(T.dscalar('t'))
    def divergence(self, t):
        return self.c / (1 - t)

    @theanify(T.dscalar('t'))
    def cumulative_divergence(self, t):
        return -self.c * T.log(1 - t)

    def get_parameters(self):
        return {"c"}

class InverseQuadratic(DivergenceFunction):

    @theanify(T.dscalar('t'))
    def divergence(self, t):
        return self.b + self.d / (1 - t) ** 2

    @theanify(T.dscalar('t'))
    def cumulative_divergence(self, t):
        return self.b * t - self.d - self.d / (t - 1)

    def get_parameters(self):
        return {"b", "d"}
