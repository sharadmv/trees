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

    def inverse_cumulative(self, y, t1, t2):
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

    @theanify(T.dscalar('y'))
    def inverse_cumulative(self, y, t1, t2):
        return 1 - T.exp(-y / self.c)

    def get_parameters(self):
        return {"c"}

class InverseQuadratic(DivergenceFunction):

    @theanify(T.dscalar('t'))
    def divergence(self, t):
        return self.b + self.d / (1 - t) ** 2

    @theanify(T.dscalar('t'))
    def cumulative_divergence(self, t):
        return self.b * t - self.d - self.d / (t - 1)

    @theanify(T.dscalar('y'))
    def inverse_cumulative(self, y, t1, t2):
        if self.b == 0:
            return y / (self.d + y)
        b = self.b
        d = self.d
        return (T.sqrt(b ** 2 + 2 * b * (d - y) + (d + y) ** 2) + b + d + y) / (2 * b)

    def get_parameters(self):
        return {"b", "d"}
