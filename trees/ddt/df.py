import theano.tensor as T
from theanify import theanify, Theanifiable
from theano.tensor.shared_randomstreams import RandomStreams

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

    @theanify(T.dscalar('t'))
    def log_divergence(self, t):
        return T.log(self.divergence(t))

    def divergence(self, t):
        raise NotImplementedError

    def cumulative_divergence(self, t):
        raise NotImplementedError

    @theanify(T.dscalar('s'), T.dscalar('t'), T.dscalar('m'))
    def no_divergence(self, s, t, m):
        return T.exp(self.log_no_divergence(s, t, m))

    def log_no_divergence(self, s, t, m):
        raise NotImplementedError

    def inverse_cumulative(self, y, t1, t2):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    @theanify(T.dscalar('t1'), T.dscalar('t2'), T.dscalar('m'))
    def log_pdf(self, t1, t2, m):
        z = (self.cumulative_divergence(t1) - self.cumulative_divergence(t2)) / m
        p = T.log(self.divergence(t2)) -  T.log(m)
        return z + p

class Inverse(DivergenceFunction):

    @theanify(T.dscalar('t'))
    def divergence(self, t):
        return self.c / (1 - t)

    @theanify(T.dscalar('t'))
    def cumulative_divergence(self, t):
        return -self.c * T.log(1 - t)

    @theanify(T.dscalar('s'), T.dscalar('t'), T.dscalar('m'))
    def log_no_divergence(self, s, t, m):
        return self.cumulative_divergence(s) - self.cumulative_divergence(t) - T.log(m)

    @theanify(T.dscalar('t'), T.dscalar('t1'), T.dscalar('m'))
    def cdf(self, t, t1, m):
        c = float(self.c)
        return (1 / (1 - t1)) ** (c / m) - ((1 - t) / (1 - t1)) ** (c / m)

    @theanify(T.dscalar('t1'), T.dscalar('t2'), T.dscalar('m'))
    def sample(self, t1, t2, m):
        y = RandomStreams().uniform()
        c = float(self.c)
        lower = self.cdf(t1, t1, m)
        upper = self.cdf(t2, t1, m)
        y = lower + y * (upper - lower)
        t = 1-((1-t1)**(c/m)*((1-t1)**(-c/m)-y))**(m/c)
        return t, self.log_pdf(t, t1, m)

    def get_parameters(self):
        return {"c"}
