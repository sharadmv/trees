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

    @theanify(T.dscalar('y'), T.dscalar('t1'), T.dscalar('t2'))
    def inverse_cumulative(self, y, t1, t2):
        z = self.cumulative_divergence(t2) - self.cumulative_divergence(t1)
        unscaled = (t1 - 1) / (t1 - t2) * (1 - T.exp(-y * z/ self.c))
        return t1 + (t2 - t1) * unscaled

    def get_parameters(self):
        return {"c"}

class InverseQuadratic(DivergenceFunction):

    @theanify(T.dscalar('t'))
    def divergence(self, t):
        return self.b + self.d / (1 - t) ** 2

    @theanify(T.dscalar('t'))
    def cumulative_divergence(self, t):
        return self.b * t - self.d - self.d / (t - 1)

    @theanify(T.dscalar('y'), T.dscalar('t1'), T.dscalar('t2'))
    def inverse_cumulative(self, y, t1, t2):
        z = self.cumulative_divergence(t2) - self.cumulative_divergence(t1)
        if self.b == 0:
            unscaled = (t1 - 1) ** 2 * y  * z/ ((t1 - t2) * (t1 * y * z - self.d - y * z))
            return t1 + (t2 - t1) * unscaled
        b = self.b
        d = self.d
        unscaled = (t1**3*b-t1**2*b*t2-2*t1**2*b-t1**2*y*z+T.sqrt((t1-t2)**2*(t1**4*b**2-4*t1**3*b**2+2*t1**3*b*y*z+6*t1**2*b**2+2*t1**2*b*d-6*t1**2*b*y*z+t1**2*y**2*z**2-4*t1*b**2-4*t1*b*d+6*t1*b*y*z-2*t1*d*y*z-2*t1*y**2*z**2+b**2+2*b*d-2*b*y*z+d**2+2*d*y*z+y**2*z**2))+2*t1*b*t2+t1*b+t1*t2*y*z+t1*d+t1*y*z-b*t2-t2*d-t2*y*z)/(2*(t1-1)*b*(t1-t2)**2)
        return t1 + (t2 - t1) * unscaled

    def get_parameters(self):
        return {"b", "d"}
