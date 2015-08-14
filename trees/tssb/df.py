from ..distribution import Distribution
import theano.tensor as T
from theanify import Theanifiable, theanify

class DepthFunction(Theanifiable, Distribution):

    def __init__(self, **parameters):
        Theanifiable.__init__(self)
        Distribution.__init__(self, parameters)
        self.compile()

    def __getattr__(self, key):
        if key in self.parameters:
            return self.parameters[key]

class QuadraticDepth(DepthFunction):

    @theanify(T.dscalar('depth'))
    def alpha(self, depth):
        return self.l ** depth * self.a

    def get_parameters(self):
        return {"a", "l"}
