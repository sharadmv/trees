"""
This file defines the global tree and MCMC interface.
"""
class Tree(object):

    def copy(self):
        raise NotImplementedError

    def marg_log_likelihood(self):
        raise NotImplementedError

    def get_node(self, index):
        raise NotImplementedError

    def point_index(self, i):
        raise NotImplementedError

class MCMCSampler(object):

    def sample(self):
        raise NotImplementedError
