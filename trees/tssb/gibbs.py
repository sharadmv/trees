import scipy.stats as stats
import numpy as np

class GibbsSampler(object):

    def __init__(self, tssb, parameters, X):
        self.tssb = tssb
        self.parameters = parameters
        self.X = X
        self.N, self.D = self.X.shape

    def initialize_assignments(self):
        for i in xrange(self.N):
            index = self.tssb.sample_one()
            self.tssb.add_point(i, index)

    def log_likelihood(self, i):
        return self.parameters.log_likelihood(self.X[i], self.tssb.points[i])

    def sample_assignment(self, i):
        log_likelihood = np.exp(self.log_likelihood(i))
        old_assignment = self.tssb.points[i]
        self.tssb.remove_point(i)
        p_slice = np.log(np.random.uniform(low=0, high=log_likelihood))
        u_min, u_max = 0, 1

        assignment = None

        while assignment is None:
            u = np.random.uniform(low=u_min, high=u_max)
            e, updates = self.tssb.uniform_index(u, return_updates=True)
            p = self.parameters.log_likelihood(self.X[i], e)

            if p > p_slice:
                assignment = e
                self.tssb.apply_node_updates(updates)
                self.tssb.add_point(i, assignment)
            elif e < old_assignment:
                u_min = u
            else:
                u_max = u
        return assignment

    def size_biased_permutation(self):
        indices = sorted(self.tssb.nodes.keys(), key=lambda x: len(x))[::-1]
        for index in indices:
            self.tssb.size_biased_permutation(index)

    def sample_sticks(self):
        for index, node in self.tssb.nodes.items():
            node.nu = stats.beta(node.point_count + 1, node.path_count + node.alpha).rvs()
            children = sorted(list(node.children.keys()))[::-1]
            count = 0
            for i in children:
                child = self.tssb[index + (i,)]
                node.psi[i] = stats.beta(child.path_count + 1, count + node.gamma).rvs()
                count += child.path_count
