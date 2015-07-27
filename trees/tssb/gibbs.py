import logging
import scipy.stats as stats
import numpy as np

class GibbsSampler(object):

    def __init__(self, tssb, parameter_process, X):
        self.tssb = tssb
        self.parameter_process = parameter_process
        self.X = X
        self.N, self.D = self.X.shape

    def initialize_assignments(self):
        for i in xrange(self.N):
            _, index = self.tssb.sample_one(point=i)
            self.tssb.add_point(i, index)
        self.tssb.garbage_collect()

    def log_likelihood(self, i, parameter):
        return self.parameter_process.data_log_likelihood(self.X[i], parameter)


    def sample_assignments(self):
        idx = np.arange(self.N)
        np.random.shuffle(idx)
        for i in idx:
            self.sample_assignment(i)

    def sample_parameters(self):
        for node in self.tssb.dfs():
            self.sample_parameter(node)

    def gibbs_sample(self):
        logging.debug("Starting Gibbs sampling iteration...")
        logging.debug("Sampling assignments...")
        self.sample_assignments()
        logging.debug("Sampling stick sizes...")
        self.sample_sticks()
        logging.debug("Applying size-biased permutation...")
        self.size_biased_permutation()
        logging.debug("Sampling parameters...")
        self.sample_parameters()

    def sample_parameter(self, node):
        data = list(node.points)
        parent = None
        if node != self.tssb.root:
            parent = node.parent.parameter
        children = []
        for _, child_node in node.children.items():
            children.append(child_node.parameter)
        children = np.array(children)
        node.parameter = self.parameter_process.sample_posterior(self.X[data], children, parent)


    def sample_assignment(self, i):
        logging.debug("Sampling assignment for %u" % i)
        node, index = self.tssb.point_index(i)
        log_likelihood = np.exp(self.log_likelihood(i, node.parameter))
        old_assignment = index
        self.tssb.remove_point(i)
        p_slice = np.log(np.random.uniform(low=0, high=log_likelihood))
        u_min, u_max = 0, 1

        assignment = None

        while assignment is None:

            if np.isclose(u_min, u_max):
                assignment = index
                continue

            u = np.random.uniform(low=u_min, high=u_max)
            candidate_node, candidate_index = self.tssb.uniform_index(u, point=i)
            p = self.log_likelihood(i, candidate_node.parameter)

            if p > p_slice:
                assignment = candidate_index
            elif candidate_index < old_assignment:
                u_min = u
            else:
                u_max = u
        self.tssb.add_point(i, assignment)
        self.tssb.garbage_collect()
        return assignment

    def size_biased_permutation(self):
        nodes = list(self.tssb.dfs())
        for node in nodes:
            node.size_biased_permutation()


    def sample_sticks(self):
        for node in self.tssb.dfs():
            node.nu = stats.beta(node.point_count + 1, node.path_count + node.alpha).rvs()
            children = sorted(list(node.children.keys()))[::-1]
            count = 0
            for i in children:
                child = node.children[i]
                node.psi[i] = stats.beta(child.path_count + 1, count + node.gamma).rvs()
                count += child.path_count
