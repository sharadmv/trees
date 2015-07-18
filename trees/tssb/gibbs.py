import logging
import scipy.stats as stats
import numpy as np
from tqdm import tqdm

class GibbsSampler(object):

    def __init__(self, tssb, parameter_process, X):
        self.tssb = tssb
        self.parameter_process = parameter_process
        self.X = X
        self.N, self.D = self.X.shape

    def initialize_assignments(self):
        for i in xrange(self.N):
            index = self.tssb.sample_one()
            self.tssb.add_point(i, index)

    def log_likelihood(self, i):
        return self.parameter_process.log_likelihood(self.X[i], self.tssb[self.tssb.points[i]].parameter)

    def sample_assignments(self):
        idx = np.arange(self.N)
        np.random.shuffle(idx)
        for i in tqdm(idx):
            self.sample_assignment(i)

    def sample_parameters(self):
        for index in tqdm(self.tssb.nodes):
            self.sample_parameter(index)

    def gibbs_sample(self):
        logging.info("Starting Gibbs sampling iteration...")
        logging.info("Sampling assignments...")
        self.sample_assignments()
        logging.info("Sampling stick sizes...")
        self.sample_sticks()
        logging.info("Applying size-biased permutation...")
        self.size_biased_permutation()
        logging.info("Sampling parameters...")
        self.sample_parameters()

    def sample_parameter(self, index):
        assert index in self.tssb.nodes
        node = self.tssb[index]
        data = list(node.points)
        parent = None
        if index != ():
            parent = self.tssb[index[:-1]].parameter
        children = []
        for child in node.children:
            children.append(self.tssb[index + (child,)].parameter)
        children = np.array(children)
        node.parameter = self.parameter_process.sample_posterior(self.X[data], children, parent)

    def sample_assignment(self, i):
        assert self.tssb.points[i] in self.tssb.nodes, '%u' % i
        log_likelihood = np.exp(self.log_likelihood(i))
        old_assignment = self.tssb.points[i]
        self.tssb.remove_point(i)
        p_slice = np.log(np.random.uniform(low=0, high=log_likelihood))
        u_min, u_max = 0, 1

        assignment = None

        while assignment is None:
            u = np.random.uniform(low=u_min, high=u_max)
            e, updates = self.tssb.uniform_index(u, return_updates=True)
            p = self.parameter_process.log_likelihood(self.X[i], updates[e].parameter)

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
