import random
import numpy as np

from .. import MCMCSampler
from node import Leaf, NonTerminal

class MetropolisHastingsSampler(MCMCSampler):

    def __init__(self, ddt, X):
        self.ddt = ddt
        self.X = X
        self.N, self.D = self.X.shape

    def initialize_assignments(self):
        nodes = [Leaf(None, i, self.X[i]) for i in xrange(self.N)]
        while len(nodes) > 1:
            random.shuffle(nodes)
            (a, b), rest = nodes[:2], nodes[2:]
            merge_time = random.uniform(0, min(a.time, b.time))
            node = NonTerminal(a, b, None, (a.state + b.state) / 2.0, merge_time)
            a.parent = node
            b.parent = node
            nodes = [node] + rest
        self.ddt.root = nodes[0]
        self.ddt.root.time = 0
        self.ddt.root.state = self.ddt.likelihood_model.mu0

    def sample_node_parent(self):
        ddt = self.ddt.copy()
        node = ddt.choice(ignore_depth=2)
        self.parent_move(ddt, node)

    def parent_move(self, ddt, node):
        old_tree_likelihood = self.ddt.log_likelihood()
        parent, grandparent = node.detach_node()

        grandparent.attach_node(parent, self.ddt.df)

    def update_latent(self):
        self.ddt.update_latent()

    def gibbs_sample(self):
        self.sample_node_parent()
        self.ddt.root.verify_times()
        self.update_latent()
