import logging
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
        nodes = [self.ddt.leaf(None, i, self.X[i]) for i in xrange(self.N)]
        while len(nodes) > 1:
            random.shuffle(nodes)
            (a, b), rest = nodes[:2], nodes[2:]
            merge_time = random.uniform(0, min(a.time, b.time))
            node = self.ddt.non_terminal(a, b, None, (a.state + b.state) / 2.0, merge_time)
            a.parent = node
            b.parent = node
            nodes = [node] + rest
        self.ddt.root = nodes[0]
        self.ddt.root.time = 0
        self.ddt.root.state = self.ddt.likelihood_model.mu0

    def sample_node_parent(self):
        ddt = self.ddt.copy()
        node, assignment = ddt.choice(ignore_depth=2)
        assert node is not ddt.root
        assert node.parent is not ddt.root
        index, time = assignment
        index = index[:-1]
        self.parent_move(ddt, node, (index, time))

    def parent_move(self, ddt, node, assignment):
        old_tree_likelihood = self.ddt.marg_log_likelihood()
        parent = node.detach_node()
        backward_prob = ddt.log_prob_assignment(assignment)

        time = float('inf')
        move = None
        try_counter = 0
        while time > parent.left.time:
            (move, time), forward_prob = ddt.sample_assignment()
            try_counter += 1
            if try_counter > 500:
                return
        ddt.attach_node(parent, (move, time))
        new_tree_likelihood = ddt.marg_log_likelihood()
        logging.debug("Move: %s" % str((move, time)))
        logging.debug("Forward: %f" % forward_prob)
        logging.debug("Backward: %f" % backward_prob)
        logging.debug("Probs: (%f, %f)" % (old_tree_likelihood, new_tree_likelihood))
        a = min(1, np.exp(new_tree_likelihood + backward_prob - old_tree_likelihood - forward_prob))
        logging.debug("Accept Probability: %f" % a)
        if np.random.random() < a:
            self.ddt = ddt

    def update_latent(self):
        ddt = self.ddt.copy()
        ddt.update_latent()
        self.ddt = ddt

    def sample(self):
        if np.random.random() < 0.5:
            self.sample_node_parent()
        else:
            self.update_latent()
