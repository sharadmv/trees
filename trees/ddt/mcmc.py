import logging
import random
import numpy as np

from .. import MCMCSampler

class MetropolisHastingsSampler(MCMCSampler):

    def __init__(self, ddt, X):
        self.ddt = ddt
        self.X = X
        self.N, self.D = self.X.shape
        self.last_move = None

    def add_constraint(self, constraint):
        self.ddt.add_constraint(constraint)
        a, b, c = constraint
        self.move_point_parent(a, force=True)
        self.move_point_parent(b, force=True)
        self.move_point_parent(c, force=True)

    def initialize_assignments(self):
        self.ddt.initialize_assignments(self.X)

    def sample_node_parent(self):
        ddt = self.ddt.copy()
        node, assignment = ddt.choice(ignore_depth=2)
        assert node is not ddt.root
        assert node.parent is not ddt.root
        index, time = assignment
        index = index[:-1]
        self.parent_move(ddt, node, (index, time))

    def move_point_parent(self, point, force=False):
        ddt = self.ddt.copy()
        node, assignment = ddt.point_index(point)
        assert node is not ddt.root
        assert node.parent is not ddt.root
        index, time = assignment
        index = index[:-1]
        time = node.parent.time
        self.parent_move(ddt, node, (index, time), force=force)

    def parent_move(self, ddt, node, assignment, force=False):
        old_tree_likelihood = self.ddt.marg_log_likelihood()
        parent = node.detach_node()
        points = parent.points()
        backward_prob = ddt.log_prob_assignment(assignment)

        time = float('inf')
        move = None
        try_counter = 0
        while time > parent.left.time:
            (move, time), forward_prob = ddt.sample_assignment(points=points)
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
        if force or np.random.random() < a:
            self.last_move = self.ddt, assignment, (move, time)
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
